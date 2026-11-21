// DSpine code-bridge and serial-refinement tail kernels. BF16 storage, FP32 arithmetic,
// BF16 rounding exactly at the reference rounding points (dspine_mlx/bridge.py, batched_tail.py).

#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/numeric/convert.cuh"
#include "ops/common/quant/nvfp4.cuh"
#include "ninfer/ops/sampling.h"
#include "ops/sampling/kernels/random.cuh"
#include "ops/dspine_bridge/launch.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <cfloat>
#include <mutex>

namespace ninfer::ops {
namespace {

constexpr int kBlock = 256;
constexpr int kTopK  = 16;

__device__ __forceinline__ float bf16f(__nv_bfloat16 v) { return __bfloat162float(v); }

__device__ __forceinline__ __nv_bfloat16 fbf16(float v) { return __float2bfloat16_rn(v); }

__device__ __forceinline__ float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

// A vocabulary-indexed bridge table (codebook/tg/tsv). BF16: data is the row-major BF16 matrix,
// scales null. NVFP4: data is the packed E2M1 code plane, scales the m128x4 E4M3 plane, divisor
// the parent weight divisor. Rows are gathered by token id and dequantized to BF16 on access.
struct TableRef {
    const void* data;
    const std::uint8_t* scales;
    float divisor;
};

// The proposal head: the tower's own output head, which the drafter reuses instead of carrying
// a copy of it. It is gathered by candidate id and dequantized on access, so the head exists
// exactly once - in the representation the tower declares. Two representations are live:
// FP8_E4M3FN_ROW_BF16 (one E4M3 byte per element, one BF16 scale per row) and Q8_G32_FP16 (one
// signed byte per element at the padded row stride, one FP16 scale per 32-column group - the
// addressing of ops/embedding/kernel.cuh::embed_gather_q8_kernel, the in-tree gather of this
// same format). `row_stride` is the code elements per row (the padded columns) and
// `scale_groups` the scale entries per row (1 for a row-scaled head).
struct HeadRef {
    const std::uint8_t* codes;
    const std::uint8_t* scales;  // BF16 row scales, or the FP16 group-scale plane
    std::int32_t row_stride;
    std::int32_t scale_groups;
};

__device__ __forceinline__ float head_row_scale(const HeadRef& head, std::int32_t row) {
    return bf16f(static_cast<const __nv_bfloat16*>(static_cast<const void*>(head.scales))[row]);
}

__device__ __forceinline__ float head_group_scale(const HeadRef& head, std::int64_t group) {
    return common::load_fp16_unaligned(head.scales + group * 2);
}

__device__ __forceinline__ __nv_bfloat16 table_elem_bf16(const TableRef& table, bool nvfp4,
                                                         int row, int k, int c) {
    if (nvfp4) {
        const auto* codes = static_cast<const std::uint8_t*>(table.data);
        const float scale =
            common::decode_e4m3(table.scales[common::nvfp4_scale_byte_offset(row, c >> 4, k)]);
        const float2 v =
            common::decode_nvfp4_e2m1x2(codes[static_cast<std::int64_t>(row) * (k >> 1) + (c >> 1)]);
        return fbf16(((c & 1) ? v.y : v.x) * scale / table.divisor);
    }
    return static_cast<const __nv_bfloat16*>(table.data)[static_cast<std::int64_t>(row) * k + c];
}

__launch_bounds__(kBlock) __global__ void prepare_read_kernel(const __nv_bfloat16* residual,
                                                            float eps, __nv_bfloat16* normalized,
                                                            float* rms, std::int32_t hidden,
                                                            std::int64_t columns) {
    const std::int64_t column = blockIdx.x;
    if (column >= columns) { return; }
    const std::int64_t base = column * hidden;
    float sum               = 0.0f;
    for (std::int32_t h = threadIdx.x; h < hidden; h += kBlock) {
        const float f = bf16f(residual[base + h]);
        sum += f * f;
    }
    __shared__ float warp_sums[kBlock / common::kWarpSize];
    const float total = common::block_reduce_sum<kBlock>(sum, warp_sums);
    __shared__ float rms_shared;
    if (threadIdx.x == 0) {
        rms_shared = sqrtf(fmaxf(total / static_cast<float>(hidden), FLT_MIN));
        rms[column] = rms_shared;
    }
    __syncthreads();
    const float denom = rms_shared + eps;
    for (std::int32_t h = threadIdx.x; h < hidden; h += kBlock) {
        normalized[base + h] = fbf16(bf16f(residual[base + h]) / denom);
    }
}

template <bool Nvfp4>
__launch_bounds__(kBlock) __global__ void normalize_codes_kernel(
    const __nv_bfloat16* projected, const std::int32_t* anchors, TableRef codebook,
    std::int32_t width, std::int32_t code_size, std::int32_t vocab, float sqrt_c, float eps,
    __nv_bfloat16* codes, std::int64_t columns) {
    const std::int64_t column = blockIdx.x;
    if (column >= columns) { return; }
    const std::int64_t base = column * code_size;
    const std::int32_t t    = static_cast<std::int32_t>(column % width);
    if (t == 0) {
        const std::int32_t anchor = anchors[column / width];
        for (std::int32_t c = threadIdx.x; c < code_size; c += kBlock) {
            codes[base + c] =
                fbf16(sqrt_c * bf16f(table_elem_bf16(codebook, Nvfp4, anchor, code_size, c)));
        }
        return;
    }
    float sum = 0.0f;
    for (std::int32_t c = threadIdx.x; c < code_size; c += kBlock) {
        const float p = bf16f(projected[base + c]);
        sum += p * p;
    }
    __shared__ float warp_sums[kBlock / common::kWarpSize];
    const float total = common::block_reduce_sum<kBlock>(sum, warp_sums);
    __shared__ float denom_shared;
    if (threadIdx.x == 0) { denom_shared = sqrtf(total) + eps; }
    __syncthreads();
    const float scale = sqrt_c / denom_shared;
    for (std::int32_t c = threadIdx.x; c < code_size; c += kBlock) {
        codes[base + c] = fbf16(bf16f(projected[base + c]) * scale);
    }
}

__launch_bounds__(kBlock) __global__ void message_kernel(const __nv_bfloat16* packed,
                                                       const __nv_bfloat16* gate_bias,
                                                       __nv_bfloat16* message, std::int32_t rank,
                                                       std::int32_t width, std::int64_t columns) {
    const std::int64_t total = static_cast<std::int64_t>(rank) * columns;
    for (std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * kBlock + threadIdx.x;
         index < total; index += static_cast<std::int64_t>(gridDim.x) * kBlock) {
        const std::int32_t r = static_cast<std::int32_t>(index % rank);
        const std::int64_t c = index / rank;
        const std::int32_t t = static_cast<std::int32_t>(c % width);
        if (t == 0) {
            message[c * rank + r] = fbf16(0.0f);
            continue;
        }
        const std::int64_t prev = (c - 1) * (3 * rank);
        const std::int64_t cur  = c * (3 * rank);
        const float q           = bf16f(packed[cur + r]);
        const float s           = bf16f(packed[prev + rank + r]);
        const float v           = bf16f(packed[prev + 2 * rank + r]);
        const float gate        = sigmoidf(q + s + bf16f(gate_bias[r]));
        message[c * rank + r]   = fbf16(gate * v);
    }
}

__launch_bounds__(kBlock) __global__ void finish_kernel(
    __nv_bfloat16* residual, const float* rms, const __nv_bfloat16* delta,
    const __nv_bfloat16* next_norm, float eps, __nv_bfloat16* normed, __nv_bfloat16* stash_raw,
    std::int32_t hidden, std::int32_t width, std::int64_t columns) {
    const std::int64_t column = blockIdx.x;
    if (column >= columns) { return; }
    const std::int64_t base = column * hidden;
    const float refine      = (column % width != 0) ? rms[column] : 0.0f;
    float sum               = 0.0f;
    for (std::int32_t h = threadIdx.x; h < hidden; h += kBlock) {
        const std::int64_t index = base + h;
        const float f            = bf16f(residual[index]);
        if (stash_raw) { stash_raw[index] = residual[index]; }
        const float out          = f + refine * bf16f(delta[index]);
        const __nv_bfloat16 ob   = fbf16(out);
        residual[index]          = ob;
        const float outr         = bf16f(ob);
        sum += outr * outr;
    }
    __shared__ float warp_sums[kBlock / common::kWarpSize];
    const float total = common::block_reduce_sum<kBlock>(sum, warp_sums);
    __shared__ float inv_shared;
    if (threadIdx.x == 0) {
        inv_shared = rsqrtf(total / static_cast<float>(hidden) + eps);
    }
    __syncthreads();
    const float inv = inv_shared;
    for (std::int32_t h = threadIdx.x; h < hidden; h += kBlock) {
        const std::int64_t index = base + h;
        const float y            = bf16f(fbf16(bf16f(residual[index]) * inv));
        normed[index]            = fbf16(y * bf16f(next_norm[h]));
    }
}

// One CTA per (batch, proposal row). Shared budget ~86 KB for R=512, K=16.
//
// `scores`, when non-null, receives the whole K x K score matrix of this (batch, proposal row)
// as FP32 [batch, candidate, sender, drafts] (first axis fastest), which is the table
// ops::dspine_serial_tail consumes. It is the same score_s the argmax collapses, written out
// untouched, so emitting it cannot move the numerics of `idx`.
template <int K, int Block, bool Nvfp4, bool HeadQ8>
__launch_bounds__(Block) __global__ void serial_scores_kernel(
    const std::int32_t* cand, const __nv_bfloat16* raw, const float* rms,
    const __nv_bfloat16* packed_q, const std::int32_t* anchors, const __nv_bfloat16* norm_w,
    const __nv_bfloat16* gate_bias, HeadRef head, TableRef tg, TableRef tsv, std::int32_t hidden,
    std::int32_t rank, std::int32_t width, std::int32_t drafts, std::int32_t batch,
    std::int32_t* idx, float* scores) {
    const std::int32_t b = blockIdx.x / drafts;
    const std::int32_t i = blockIdx.x % drafts;
    if (b >= batch) { return; }

    extern __shared__ char smem[];
    __nv_bfloat16* tg_s  = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* tsv_s = tg_s + K * rank;
    float* m_s           = reinterpret_cast<float*>(tsv_s + K * 2 * rank);
    float* q_s           = m_s + K * rank;
    float* bias_s        = q_s + rank;
    float* base_s        = bias_s + rank;
    float* partial_s     = base_s + K;
    float* score_s       = partial_s + K * (Block / K);
    int* cand_s          = reinterpret_cast<int*>(score_s + K * K);
    int* sid_s           = cand_s + K;
    float* rms_s         = reinterpret_cast<float*>(sid_s + K);

    const std::int64_t row_base = static_cast<std::int64_t>(b) * width + 1 + i;
    const std::int32_t tid      = threadIdx.x;

    if (tid < K) {
        cand_s[tid] = cand[(static_cast<std::int64_t>(b) * drafts + i) * K + tid];
        sid_s[tid]  = (i == 0)
                          ? anchors[b]
                          : cand[(static_cast<std::int64_t>(b) * drafts + i - 1) * K + tid];
    }
    if (tid == 0) { *rms_s = rms[row_base]; }
    __syncthreads();

    for (int index = tid; index < K * rank; index += Block) {
        const int k = index / rank;
        const int r = index % rank;
        tg_s[k * rank + r] = table_elem_bf16(tg, Nvfp4, cand_s[k], rank, r);
    }
    for (int index = tid; index < K * 2 * rank; index += Block) {
        const int k = index / (2 * rank);
        const int r = index % (2 * rank);
        tsv_s[k * 2 * rank + r] = table_elem_bf16(tsv, Nvfp4, sid_s[k], 2 * rank, r);
    }
    for (int r = tid; r < rank; r += Block) {
        q_s[r]    = bf16f(packed_q[row_base * rank + r]);
        bias_s[r] = bf16f(gate_bias[r]);
    }
    __syncthreads();

    // base_k = dot(raw * norm_w, dequant(head[cand_k])), streamed over H. The head row is
    // gathered by candidate id and dequantized element by element; no copy of the head exists.
    {
        const int k     = tid % K;
        const int group = tid / K;
        const std::int64_t raw_base = row_base * hidden;
        const std::int64_t head_base =
            static_cast<std::int64_t>(cand_s[k]) * head.row_stride;
        const std::int64_t scale_base =
            HeadQ8 ? static_cast<std::int64_t>(cand_s[k]) * head.scale_groups : 0;
        const float row_scale = HeadQ8 ? 0.0f : head_row_scale(head, cand_s[k]);
        float acc             = 0.0f;
        for (int h = group; h < hidden; h += Block / K) {
            float code;
            if constexpr (HeadQ8) {
                const float scale = head_group_scale(head, scale_base + (h >> 5));
                const auto  code8 = static_cast<std::int8_t>(head.codes[head_base + h]);
                code              = bf16f(fbf16(static_cast<float>(code8) * scale));
            } else {
                code = bf16f(fbf16(common::decode_e4m3(head.codes[head_base + h]) * row_scale));
            }
            acc += bf16f(raw[raw_base + h]) * bf16f(norm_w[h]) * code;
        }
        partial_s[k * (Block / K) + group] = acc;
    }
    __syncthreads();
    if (tid < K) {
        float acc = 0.0f;
        for (int g = 0; g < Block / K; ++g) { acc += partial_s[tid * (Block / K) + g]; }
        base_s[tid] = acc;
    }
    __syncthreads();

    for (int index = tid; index < K * rank; index += Block) {
        const int j = index / rank;
        const int r = index % rank;
        const float tvg = bf16f(tsv_s[j * 2 * rank + r]);
        const float tvv = bf16f(tsv_s[j * 2 * rank + rank + r]);
        m_s[j * rank + r] = sigmoidf(q_s[r] + tvg + bias_s[r]) * tvv;
    }
    __syncthreads();

    {
        const int j = tid / K;
        const int k = tid % K;
        float dot   = 0.0f;
        for (int r = 0; r < rank; ++r) {
            dot += m_s[j * rank + r] * bf16f(tg_s[k * rank + r]);
        }
        score_s[j * K + k] = base_s[k] + (*rms_s) * dot;
    }
    __syncthreads();
    if (scores != nullptr) {
        for (int index = tid; index < K * K; index += Block) {
            const int j = index / K;
            const int k = index % K;
            scores[(((static_cast<std::int64_t>(i) * K + j) * K + k) * batch) + b] =
                score_s[j * K + k];
        }
    }
    if (tid < K) {
        int best      = 0;
        float best_v  = score_s[tid * K];
        for (int k = 1; k < K; ++k) {
            const float v = score_s[tid * K + k];
            if (v > best_v) {
                best_v = v;
                best   = k;
            }
        }
        idx[(static_cast<std::int64_t>(b) * drafts + i) * K + tid] = best;
    }
}

__launch_bounds__(32) __global__ void serial_chain_kernel(const std::int32_t* cand,
                                                        const std::int32_t* idx,
                                                        std::int32_t* drafts_out,
                                                        std::int32_t drafts, std::int32_t batch) {
    const std::int32_t b = blockIdx.x * 32 + threadIdx.x;
    if (b >= batch) { return; }
    const std::int64_t base = static_cast<std::int64_t>(b) * drafts;
    int j                   = idx[base * kTopK];
    for (int i = 0; i < drafts; ++i) {
        drafts_out[base + i] = cand[(base + i) * kTopK + j];
        if (i + 1 < drafts) { j = idx[(base + i + 1) * kTopK + j]; }
    }
}

// --- serial tail (policy sampling) -------------------------------------------------------
//
// Mirror of models/common/dspine/dspine_serial_tail.cpp, which is the executable
// specification and the CPU-tested oracle. One block per batch row: each of the first kTopK
// threads owns one sender and filters that sender's row; thread 0 walks the chain serially.
// No allocation, no sync with the host, no dependence on the compact row index - capturable.

// propose.py:73-90. Rank on the distribution (lower slot wins ties), keep while
// (rank < top_k) and (strict prefix mass <= top_p) and (p >= min_p * max p), renormalise.
__device__ __forceinline__ void tail_filter(const float* scores, const ops::SamplingConfig& config,
                                            float* q, bool* keep) {
    if (!(config.temperature > 0.0f)) {
        int best = 0;
        for (int k = 1; k < kTopK; ++k) {
            if (scores[k] > scores[best]) { best = k; }
        }
        for (int k = 0; k < kTopK; ++k) {
            keep[k] = k == best;
            q[k]    = keep[k] ? 1.0f : 0.0f;
        }
        return;
    }
    const float inv_temp = 1.0f / config.temperature;
    float peak           = scores[0] * inv_temp;
    for (int k = 1; k < kTopK; ++k) { peak = fmaxf(peak, scores[k] * inv_temp); }
    float p[kTopK];
    float total = 0.0f;
    for (int k = 0; k < kTopK; ++k) {
        p[k] = expf(scores[k] * inv_temp - peak);
        total += p[k];
    }
    for (int k = 0; k < kTopK; ++k) { p[k] /= total; }
    float maxima = p[0];
    for (int k = 1; k < kTopK; ++k) { maxima = fmaxf(maxima, p[k]); }
    float kept_mass = 0.0f;
    for (int k = 0; k < kTopK; ++k) {
        int rank          = 0;
        float mass_before = 0.0f;
        for (int j = 0; j < kTopK; ++j) {
            const bool is_ahead = p[j] > p[k] || (p[j] == p[k] && j < k);
            if (!is_ahead) { continue; }
            ++rank;
            mass_before += p[j];
        }
        bool keep_slot = rank < config.top_k && mass_before <= config.top_p;
        if (config.min_p > 0.0f) { keep_slot = keep_slot && p[k] >= config.min_p * maxima; }
        keep[k] = keep_slot;
        q[k]    = keep_slot ? p[k] : 0.0f;
        kept_mass += q[k];
    }
    for (int k = 0; k < kTopK; ++k) { q[k] /= kept_mass; }
}

// propose.py:150-152: choice = #{cdf <= u * cdf[last]}, clamped to the last kept slot.
__device__ __forceinline__ int tail_draw(const float* q, const bool* keep, float u) {
    float cdf[kTopK];
    float running = 0.0f;
    for (int k = 0; k < kTopK; ++k) {
        running += q[k];
        cdf[k] = running;
    }
    const float threshold = u * cdf[kTopK - 1];
    int choice            = 0;
    for (int k = 0; k < kTopK; ++k) {
        if (cdf[k] <= threshold) { ++choice; }
    }
    int last = 0;
    for (int k = 0; k < kTopK; ++k) {
        if (keep[k]) { last = k; }
    }
    return choice < last ? choice : last;
}

// Layouts (first axis fastest, as everywhere in this file):
//   table   FP32 [batch, kTopK(cand), kTopK(sender), drafts]
//   cand    I32  [batch, drafts, kTopK]
//   drafts  I32  [batch, drafts]
//   q_out   FP32 [batch, drafts, kTopK]  (the walked sender's row)
//   idx_out I32  [batch, drafts, kTopK]  (per sender)
__launch_bounds__(32) __global__ void serial_tail_kernel(
    const float* table, const std::int32_t* cand, const ops::SamplingConfig* configs,
    const std::int32_t* positions, std::int32_t* drafts_out, float* q_out, std::int32_t* idx_out,
    std::int32_t drafts, std::int32_t batch) {
    const int b = blockIdx.x;
    if (b >= batch) { return; }
    const ops::SamplingConfig config = configs[b];

    __shared__ float q_shared[kTopK * kTopK];
    __shared__ int idx_shared[kTopK];
    __shared__ int sender_shared;
    if (threadIdx.x == 0) { sender_shared = 0; }
    __syncthreads();

    for (int i = 0; i < drafts; ++i) {
        const int slot = threadIdx.x;
        if (slot < kTopK) {
            float scores[kTopK];
            for (int k = 0; k < kTopK; ++k) {
                scores[k] = table[b + batch * (k + kTopK * (slot + kTopK * i))];
            }
            float q[kTopK];
            bool keep[kTopK];
            tail_filter(scores, config, q, keep);
            int index = 0;
            if (!(config.temperature > 0.0f)) {
                for (int k = 1; k < kTopK; ++k) {
                    if (keep[k]) { index = k; }
                }
            } else {
                const float u =
                    // Same purpose the host core uses (models/common/dspine/
                    // dspine_serial_tail.h kSerialTailPurpose == this enum value); the GPU test
                    // asserts the two agree.
                    sampling_detail::sampling_uniform(config.seed, positions[b],
                                                      kSamplePurposeDFlash2Proposal,
                                                      static_cast<unsigned int>(i));
                index = tail_draw(q, keep, u);
            }
            for (int k = 0; k < kTopK; ++k) { q_shared[slot * kTopK + k] = q[k]; }
            idx_shared[slot] = index;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            const int sender = sender_shared;
            const int choice = idx_shared[sender];
            drafts_out[b + batch * i] = cand[b + batch * (i + drafts * choice)];
            for (int k = 0; k < kTopK; ++k) {
                q_out[b + batch * (i + drafts * k)] = q_shared[sender * kTopK + k];
            }
            sender_shared = choice;
        }
        for (int sender = threadIdx.x; sender < kTopK; sender += blockDim.x) {
            idx_out[b + batch * (i + drafts * sender)] = idx_shared[sender];
        }
        __syncthreads();
    }
}

} // namespace

namespace detail {

namespace {
TableRef table_ref(const Weight& weight) {
    if (weight.qtype == QType::NVFP4) {
        return {weight.qdata, static_cast<const std::uint8_t*>(weight.scales),
                weight.weight_scale};
    }
    return {weight.qdata, nullptr, 0.0F};
}
} // namespace

void dspine_prepare_read_launch(const Tensor& residual, float eps, Tensor& normalized, Tensor& rms,
                                cudaStream_t stream) {
    const std::int32_t hidden = residual.ne[0];
    const std::int64_t columns = residual.numel() / hidden;
    prepare_read_kernel<<<static_cast<unsigned>(columns), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(residual.data), eps,
        static_cast<__nv_bfloat16*>(normalized.data), static_cast<float*>(rms.data), hidden, columns);
}

void dspine_normalize_codes_launch(const Tensor& projected, const Tensor& anchors,
                                   const Weight& codebook, std::int32_t width, float eps,
                                   Tensor& codes, cudaStream_t stream) {
    const std::int32_t code_size = projected.ne[0];
    const std::int32_t vocab     = codebook.n;
    const std::int64_t columns   = projected.numel() / code_size;
    const float sqrt_c           = sqrtf(static_cast<float>(code_size));
    const TableRef cb            = table_ref(codebook);
    if (codebook.qtype == QType::NVFP4) {
        normalize_codes_kernel<true><<<static_cast<unsigned>(columns), kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(projected.data),
            static_cast<const std::int32_t*>(anchors.data), cb, width, code_size, vocab, sqrt_c, eps,
            static_cast<__nv_bfloat16*>(codes.data), columns);
    } else {
        normalize_codes_kernel<false><<<static_cast<unsigned>(columns), kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(projected.data),
            static_cast<const std::int32_t*>(anchors.data), cb, width, code_size, vocab, sqrt_c, eps,
            static_cast<__nv_bfloat16*>(codes.data), columns);
    }
}

void dspine_message_launch(const Tensor& packed, const Tensor& gate_bias, Tensor& message,
                           cudaStream_t stream) {
    const std::int32_t rank    = packed.ne[0] / 3;
    const std::int32_t width   = packed.ne[1];
    const std::int64_t columns = packed.numel() / packed.ne[0];
    const std::int64_t total   = static_cast<std::int64_t>(rank) * columns;
    const unsigned blocks =
        static_cast<unsigned>((total + kBlock - 1) / kBlock);
    message_kernel<<<blocks, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(packed.data),
        static_cast<const __nv_bfloat16*>(gate_bias.data),
        static_cast<__nv_bfloat16*>(message.data), rank, width, columns);
}

void dspine_finish_launch(Tensor& residual, const Tensor& rms, const Tensor& delta,
                          const Tensor& next_norm, float eps, Tensor& normed, Tensor* stash_raw,
                          cudaStream_t stream) {
    const std::int32_t hidden  = residual.ne[0];
    const std::int32_t width   = residual.ne[1];
    const std::int64_t columns = residual.numel() / hidden;
    finish_kernel<<<static_cast<unsigned>(columns), kBlock, 0, stream>>>(
        static_cast<__nv_bfloat16*>(residual.data), static_cast<const float*>(rms.data),
        static_cast<const __nv_bfloat16*>(delta.data),
        static_cast<const __nv_bfloat16*>(next_norm.data), eps,
        static_cast<__nv_bfloat16*>(normed.data),
        stash_raw ? static_cast<__nv_bfloat16*>(stash_raw->data) : nullptr, hidden, width, columns);
}

namespace {
// The head planes the kernel gathers, in the representation the artifact declares. The gates of
// the two accepted representations are in dspine_bridge.cpp; the strides follow from them.
HeadRef head_ref(const Weight& head) {
    HeadRef ref;
    ref.codes  = static_cast<const std::uint8_t*>(head.qdata);
    ref.scales = static_cast<const std::uint8_t*>(head.scales);
    if (head.qtype == QType::Q8_G32_FP16) {
        ref.row_stride   = head.padded_shape[1];
        ref.scale_groups = head.padded_shape[1] / 32;
    } else {
        ref.row_stride   = head.k;
        ref.scale_groups = 1;
    }
    return ref;
}
} // namespace

void dspine_serial_scores_launch(const Tensor& cand, const Tensor& raw, const Tensor& rms,
                                 const Tensor& packed_q, const Tensor& anchors,
                                 const Tensor& norm_w, const Tensor& gate_bias, const Weight& head,
                                 const Weight& tg, const Weight& tsv, std::int32_t width,
                                 Tensor& idx, Tensor* scores, cudaStream_t stream) {
    const std::int32_t hidden = raw.ne[0];
    const std::int32_t rank   = packed_q.ne[0];
    const std::int32_t drafts = cand.ne[1];
    const std::int32_t batch  = cand.ne[2];
    const std::size_t shared =
        static_cast<std::size_t>(kTopK) * rank * 2 +          // tg bf16
        static_cast<std::size_t>(kTopK) * 2 * rank * 2 +      // tsv bf16
        static_cast<std::size_t>(kTopK) * rank * 4 +          // m fp32
        static_cast<std::size_t>(rank) * 4 * 2 +              // q, bias
        static_cast<std::size_t>(kTopK) * 4 +                 // base
        static_cast<std::size_t>(kTopK) * (kBlock / kTopK) * 4 +  // partial
        static_cast<std::size_t>(kTopK) * kTopK * 4 +         // score
        static_cast<std::size_t>(kTopK) * 4 * 2 +             // cand, sid
        4;                                                    // rms
    static std::once_flag once;
    std::call_once(once, [&] {
        const auto admit = [&](auto kernel) {
            CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(shared)));
        };
        admit(serial_scores_kernel<kTopK, kBlock, false, false>);
        admit(serial_scores_kernel<kTopK, kBlock, true, false>);
        admit(serial_scores_kernel<kTopK, kBlock, false, true>);
        admit(serial_scores_kernel<kTopK, kBlock, true, true>);
    });
    const TableRef tg_ref  = table_ref(tg);
    const TableRef tsv_ref = table_ref(tsv);
    const HeadRef  head_ref_local = head_ref(head);
    const bool     nvfp4          = tg.qtype == QType::NVFP4;
    const bool     head_q8        = head.qtype == QType::Q8_G32_FP16;
    const auto dispatch = [&](auto kernel) {
        kernel<<<static_cast<unsigned>(drafts * batch), kBlock, shared, stream>>>(
            static_cast<const std::int32_t*>(cand.data),
            static_cast<const __nv_bfloat16*>(raw.data), static_cast<const float*>(rms.data),
            static_cast<const __nv_bfloat16*>(packed_q.data),
            static_cast<const std::int32_t*>(anchors.data),
            static_cast<const __nv_bfloat16*>(norm_w.data),
            static_cast<const __nv_bfloat16*>(gate_bias.data), head_ref_local, tg_ref, tsv_ref,
            hidden, rank, width, drafts, batch, static_cast<std::int32_t*>(idx.data),
            scores ? static_cast<float*>(scores->data) : nullptr);
    };
    if (nvfp4) {
        if (head_q8) { dispatch(serial_scores_kernel<kTopK, kBlock, true, true>); }
        else { dispatch(serial_scores_kernel<kTopK, kBlock, true, false>); }
    } else {
        if (head_q8) { dispatch(serial_scores_kernel<kTopK, kBlock, false, true>); }
        else { dispatch(serial_scores_kernel<kTopK, kBlock, false, false>); }
    }
}

void dspine_serial_tail_launch(const Tensor& table, const Tensor& cand,
                               const ops::SamplingConfig* configs, const Tensor& positions,
                               Tensor& drafts, Tensor& proposal_q, Tensor& idx,
                               cudaStream_t stream) {
    const std::int32_t batch = drafts.ne[0];
    serial_tail_kernel<<<static_cast<unsigned>(batch), 32, 0, stream>>>(
        static_cast<const float*>(table.data), static_cast<const std::int32_t*>(cand.data),
        configs, static_cast<const std::int32_t*>(positions.data),
        static_cast<std::int32_t*>(drafts.data), static_cast<float*>(proposal_q.data),
        static_cast<std::int32_t*>(idx.data), drafts.ne[1], batch);
}

void dspine_serial_chain_launch(const Tensor& cand, const Tensor& idx, Tensor& drafts,
                                cudaStream_t stream) {
    const std::int32_t batch = drafts.ne[1];
    serial_chain_kernel<<<static_cast<unsigned>((batch + 31) / 32), 32, 0, stream>>>(
        static_cast<const std::int32_t*>(cand.data), static_cast<const std::int32_t*>(idx.data),
        static_cast<std::int32_t*>(drafts.data), drafts.ne[0], batch);
}

} // namespace detail
} // namespace ninfer::ops
