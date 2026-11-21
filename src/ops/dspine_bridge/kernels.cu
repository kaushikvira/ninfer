// DSpine code-bridge and serial-refinement tail kernels. BF16 storage, FP32 arithmetic,
// BF16 rounding exactly at the reference rounding points (dspine_mlx/bridge.py, batched_tail.py).

#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/quant/nvfp4.cuh"
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
template <int K, int Block, bool Nvfp4>
__launch_bounds__(Block) __global__ void serial_scores_kernel(
    const std::int32_t* cand, const __nv_bfloat16* raw, const float* rms,
    const __nv_bfloat16* packed_q, const std::int32_t* anchors, const __nv_bfloat16* norm_w,
    const __nv_bfloat16* gate_bias, const std::uint8_t* head_codes, const __nv_bfloat16* head_scales,
    TableRef tg, TableRef tsv, std::int32_t hidden, std::int32_t rank,
    std::int32_t width, std::int32_t drafts, std::int32_t batch, std::int32_t* idx) {
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

    // base_k = dot(raw * norm_w, dequant(head[cand_k])), streamed over H.
    {
        const int k     = tid % K;
        const int group = tid / K;
        const std::int64_t raw_base = row_base * hidden;
        const std::int64_t head_base =
            static_cast<std::int64_t>(cand_s[k]) * hidden;
        const float scale = bf16f(head_scales[cand_s[k]]);
        float acc         = 0.0f;
        for (int h = group; h < hidden; h += Block / K) {
            const float code = bf16f(fbf16(common::decode_e4m3(head_codes[head_base + h]) * scale));
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

void dspine_serial_scores_launch(const Tensor& cand, const Tensor& raw, const Tensor& rms,
                                 const Tensor& packed_q, const Tensor& anchors,
                                 const Tensor& norm_w, const Tensor& gate_bias, const Weight& head,
                                 const Weight& tg, const Weight& tsv, std::int32_t width,
                                 Tensor& idx, cudaStream_t stream) {
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
        CUDA_CHECK(cudaFuncSetAttribute(serial_scores_kernel<kTopK, kBlock, false>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(shared)));
        CUDA_CHECK(cudaFuncSetAttribute(serial_scores_kernel<kTopK, kBlock, true>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(shared)));
    });
    const TableRef tg_ref  = table_ref(tg);
    const TableRef tsv_ref = table_ref(tsv);
    const bool nvfp4       = tg.qtype == QType::NVFP4;
    if (nvfp4) {
        serial_scores_kernel<kTopK, kBlock, true>
            <<<static_cast<unsigned>(drafts * batch), kBlock, shared, stream>>>(
                static_cast<const std::int32_t*>(cand.data),
                static_cast<const __nv_bfloat16*>(raw.data), static_cast<const float*>(rms.data),
                static_cast<const __nv_bfloat16*>(packed_q.data),
                static_cast<const std::int32_t*>(anchors.data),
                static_cast<const __nv_bfloat16*>(norm_w.data),
                static_cast<const __nv_bfloat16*>(gate_bias.data),
                static_cast<const std::uint8_t*>(head.qdata),
                static_cast<const __nv_bfloat16*>(head.scales),
                tg_ref, tsv_ref, hidden, rank, width, drafts, batch,
                static_cast<std::int32_t*>(idx.data));
    } else {
        serial_scores_kernel<kTopK, kBlock, false>
            <<<static_cast<unsigned>(drafts * batch), kBlock, shared, stream>>>(
                static_cast<const std::int32_t*>(cand.data),
                static_cast<const __nv_bfloat16*>(raw.data), static_cast<const float*>(rms.data),
                static_cast<const __nv_bfloat16*>(packed_q.data),
                static_cast<const std::int32_t*>(anchors.data),
                static_cast<const __nv_bfloat16*>(norm_w.data),
                static_cast<const __nv_bfloat16*>(gate_bias.data),
                static_cast<const std::uint8_t*>(head.qdata),
                static_cast<const __nv_bfloat16*>(head.scales),
                tg_ref, tsv_ref, hidden, rank, width, drafts, batch,
                static_cast<std::int32_t*>(idx.data));
    }
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
