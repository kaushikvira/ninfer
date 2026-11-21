#pragma once

#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/sampling.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

// DSpine code-bridge and serial-refinement tail kernels. Every rounding point mirrors the
// reference bridge: BF16 storage with FP32 arithmetic, BF16 rounding exactly where the
// reference rounds. All ops are CUDA-graph capturable and allocate nothing internally.
//
// Column layout is [rows, width, batch] with width = drafts + 1; column t = 0 of each batch
// row is the anchor. "columns" below means width * batch.

// raw = residual (the draft layer already accumulated its residual stream).
// normalized = bf16(fp32(raw) / (rms + eps)), rms = sqrt(max(mean(fp32(raw)^2), FLT_MIN)).
// residual [H, columns] BF16 -> normalized [H, columns] BF16, rms [columns] FP32.
void dspine_prepare_read(const Tensor& residual, float eps, Tensor& normalized, Tensor& rms,
                         cudaStream_t stream);

// projected [C, columns] BF16 -> codes [C, columns] BF16.
// codes = bf16(sqrt(C) * fp32(projected) / (sqrt(sum(fp32(projected)^2)) + eps)); anchor
// columns (t == 0) are replaced by bf16(sqrt(C) * fp32(codebook[anchor_id, :])).
// anchors [batch] I32, codebook [vocab, C] BF16 or NVFP4 (rows dequantized on gather).
void dspine_normalize_codes(const Tensor& projected, const Tensor& anchors, const Weight& codebook,
                            std::int32_t width, float eps, Tensor& codes, cudaStream_t stream);

// packed [3R, width, batch] BF16 = [gate; predecessor_gate; prev] projections of the codes.
// message[t] = bf16(sigmoid(fp32(q[t]) + fp32(s[t-1]) + fp32(bias)) * fp32(v[t-1])) for t > 0,
// message[0] = 0, where q/s/v are the three R-row slices. gate_bias [R] BF16.
// message [R, width, batch] BF16.
void dspine_message(const Tensor& packed, const Tensor& gate_bias, Tensor& message,
                    cudaStream_t stream);

// Refines the residual stream and fuses the next layer's input norm.
// out = bf16(fp32(raw) + rms * fp32(delta) * (t != 0)); residual <- out.
// normed = bf16(fp32(bf16(out * rsqrt(mean(out^2) + eps))) * fp32(next_norm)).
// residual [H, width, batch] BF16 in/out, rms [width * batch] FP32, delta [H, width, batch] BF16,
// next_norm [H] BF16, normed [H, width, batch] BF16. stash_raw, when non-null, receives a copy
// of the pre-refinement residual (the serial tail consumes the raw stream, not the refined one).
void dspine_finish(Tensor& residual, const Tensor& rms, const Tensor& delta, const Tensor& next_norm,
                   float eps, Tensor& normed, Tensor* stash_raw, cudaStream_t stream);

// Serial-refinement scoring. cand [16, drafts, batch] I32 holds the top-16 candidate ids per
// proposal row (any fixed column order; the chain is permutation invariant within a row).
// raw [H, width, batch] BF16, rms [width, batch] FP32 and packed_q [R, width, batch] BF16 are
// the last-layer stash; row t = 1 + i is read for proposal row i. anchors [batch] I32,
// norm_w [H] BF16 (final norm), gate_bias [R] BF16, tg [vocab, R] and tsv [vocab, 2R] BF16 or
// NVFP4 (rows dequantized on gather; formats must match).
//
// head [vocab, H] is the tower's own output head - the drafter reuses it, it carries no copy -
// so it arrives in the representation the tower declares: FP8_E4M3FN_ROW_BF16 (RowScale) or
// Q8_G32_FP16 (RowSplit, group 32, FP16 group scales, the prod V3 tower's format). Both are
// gathered by candidate id and dequantized per element inside the kernel, the way the tg/tsv
// and codebook rows already are; a copy of the head into one shared representation is never
// taken (it would cost 1.3 - 2.5 GiB of VRAM the drafter is not sized for).
//
// Per (batch, row i, sender j): base_k = dot(raw * norm_w, dequant(head[cand_k]));
// sid_j = (i == 0 ? anchor : cand[i-1, j]); m_j = sigmoid(q + tsv[sid_j][:R] + bias) *
// tsv[sid_j][R:]; score_jk = base_k + rms * dot(m_j, tg[cand_k]); idx_j = argmax_k score_jk.
// idx [16, drafts, batch] I32.
//
// scores, when non-null, additionally receives the K x K score matrix of every (batch, row) as
// FP32 [batch, 16 (candidate), 16 (sender), drafts] (first axis fastest) - the table
// dspine_serial_tail consumes, and the un-collapsed form of the same score_s the argmax reads.
// It is a caller-provided buffer; the Op allocates nothing.
void dspine_serial_scores(const Tensor& cand, const Tensor& raw, const Tensor& rms,
                          const Tensor& packed_q, const Tensor& anchors, const Tensor& norm_w,
                          const Tensor& gate_bias, const Weight& head, const Weight& tg,
                          const Weight& tsv, std::int32_t width, Tensor& idx, Tensor* scores,
                          cudaStream_t stream);

/**
 * Refinement-tail selection under the live request policy (the stochastic sibling of
 * dspine_serial_scores + dspine_serial_chain, which are the greedy pair).
 *
 * For each slot i, every sender j has its own row of kTopK candidate scores. The row is turned
 * into q = filter(softmax(score * inv_temp)) with the request's joint top-k / strict-prefix
 * top-p / min-p (propose.py:73-90), one candidate is drawn from q with the counter-based stream
 * keyed on (configs[b].seed, positions[b], kSamplePurposeDFlash2Proposal, i), and the walk then
 * advances sender = drawn index (propose.py:114-152). drafts[i] is the walked candidate id.
 *
 * configs is Device SamplingConfig[batch]. temperature<=0 selects the greedy degenerate: the
 * argmax chain with a one-hot q, i.e. exactly dspine_serial_scores + dspine_serial_chain.
 * Presence/frequency penalties and token masks are NOT expressible by the reference tail, so a
 * caller must reject such a request before launching (models/common/dspine/
 * dspine_serial_tail.h does that at plan time); this Op reads neither field.
 *
 * Layouts are first-axis-fastest: table FP32 [batch, kTopK, kTopK, drafts] (candidate, sender,
 * slot), cand I32 [batch, drafts, kTopK], positions I32 [batch], drafts I32 [batch, drafts],
 * proposal_q FP32 [batch, drafts, kTopK] (the walked sender's row - the distribution that
 * produced that draft, which speculative_accept_sparse_drafts verifies against), idx I32
 * [batch, drafts, kTopK] (per sender, the twin of dspine_serial_scores' idx). scores must be
 * finite and each q row non-empty; no hidden allocation, no host synchronization, and the
 * result is a pure function of its inputs, so the Op is CUDA-graph capturable and replayable.
 */
void dspine_serial_tail(const Tensor& table, const Tensor& cand,
                        const ops::SamplingConfig* configs, const Tensor& positions, Tensor& drafts,
                        Tensor& proposal_q, Tensor& idx, cudaStream_t stream);

// Walks the sender chain: j = idx[0, 0]; drafts[i] = cand[i, j]; j = idx[i+1, j].
// cand/idx [16, drafts, batch] I32 -> drafts [drafts, batch] I32.
void dspine_serial_chain(const Tensor& cand, const Tensor& idx, Tensor& drafts, cudaStream_t stream);

} // namespace ninfer::ops
