#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {
struct SamplingConfig;
}

namespace ninfer::ops::detail {

void dspine_prepare_read_launch(const Tensor& residual, float eps, Tensor& normalized, Tensor& rms,
                                cudaStream_t stream);
void dspine_normalize_codes_launch(const Tensor& projected, const Tensor& anchors,
                                   const Weight& codebook, std::int32_t width, float eps,
                                   Tensor& codes, cudaStream_t stream);
void dspine_message_launch(const Tensor& packed, const Tensor& gate_bias, Tensor& message,
                           cudaStream_t stream);
void dspine_finish_launch(Tensor& residual, const Tensor& rms, const Tensor& delta,
                          const Tensor& next_norm, float eps, Tensor& normed, Tensor* stash_raw,
                          cudaStream_t stream);
void dspine_serial_scores_launch(const Tensor& cand, const Tensor& raw, const Tensor& rms,
                                 const Tensor& packed_q, const Tensor& anchors,
                                 const Tensor& norm_w, const Tensor& gate_bias, const Weight& head,
                                 const Weight& tg, const Weight& tsv, std::int32_t width,
                                 Tensor& idx, Tensor* scores, cudaStream_t stream);
void dspine_serial_chain_launch(const Tensor& cand, const Tensor& idx, Tensor& drafts,
                                cudaStream_t stream);
void dspine_serial_tail_launch(const Tensor& table, const Tensor& cand,
                               const ops::SamplingConfig* configs, const Tensor& positions,
                               Tensor& drafts, Tensor& proposal_q, Tensor& idx,
                               cudaStream_t stream);

} // namespace ninfer::ops::detail
