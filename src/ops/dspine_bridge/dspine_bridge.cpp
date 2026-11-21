#include "ninfer/ops/dspine_bridge.h"

#include "ops/common/host/operands.h"
#include "ops/dspine_bridge/launch.h"

#include <stdexcept>

namespace ninfer::ops {
namespace {

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(message); }
}

bool is_bf16_matrix(const Tensor& tensor, std::int32_t rows, std::int64_t columns) {
    return tensor.dtype == DType::BF16 && tensor.is_contiguous() && tensor.ne[0] == rows &&
           tensor.numel() == static_cast<std::int64_t>(rows) * columns;
}

// A vocabulary-indexed bridge table stored either BF16 (contiguous) or NVFP4 (block-scale planes).
void require_table(const Weight& weight, std::int32_t k, std::int32_t n, const char* message) {
    require((weight.qtype == QType::BF16 || weight.qtype == QType::NVFP4) && weight.qdata &&
                weight.k == k && weight.n == n,
            message);
    if (weight.qtype == QType::NVFP4) { common::validate_nvfp4_weight(weight, message); }
}

// The proposal head, in one of the two representations the op dequantizes on gather. The prod V3
// tower head is Q8_G32_FP16; FP8_E4M3FN_ROW_BF16 is the P0b-verified one. A row stride is part of
// the contract, not an implementation detail: RowSplit pads the columns to 128 (core/
// weight_view.cpp), and reading a padded row as a packed one reads the wrong head.
void require_head(const Weight& head, std::int32_t hidden, const char* message) {
    require(head.n > 0 && head.k == hidden && head.qdata && head.scales, message);
    require(head.qhigh == nullptr && head.high_plane_bytes == 0, message);
    require(head.padded_shape[0] == head.n && head.padded_shape[1] >= head.k, message);
    if (head.qtype == QType::FP8_E4M3FN_ROW_BF16) {
        require(head.layout == QuantLayout::RowScale && head.scale_dtype == DType::BF16 &&
                    head.padded_shape[1] == head.k,
                message);
        return;
    }
    // The group of a column is `column / 32`, so the logical columns must be whole groups; the
    // padded row stride is what the code plane is addressed by.
    require(head.qtype == QType::Q8_G32_FP16 && head.layout == QuantLayout::RowSplit &&
                head.scale_dtype == DType::FP16 && head.group == 32 &&
                head.group_size == static_cast<std::uint32_t>(32) && (head.k % 32) == 0 &&
                head.padded_shape[1] % 32 == 0,
            message);
}

} // namespace

void dspine_prepare_read(const Tensor& residual, float eps, Tensor& normalized, Tensor& rms,
                         cudaStream_t stream) {
    require(residual.dtype == DType::BF16 && residual.is_contiguous() && residual.ne[0] > 0,
            "dspine prepare_read: residual must be contiguous BF16 [H, columns]");
    const std::int32_t hidden  = residual.ne[0];
    const std::int64_t columns = residual.numel() / hidden;
    require(is_bf16_matrix(normalized, hidden, columns),
            "dspine prepare_read: normalized shape differs from residual");
    require(rms.dtype == DType::FP32 && rms.is_contiguous() && rms.numel() == columns,
            "dspine prepare_read: rms must be FP32 [columns]");
    detail::dspine_prepare_read_launch(residual, eps, normalized, rms, stream);
}

void dspine_normalize_codes(const Tensor& projected, const Tensor& anchors, const Weight& codebook,
                            std::int32_t width, float eps, Tensor& codes, cudaStream_t stream) {
    require(projected.dtype == DType::BF16 && projected.is_contiguous() && projected.ne[0] > 0,
            "dspine normalize_codes: projected must be contiguous BF16 [C, columns]");
    const std::int32_t code_size = projected.ne[0];
    const std::int64_t columns   = projected.numel() / code_size;
    require(width > 0 && columns % width == 0,
            "dspine normalize_codes: columns must be a positive multiple of width");
    const std::int64_t batch = columns / width;
    require(anchors.dtype == DType::I32 && anchors.is_contiguous() && anchors.numel() == batch,
            "dspine normalize_codes: anchors must be I32 [batch]");
    require(codebook.n > 0, "dspine normalize_codes: codebook vocab must be positive");
    require_table(codebook, code_size, codebook.n,
                  "dspine normalize_codes: codebook must be BF16/NVFP4 [vocab, C]");
    require(is_bf16_matrix(codes, code_size, columns),
            "dspine normalize_codes: codes shape differs from projected");
    detail::dspine_normalize_codes_launch(projected, anchors, codebook, width, eps, codes, stream);
}

void dspine_message(const Tensor& packed, const Tensor& gate_bias, Tensor& message,
                    cudaStream_t stream) {
    require(packed.dtype == DType::BF16 && packed.is_contiguous() && packed.ne[0] > 0 &&
                packed.ne[0] % 3 == 0,
            "dspine message: packed must be contiguous BF16 [3R, width, batch]");
    const std::int32_t rank = packed.ne[0] / 3;
    require(gate_bias.dtype == DType::BF16 && gate_bias.is_contiguous() &&
                gate_bias.numel() == rank,
            "dspine message: gate_bias must be BF16 [R]");
    require(message.dtype == DType::BF16 && message.is_contiguous() && message.ne[0] == rank &&
                message.numel() == packed.numel() / 3,
            "dspine message: message must be BF16 [R, width, batch]");
    detail::dspine_message_launch(packed, gate_bias, message, stream);
}

void dspine_finish(Tensor& residual, const Tensor& rms, const Tensor& delta, const Tensor& next_norm,
                   float eps, Tensor& normed, Tensor* stash_raw, cudaStream_t stream) {
    require(residual.dtype == DType::BF16 && residual.is_contiguous() && residual.ne[0] > 0,
            "dspine finish: residual must be contiguous BF16 [H, width, batch]");
    const std::int32_t hidden  = residual.ne[0];
    const std::int64_t columns = residual.numel() / hidden;
    require(rms.dtype == DType::FP32 && rms.is_contiguous() && rms.numel() == columns,
            "dspine finish: rms must be FP32 [width * batch]");
    require(is_bf16_matrix(delta, hidden, columns),
            "dspine finish: delta shape differs from residual");
    require(next_norm.dtype == DType::BF16 && next_norm.is_contiguous() &&
                next_norm.numel() == hidden,
            "dspine finish: next_norm must be BF16 [H]");
    require(is_bf16_matrix(normed, hidden, columns),
            "dspine finish: normed shape differs from residual");
    if (stash_raw) {
        require(is_bf16_matrix(*stash_raw, hidden, columns),
                "dspine finish: stash_raw shape differs from residual");
    }
    detail::dspine_finish_launch(residual, rms, delta, next_norm, eps, normed, stash_raw, stream);
}

void dspine_serial_scores(const Tensor& cand, const Tensor& raw, const Tensor& rms,
                          const Tensor& packed_q, const Tensor& anchors, const Tensor& norm_w,
                          const Tensor& gate_bias, const Weight& head, const Weight& tg,
                          const Weight& tsv, std::int32_t width, Tensor& idx, Tensor* scores,
                          cudaStream_t stream) {
    require(cand.dtype == DType::I32 && cand.is_contiguous() && cand.ne[0] == 16,
            "dspine serial_scores: cand must be I32 [16, drafts, batch]");
    const std::int32_t drafts = cand.ne[1];
    const std::int32_t batch  = cand.ne[2];
    require(drafts > 0 && batch > 0 && width == drafts + 1,
            "dspine serial_scores: width must equal drafts + 1");
    require(raw.dtype == DType::BF16 && raw.is_contiguous() && raw.ne[0] > 0 &&
                raw.ne[1] == width && raw.ne[2] == batch,
            "dspine serial_scores: raw must be BF16 [H, width, batch]");
    const std::int32_t hidden = raw.ne[0];
    require(rms.dtype == DType::FP32 && rms.is_contiguous() && rms.ne[0] == width &&
                rms.ne[1] == batch,
            "dspine serial_scores: rms must be FP32 [width, batch]");
    require(packed_q.dtype == DType::BF16 && packed_q.is_contiguous() && packed_q.ne[0] > 0 &&
                packed_q.ne[1] == width && packed_q.ne[2] == batch,
            "dspine serial_scores: packed_q must be BF16 [R, width, batch]");
    const std::int32_t rank = packed_q.ne[0];
    require(anchors.dtype == DType::I32 && anchors.is_contiguous() && anchors.numel() == batch,
            "dspine serial_scores: anchors must be I32 [batch]");
    require(norm_w.dtype == DType::BF16 && norm_w.is_contiguous() && norm_w.numel() == hidden,
            "dspine serial_scores: norm_w must be BF16 [H]");
    require(gate_bias.dtype == DType::BF16 && gate_bias.is_contiguous() &&
                gate_bias.numel() == rank,
            "dspine serial_scores: gate_bias must be BF16 [R]");
    require_head(head, hidden,
                 "dspine serial_scores: head must be the tower's output head [vocab, H] in "
                 "FP8_E4M3FN_ROW_BF16 (RowScale) or Q8_G32_FP16 (RowSplit, group 32, FP16 scales)");
    require_table(tg, rank, head.n, "dspine serial_scores: tg must be BF16/NVFP4 [vocab, R]");
    require_table(tsv, 2 * rank, head.n, "dspine serial_scores: tsv must be BF16/NVFP4 [vocab, 2R]");
    require(tg.qtype == tsv.qtype, "dspine serial_scores: tg/tsv storage formats must match");
    require(idx.dtype == DType::I32 && idx.is_contiguous() && idx.ne[0] == 16 &&
                idx.ne[1] == drafts && idx.ne[2] == batch,
            "dspine serial_scores: idx must be I32 [16, drafts, batch]");
    // The same geometry dspine_serial_tail requires of the table it consumes.
    if (scores != nullptr) {
        require(scores->dtype == DType::FP32 && scores->is_contiguous() &&
                    scores->ne[0] == batch && scores->ne[1] == 16 && scores->ne[2] == 16 &&
                    scores->ne[3] == drafts,
                "dspine serial_scores: scores must be FP32 [batch, 16 (candidate), "
                "16 (sender), drafts]");
    }
    detail::dspine_serial_scores_launch(cand, raw, rms, packed_q, anchors, norm_w, gate_bias, head,
                                        tg, tsv, width, idx, scores, stream);
}

void dspine_serial_tail(const Tensor& table, const Tensor& cand,
                        const ops::SamplingConfig* configs, const Tensor& positions,
                        Tensor& drafts, Tensor& proposal_q, Tensor& idx, cudaStream_t stream) {
    require(cand.dtype == DType::I32 && cand.is_contiguous() && cand.ne[2] == 16,
            "dspine serial_tail: cand must be I32 [batch, drafts, 16]");
    const std::int32_t batch   = cand.ne[0];
    const std::int32_t ndrafts = cand.ne[1];
    require(batch > 0 && ndrafts > 0 && ndrafts <= 15,
            "dspine serial_tail: drafts must be in [1,15] and batch positive");
    require(table.dtype == DType::FP32 && table.is_contiguous() &&
                table.ne[0] == batch && table.ne[1] == 16 && table.ne[2] == 16 &&
                table.ne[3] == ndrafts,
            "dspine serial_tail: table must be FP32 [batch, 16 (cand), 16 (sender), drafts]");
    require(configs != nullptr, "dspine serial_tail: sampling configs are required");
    require(positions.dtype == DType::I32 && positions.is_contiguous() &&
                positions.ne[0] == batch,
            "dspine serial_tail: positions must be I32 [batch]");
    require(drafts.dtype == DType::I32 && drafts.is_contiguous() && drafts.ne[0] == batch &&
                drafts.ne[1] == ndrafts,
            "dspine serial_tail: drafts must be I32 [batch, drafts]");
    require(proposal_q.dtype == DType::FP32 && proposal_q.is_contiguous() &&
                proposal_q.ne[0] == batch && proposal_q.ne[1] == ndrafts && proposal_q.ne[2] == 16,
            "dspine serial_tail: proposal_q must be FP32 [batch, drafts, 16]");
    require(idx.dtype == DType::I32 && idx.is_contiguous() && idx.ne[0] == batch &&
                idx.ne[1] == ndrafts && idx.ne[2] == 16,
            "dspine serial_tail: idx must be I32 [batch, drafts, 16]");
    detail::dspine_serial_tail_launch(table, cand, configs, positions, drafts, proposal_q, idx,
                                      stream);
}

void dspine_serial_chain(const Tensor& cand, const Tensor& idx, Tensor& drafts,
                         cudaStream_t stream) {
    require(cand.dtype == DType::I32 && cand.is_contiguous() && cand.ne[0] == 16,
            "dspine serial_chain: cand must be I32 [16, drafts, batch]");
    require(idx.dtype == DType::I32 && idx.is_contiguous() && idx.ne[0] == 16 &&
                idx.ne[1] == cand.ne[1] && idx.ne[2] == cand.ne[2],
            "dspine serial_chain: idx shape differs from cand");
    require(drafts.dtype == DType::I32 && drafts.is_contiguous() &&
                drafts.ne[0] == cand.ne[1] && drafts.ne[1] == cand.ne[2],
            "dspine serial_chain: drafts must be I32 [drafts, batch]");
    detail::dspine_serial_chain_launch(cand, idx, drafts, stream);
}

} // namespace ninfer::ops
