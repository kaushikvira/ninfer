#pragma once

// DFlash2 teacher-forcing feature dump (tfdump/1 layout), ported from the capy-flash
// qwen4_tfdump (model-agnostic serializer only). Off by default; enabled by the
// environment:
//
//   NINFER_DFLASH_TFDUMP=<dir>            root output directory (unset = off)
//   NINFER_DFLASH_TFDUMP_LAYERS=<list>    comma list of tap layer indices | all | none
//                                         (default all tapped layers)
//   NINFER_DFLASH_TFDUMP_TOPK=<k>         write the k most-likely tokens per position to
//                                         topk.bin instead of full logits (default 0 = full)
//   NINFER_DFLASH_TFDUMP_COMPACT=1        append per-layer residuals to hidden_LXX.bin
//                                         (records {i32 first_position, i32 columns, i32 rows,
//                                         bf16[columns][rows]}) instead of per-position .npy
//   NINFER_DFLASH_TFDUMP_LOGITS=0         disable the per-position logits
//
// On-disk layout under <dir>:
//   tokens.npy        int64 [T]   the scored sequence
//   positions.npy     int32 [T]   the token positions
//   hidden_LXX.bin    (compact)   per tapped layer XX, appended records
//   topk.bin          (topk>0)    records {i32 position, i32 ids[k], f32 logprobs[k]}
//   logits/pNNNNNN.npy (topk==0)  f32 [V]  full output-head logits after position N
//   meta.json
//
// This is an additive consumer of the dense DFlashFeatureSink tap: it D2H-copies each tapped
// layer residual as it is captured and writes it to disk. It never changes the generated tokens
// and adds no work when NINFER_DFLASH_TFDUMP is unset (the sink holds a null pointer).

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

class DFlashTfDump final {
public:
    // Null when NINFER_DFLASH_TFDUMP is unset. `tap_layers` are the actual tapped layer indices
    // (the DFlash target_layer_ids, e.g. {5,19,33,47,61}) used for the per-layer selection.
    [[nodiscard]] static std::unique_ptr<DFlashTfDump>
    from_environment(std::span<const std::uint32_t> tap_layers);

    explicit DFlashTfDump(std::filesystem::path directory, std::vector<std::uint32_t> tap_layers,
                          std::vector<bool> selected, bool logits, bool compact, std::int32_t topk);

    // D2H-copy the tapped layer residual (contiguous BF16 [rows, T]) and write it to disk.
    // `first_position` is the sequence position of column 0 (0 when unknown).
    void capture_layer(std::int32_t layer, const Tensor& value, std::int32_t first_position,
                       cudaStream_t stream);

    // Store the token positions (int32 [T]) for naming + positions.npy.
    void capture_positions(const Tensor& positions, cudaStream_t stream);

    // Append the chunk's token ids to tokens.npy (int64 [T]); rewritten with the running
    // total, so chunked prefills accumulate across chunks.
    void write_tokens(std::span<const std::int32_t> tokens);
    // BF16 [V, T] host logits (vocab-major); position c is column c, vocab i is logits[i * T + c].
    void write_logits(std::span<const std::uint16_t> logits, std::int32_t rows, std::int32_t cols,
                      std::span<const std::int32_t> positions) const;
    // D2H a device BF16 [V, len] logits tensor and write its top-k logprobs (chunk-relative
    // positions 0..len-1). No-op when logits are disabled.
    void capture_logits(const Tensor& logits, cudaStream_t stream);
    void write_meta(const std::string& producer, const std::string& artifact) const;

    [[nodiscard]] bool enabled() const noexcept { return true; }

private:
    std::filesystem::path directory_;
    std::vector<std::uint32_t> tap_layers_; // actual tapped layer indices
    std::vector<bool> selected_;            // per tap_layers_ entry
    bool logits_  = true;
    bool compact_ = true;
    std::int32_t topk_ = 0;
    std::vector<std::uint16_t> scratch_;
    std::vector<std::int32_t> positions_;
    std::vector<std::int64_t> tokens_;
};

} // namespace ninfer::models::qwen3_5::execution
