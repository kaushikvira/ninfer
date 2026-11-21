#pragma once

// DSpine serial-refinement tail: candidate-chain selection under the live request policy.
//
// The tail is the second half of a DSpine proposal round. The draft forward (bridge
// kernels, ops::dspine_serial_scores) produces, per proposal slot, a K x K score table:
// row `sender` is the sender state (the anchor for slot 0, the previously chosen
// candidate afterwards) and column `k` the K candidate ids of that slot. P0b proved that
// table bit-exact against Tencent's reference. This file consumes that table and turns it
// into the proposal the accept path verifies:
//
//   serial_sample_chain (dspine_mlx/dspine_mlx/propose.py:114-152)
//     p      = softmax(row * inv_temp)
//     q,keep = filter_probs(p, top_k, top_p, min_p)          (propose.py:73-90)
//     choice = #{cdf <= u * cdf[last]}  clamped to the last kept slot
//     drafts[i] = cand[choice][i]; sender = choice
//
// Everything here is plain C++ over caller-owned buffers: no allocation, no I/O, no
// device access, no stream or host synchronisation. That is the contract the device
// kernel (ops::dspine_serial_tail) must reproduce, and the reason this TU carries no
// CUDA include: it doubles as the CPU oracle the unit tests drive, so the policy is
// testable without a GPU.
//
// Layouts (row-major, the caller's flat buffers):
//   table      FP32 [drafts][slots][slots][batch]  (per slot: sender rows x candidate cols)
//   logits     FP32 [vocab][width][batch]          width = drafts + 1 (anchor column 0)
//   cand       I32  [slots][drafts][batch]
//   drafts_out I32  [drafts][batch]
//   proposal_q FP32 [slots][drafts][batch]         the walked sender's filtered q per slot
//   idx_out    I32  [slots][drafts][batch]         chosen candidate index per (slot, sender);
//                                                  the grep-comparable twin of
//                                                  ops::dspine_serial_scores' idx

#include <cstdint>

namespace ninfer::models::common::dspine {

// Tencent's refinement tail scores exactly this many candidates per slot
// (`dspine_mlx/propose.py` head_topk = 16; `kernels.cu` kTopK).
inline constexpr std::int32_t kSerialTailSlots = 16;
// The DSpine block is 16 wide: one anchor column plus 15 drafts. The CLI exposes
// drafts in [1,15] (mirrors the DFlash2 range in src/product/speculative_options.h).
inline constexpr std::int32_t kSerialTailMinDrafts = 1;
inline constexpr std::int32_t kSerialTailMaxDrafts = 15;

// The engine's counter-based RNG purpose used for proposal draws. DSpine and DFlash2 are
// mutually exclusive backends, so sharing the purpose keeps the draw stream definition in
// one place (ops::kSamplePurposeDFlash2Proposal == 5, include/nfer/ops/sampling.h).
inline constexpr std::int32_t kSerialTailPurpose = 5;

// The live request policy the tail samples under.
struct SerialTailPolicy {
    float temperature = 0.0f;  // <= 0 selects the greedy (argmax) chain
    std::int32_t top_k = 20;   // [1,20] per sampling.h; >= slots keeps every candidate
    float top_p        = 1.0f; // >= 1 disables the nucleus cut
    float min_p        = 0.0f; // <= 0 disables the min-p cut

    [[nodiscard]] bool greedy() const { return !(temperature > 0.0f); }
};

// Derives the tail policy from one request row's sampling configuration.
//
// The reference tail consumes exactly (inv_temp, top_k, top_p, min_p). A request that also
// asks for presence/frequency penalties or a token mask cannot be expressed, so this throws
// instead of silently dropping the constraint: a proposal sampled under a different
// distribution than the acceptor verifies would corrupt acceptance silently.
//
// Temperature is validated here rather than at the call site: NaN, inf and negatives beyond
// the greedy sentinel are rejected, and temperature == 0 (or negative) is the documented
// greedy degenerate.
SerialTailPolicy serial_tail_policy(float temperature, std::int32_t top_k, float top_p,
                                    float min_p, bool has_penalties, bool has_token_mask);

// Filtered, renormalised distribution over the slot's candidates.
//
// Mirrors propose.py:73-90 exactly: rank by descending probability with the lower slot index
// winning ties, keep while (rank < top_k) and (strict prefix mass before the slot <= top_p)
// and (p >= min_p * max p), then renormalise the kept mass. `q` is written for every slot
// (zero where the slot was cut) and `keep` reports the mask. The best candidate always
// survives, so the support is never empty.
void serial_tail_filter(const float* scores, std::int32_t slots, const SerialTailPolicy& policy,
                        float* q, bool* keep);

// One draw from `q` with uniform `u` in [0,1): choice = #{cdf <= u * cdf[last]}, clamped to
// the last kept slot (propose.py:150-152). Sequential fp32 summation, no RNG state.
std::int32_t serial_tail_draw(const float* q, const bool* keep, std::int32_t slots, float u);

// Counter-based uniform in [0,1) identical to ops/sampling/kernels/random.cuh:14-28. Pure
// function of (seed, position, purpose, sub) so CUDA-graph replay is bit-identical.
float serial_tail_uniform(std::uint64_t seed, std::int32_t position, std::int32_t purpose,
                          std::uint32_t sub);

// Builds the K x K score table for a plain per-slot-logit proposal: entry (sender, k) of slot
// i is logits[cand[k][i][b]][1 + i][b]. This is the adapter for the sibling draft-forward
// stage's logits; the bridge path supplies its own table from ops::dspine_serial_scores.
void serial_tail_table_from_logits(const float* logits, std::int32_t vocab, std::int32_t width,
                                   const std::int32_t* cand, std::int32_t slots,
                                   std::int32_t drafts, std::int32_t batch, float* table);

// Runs one proposal round for `batch` rows. Every (slot, sender) pair gets its filtered
// distribution and index - the walk consumes only the sender row it reaches, but writing the
// whole table keeps the output shape identical to ops::dspine_serial_scores so the greedy
// path can be compared against the P0 oracle entry by entry.
//
// `seeds` [batch] and `positions` [batch] are the request seed and the frontier position the
// draws are keyed on; slot i uses sub-index i, so every draw is a pure function of the
// request (graph-safe, replay-identical). Greedy rows take the argmax chain (lowest slot
// index on ties) and write a one-hot q, so the sparse acceptor's q contract (normalised,
// positive at the chosen id) holds for mixed greedy/stochastic batches too.
void serial_tail_run(const float* table, const std::int32_t* cand, std::int32_t slots,
                     std::int32_t drafts, std::int32_t batch, const SerialTailPolicy* policies,
                     const std::uint64_t* seeds, const std::int32_t* positions,
                     std::int32_t* drafts_out, float* proposal_q, std::int32_t* idx_out);

// Shape contract of the tail's output against the accept entry it feeds
// (ops::speculative_accept_sparse_drafts): candidate_ids [16,K,B], proposal_q FP32 [16,K,B],
// drafts I32 [K,B], with K in [1,15]. Returns a printable reason, or nullptr when valid.
const char* serial_tail_accept_geometry_error(std::int32_t slots, std::int32_t drafts,
                                              std::int32_t batch);

// Validation the acceptor relies on, over one round's output: every q row is finite,
// non-negative and normalises to 1 (within `tolerance`); each live candidate column is
// distinct (speculative_accept_sparse_drafts requires that); and the walked draft id resolves
// to a unique candidate slot that carries positive proposal mass. Returns nullptr when the
// round is valid, otherwise a printable reason. Host-side; the tests drive it today and the
// stage-3 device path reuses it as a self-check.
const char* serial_tail_accept_q_error(const float* proposal_q, const std::int32_t* cand,
                                       const std::int32_t* drafts_out, std::int32_t slots,
                                       std::int32_t drafts, std::int32_t batch, float tolerance);

} // namespace ninfer::models::common::dspine
