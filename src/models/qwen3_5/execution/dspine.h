#pragma once

// P2 stage 4 - the DSpine proposal round in the engine's execution layer.
//
// D-42 closed the forward probe with a finding this file exists to answer: "the accept op is
// real in the engine but has NO DSpine wiring - drafts are never verified against target
// logits". The five layers, the six bridge ops, the head gather, the score table and the
// sampled tail are all built, gate-verified (P0b bit-exact, D-37, D-40) and walked on the real
// drafter (D-39, D-42); `SpeculativeBackend::Dspine` existed only in load options, config and
// the CLI, so the one number that decides this project - accepted tokens per round against
// DFlash2's measured 2.998 (D-40) - was not measurable at all.
//
// THE SEAM THIS ATTACHES TO (file:line at 72f66e79, the map the task order asks for)
//   draft call        program/decode.cpp:65  run_dflash_draft() -> execution::draft_propose at
//                     decode.cpp:92; the decode loop's entry is begin_draft() at decode.cpp:95,
//                     which routes Mtp (:111) / masked-draft (:112), then prepare_verify_ids
//                     (:119) builds the (K+1)-wide verify block.
//   draft context     program/decode.cpp:29 and :45 append_draft_history() ->
//                     execution::draft_append_context (:59); the prefill half is
//                     program/execution.cpp:74-88.
//   target verify     program/decision.cpp:51 project_decode() = the TOWER head over the
//                     (K+1)-wide block plus ops::argmax into frame.target_argmax (:67);
//                     decide_decode() :71 is the verify, and its DFlash2 arm is
//                     ops::speculative_accept_sparse_drafts at decision.cpp:84-93.
//   accounting        decision.cpp:204-211: accepted_drafts = count - 1, draft_extents =
//                     host_ingress.extents[row], row_stride = draft_window + 1 (:197). It reads
//                     the ACCEPT OP's counts, not a backend, so a DSpine round on that same op
//                     is counted by the same arithmetic that produced D-40's 2.998
//                     (serve/metrics.cpp:140, :431-436).
//   KV/position       decode.cpp:119; program/commit.cpp:43; session/sequence.cpp:74
//                     backend_frontier; checkpoints.cpp:146; pause.cpp:224; reclaim.cpp:479/505/554
//                     - all keyed on is_masked_draft_backend (load_options.h:52).
//   graph capture     program/graphs.cpp:213 draft_graph_profiles(); program/graph_bodies.cpp:56
//                     bodies.draft -> run_dflash_draft (:63), :68 bodies.accept -> decide_decode;
//                     the target tap sink the drafter feeds on is graph_bodies.cpp:38-45 and
//                     decode.cpp:186-196.
//   backend switches  load_options.h:42,:52; config.cpp:248,:254;
//                     planning/startup.cpp:547-576; frames.cpp:174; graphs.cpp:213;
//                     decision.cpp:81; decode.cpp:73; product/speculative_options.h:28,:54,:59.
//
// WHY A TAIL-ENTRY AND NOT A WHOLE-ROUND COPY
//   The five draft layers plus the bridge chain are the proposal stack the D-26 gate measures
//   (criterion 3) and the probe walks; copying them into the executor would put a second
//   implementation of the same mathematics on the serving path, which D-42's locked decision 3
//   forbids ("do not re-implement any of it in the executor; call it"). What the engine has no
//   name for - and the thing that makes acceptance unmeasurable - is the half from the proposal
//   head onward: the K-way head, the score table, the sampled tail, and the layout bridge from
//   the tail's layouts into the ones `speculative_accept_sparse_drafts` will take. That is what
//   dspine_propose_tail() owns. It issues the verified ops and implements none of them.
//
// THE THREE CANDIDATE/PROPOSAL_Q LAYOUTS (D-40's RED 2, third instance)
//   head and accept   element (candidate, slot, batch) at candidate + 16*(slot + K*batch)
//                     - linear_topk's own output, and the accept op's candidate_ids and
//                     proposal_q (src/ops/sampling/acceptance/kernel.cuh:109-111,
//                     entry.cpp's {16,K,B} operand shapes).
//   tail in and out   element (batch, slot, candidate) at batch_index + batch*(slot + K*candidate)
//                     - src/ops/dspine_bridge/kernels.cu:466-473.
//   serial_scores     takes the HEAD layout (kernels.cu:229) while serial_tail takes the TAIL
//                     layout, and the accept op demands a CONTIGUOUS [16,K,B] proposal_q, so the
//                     round must bridge head -> tail (the tail's candidate input) and tail ->
//                     head (the proposal_q the acceptor verifies against). At batch 1 the two are
//                     (candidate, slot) at c + 16*s and at s + 15*c: a transpose with an unequal
//                     stride, which cudaMemcpy2DAsync cannot express, and "no new kernels" is
//                     locked (D-14/D-31/D-35). So the bridge is an explicit host permute over 16
//                     * K words per row - the same move the probe's seed_tail_inputs makes - on
//                     the eager route, and it is a measured cost of this stage, reported: Dspine
//                     has no Graph profiles (graphs.cpp:213 returns {} for it), so this route is
//                     eager and the round-trip is on the round's critical path.
//
// NINFER TARGET: Dspine is a masked-block drafter (block 16: one anchor column plus 15 drafts,
// mask token id, target taps, a paged context of its own), so it attaches to the masked-draft
// gates, and every change on the DFlash2 path is `if (backend == Dspine)` additive.

#include "core/arena.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "models/qwen3_5/execution/parameters.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/softmax_attention.h"

#include <cstdint>

namespace ninfer::models::qwen3_5::execution {

// The drafter's execution view, the DSpine twin of `DraftExecutionView` (execution/draft.h:16-27).
// The five draft layers are all full-context - the drafter declares no sliding window
// (models/common/dspine/config.h:sliding_window is 0 in the shipped artifacts, and
// planning/startup.cpp:113-120 gives it the paged half of the masked-draft state and no cyclic
// local state at all) - so the view names one paged context and no cyclic one.
struct DspineExecutionView {
    const Parameters& parameters;
    MemoryArena& work;
    CudaExecutionView cuda;
    const PagedKVCache* context = nullptr;   // the drafter's own paged context, five layers

    [[nodiscard]] PagedKVBatchLayerView context_layer(std::uint32_t layer) const;
};

// THE DRAFT-CONTEXT APPEND - the DSpine arm of the decode loop's per-round context update
// (program/decode.cpp:29, :45). It is the same half the DFlash2 path runs in
// `append_context_impl` (execution/draft.cpp:41): the target-tap field the tower's feature sink
// wrote is projected once (`feature_projection`), normed once (`context_norm`), and every one of
// the drafter's five layers then projects that normed field into ITS OWN context key and value,
// norms the key, RoPEs it at the append positions, and appends the pair to that layer's own paged
// pages. Those pages are what `dspine_walk`'s per-layer context attention then reads, so the
// append is the KV and position advance the masked-block drafter needs: the block's positions
// start where the appended prefix ends.
//
// The context key and value are the K and V RANGES of the layer's packed Q/K/V object
// (models/common/dspine/weights.cpp binds them with require_shared_binding), and this engine has
// no NVFP4 linear compiled at their apart geometry (1024 x 5120) nor at the paired one the
// DFlash2 path uses (2048 x 5120, linear_pair). The split family, which the drafter's own packed
// parent reaches, is the one compiled route to those bytes: the projection is issued over the
// parent and the Q rows it also computes are dropped into a scratch stream. At a 16-wide context
// that is 4096 of 6144 output rows thrown away - a measured cost of the missing context geometry,
// recorded, not hidden behind a new kernel (the probe's own words, test_dspine_forward_real.cpp).
//
// `features` is the packed target-tap field [hidden * taps, width, batch], `positions`/`counts`
// the append positions and counts the loop derives with `prepare_ragged_prefix`, and
// `table_rows` the drafter paged context's row per batch lane.
void dspine_append_context(DspineExecutionView view, const Tensor& features,
                           const Tensor& positions, const Tensor& counts, const Tensor& table_rows,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope);

// THE FIVE-LAYER WALK - the drafter's proposal stack, from the mask block to the streams the
// proposal tail consumes. Ported step for step from the probe that walks it on the real weights
// (tests/models/qwen3_5/test_dspine_forward_real.cpp: run_forward and run_bridge), so the probe
// stays the parity harness for it and no second dialect of the mathematics exists.
//
//   block arrangement   ops::prepare_masked_block (the mask token in the draft positions)
//   embedding           the tower's own token embedding
//   five layers         { input norm (taken from the previous layer's fused write where there
//                        was one), split Q/K/V over the packed parent, query and key head norms
//                        then RoPE, paged context attention over the layer's own pages, the
//                        attention output projection plus the residual, the post-attention norm,
//                        the SwiGLU pair, the down projection into the residual }
//   after each layer    the seven-step code bridge { prepare_read, the code projection, the
//                        normalize_codes over the real codebook, the gate / predecessor / value
//                        projections into the packed block, the message, the write projection,
//                        the finish that refines the residual in place and fuses the next
//                        layer's input norm }
//
// The LAST layer's fused write is the tail's input: its `stash` (the pre-refinement residual),
// its `rms` (the per-column read scale) and its `message` (the packed Q) are exactly
// `dspine_serial_scores`' raw / rms / packed_q, and its normed block is the proposal hidden the
// tower's head scores. `dspine_propose_tail` takes those streams from here.
//
// ROUND RESET (D-42). The walk opens the way the engine's own masked-draft round does,
// `propose_batch_impl`'s `execution.work.reset()` (execution/draft.cpp:347): every transient it
// uses is arena storage the reset rewinds, the residual stream is re-derived from
// `prepare_masked_block` + `embedding` and never carried in, and the streams the round writes
// only IN PART are zeroed by `dspine_propose_tail`'s own round zero (its proposal distribution is
// written per walked sender row, dspine_serial_tail.cpp:258-268). That is what makes a round a
// pure function of its inputs - `0 of 37 streams part` over 32 rounds in D-42's probe - and on
// the decode path it is what keeps round N+1 from reading round N's partial planes, arrival
// counters, quantizer bounds, or sender rows. The drafter's five pages are NOT part of that zero:
// they are the sequence's history, advanced by `dspine_append_context`, and zeroing them would
// erase the context the round attends over.
struct DspineWalkInputs {
    Tensor anchors;         // [batch] I32 - each row's anchor token
    Tensor frontiers;       // [batch] I32 - the appended context length per row
    Tensor valid_columns;   // [batch] I32
    Tensor table_rows;      // [batch] I32 - the drafter context's page-table row per lane
};

struct DspineWalkOutputs {
    Tensor ids;            // [width, batch] I32 - the mask block, the anchor column plus the drafts
    Tensor positions;      // [width, batch] I32 - the block's draft positions
    Tensor bridge_raw;     // [hidden, width, batch] BF16 - the last write's stash
    Tensor bridge_rms;     // [width * batch] FP32       - the last write's read scale
    Tensor bridge_message; // [rank, width, batch] BF16  - the last write's packed message
    Tensor proposal_hidden;// [hidden, width, batch] BF16 - the last write's normed block
};

void dspine_walk(DspineExecutionView view, const DspineWalkInputs& inputs,
                 DspineWalkOutputs& outputs, ops::ContextAttentionExecutionEnvelope envelope);

// The arena bytes one DSpine round needs at the given geometry: the walk's transients (the
// streams it allocates for the whole round, plus the widest of its per-layer scopes), the
// quantized activations its A16-only projections stage, and the context append's own. The
// proposal tail's share is `dspine_propose_tail`'s, sized by the caller as it always was; the two
// are the two halves of the round, and the workspace is the round. A plan sized for the tail
// alone is a walk that runs out of arena mid-layer, which is a launch failure in the middle of
// the round, not a refusal.
[[nodiscard]] std::size_t dspine_round_workspace_bytes(
    const models::dspine::Config& draft, std::int32_t batch, std::int32_t width,
    ops::ContextAttentionExecutionEnvelope envelope, std::int32_t multiprocessors);

// The proposal tail's inputs: the streams the draft layers' LAST fused bridge write leaves
// behind, at the shapes `dspine_serial_scores` names them by. `stash` is the pre-refinement
// residual of the 16-wide block, `rms` its per-column read scale, `packed_q` the bridge message;
// row t = 1 + i of all three is the one proposal slot i scores against.
// The weights the tail names, pre-resolved: the tower's own output head (shared with the
// target, never copied), and the drafter's final norm, gate bias, tg and tsv.
struct DspineTailView {
    Weight head;
    Tensor final_norm, gate_bias;
    Weight tg, tsv;
    std::int32_t rank = 0;
};

struct DspineTailInputs {
    Tensor bridge_raw;      // [hidden, width, batch] BF16 - the pre-refinement residual
    Tensor bridge_rms;      // [width * batch] FP32     - the bridge's read scale
    Tensor bridge_message;  // [rank, width, batch] BF16 - the bridge's packed message
    Tensor proposal_hidden; // [hidden, width, batch] BF16 - the last fused write's normed block
    Tensor anchors;         // [batch] I32 - the block's anchor token per row
    Tensor draw_positions;  // [batch] I32 - each row's draw coordinate
    const ops::SamplingConfig* sampling = nullptr;
};

// The proposal tail's outputs, at the shapes the accept op takes them in: drafts is [K, batch]
// I32, candidate_ids [16, K, batch] I32 and proposal_q [16, K, batch] FP32, with the two
// proposal streams contiguous in the head layout (kernel.cuh:109-111). `extents` is the
// per-row count of live drafts, [batch] I32, which is what the accept op's
// `current_extents` is: K for every live row, 0 for a row the caller has retired.
struct DspineTailOutputs {
    Tensor drafts;
    Tensor candidate_ids;
    Tensor proposal_q;
};

// Runs the second half of one DSpine proposal round: the tower head's K-way top-k over the
// proposal hidden, `dspine_serial_scores` over the real tg/tsv tables (the score table), the
// sampled `dspine_serial_tail` under the row's live sampling policy, and the layout bridges
// the accept op's own contract requires. It writes the caller's `outputs` and nothing else:
// the drafts it publishes are the ones `speculative_accept_sparse_drafts` verifies, and the
// proposal_q it publishes is the distribution that produced each one (the accept op consumes
// it directly, per include/ninfer/ops/speculative_accept.h).
//
// The proposal head is the TOWER's own output head, so it arrives in the representation the
// tower declares (Q8_G32_FP16 on the prod V3 tower, D-37) and is gathered by candidate id
// inside the op: no copy of it is ever taken.
//
// `proposals` is the [hidden, K, batch] gather of the 15 proposal columns out of the 16-wide
// block: column 0 is the anchor and is not a proposal, so the caller hands the tail the
// K columns behind it (the same 2-D gather form the DFlash2 proposal path uses).
void dspine_propose_tail(const DspineTailView& view, const DspineTailInputs& inputs,
                         DspineTailOutputs& outputs, MemoryArena& work,
                         CudaExecutionView execution);

// The layout bridge, exposed for the round's own self-check and for the probe: the tail's
// [batch, slot, candidate] candidate rows to the head/accept layout, element-wise, at `count`
// rows. A pure relabel of the same 16 * K * batch words.
void dspine_bridge_tail_to_head(const std::int32_t* tail, std::int32_t* head, std::int32_t slots,
                                 std::int32_t drafts, std::int32_t batch);
void dspine_bridge_tail_to_head(const float* tail, float* head, std::int32_t slots,
                                 std::int32_t drafts, std::int32_t batch);
void dspine_bridge_head_to_tail(const std::int32_t* head, std::int32_t* tail, std::int32_t slots,
                                 std::int32_t drafts, std::int32_t batch);

} // namespace ninfer::models::qwen3_5::execution
