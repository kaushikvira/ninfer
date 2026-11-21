#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/execution/draft.h"
#include "models/qwen3_5/execution/dspine.h"
#include "models/qwen3_5/execution/text.h"
#include "ninfer/ops/sequence.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {
namespace {

execution::DraftExecutionView draft_view(RankProgramImpl& program) {
    auto& state = *program.resources.dflash;
    return {program.parameters,
            program.resources.work,
            program.device.execution_view(),
            state.local,
            state.full ? &*state.full : nullptr,
            program.proposal_head};
}

Tensor dense_prefix(const Tensor& tensor, std::initializer_list<std::int32_t> shape) {
    return Tensor({tensor.data, tensor.bytes(), tensor.device}, tensor.dtype, shape);
}

} // namespace

void RankProgramImpl::append_draft_history(ComputeOperation& operation) {
    if (!resources.dflash) return;
    // DSpine shares the masked-draft frame and the drafter's paged context state
    // (planning/startup.cpp:113), so this call is reached by a DSpine round. Everything below the
    // block-arrangement scan is backend-generic - it packs the target's tap field out of the
    // residency state with `prepare_ragged_prefix` and hands the three streams to the backend's
    // own context append - and the DFlash2 body of that append is what reads
    // `parameters.profile->config.draft` (execution/draft.cpp:44) and `parameters.draft->` (ibid
    // :96,:98,:109), both empty for a DSpine profile (models/common/dspine/config.h:96-97). DSpine
    // takes the sibling arm at execution::dspine_append_context, named in the overload below.
    const bool forward    = std::holds_alternative<ForwardWork>(operation.request.work.operation);
    const auto batch      = forward ? 1 : static_cast<int>(operation.shape.rows);
    std::uint32_t minimum = std::numeric_limits<std::uint32_t>::max(), maximum = 0;
    const auto rows = forward ? std::span(operation.rows).subspan(operation.forward_row, 1)
                              : std::span(operation.rows);
    for (const auto& row : rows) {
        const auto count = row.main_frontier - row.backend_frontier;
        minimum          = std::min(minimum, count);
        maximum          = std::max(maximum, count);
    }
    if (!maximum) return;
    append_draft_history(batch, {minimum, maximum});
}

void RankProgramImpl::append_draft_history(std::uint32_t batch_size,
                                           ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    auto& frame      = resources.frame;
    auto& state      = *resources.dflash;
    auto& draft      = *frame.dflash;
    const auto batch = static_cast<std::int32_t>(batch_size);
    const auto width = static_cast<std::int32_t>(draft_window + 1);
    auto scope       = resources.work.scope();
    auto features = resources.work.alloc(DType::BF16, {state.pending_features.ne[0], width, batch});
    auto positions = dense_prefix(draft.append_positions, {width, batch});
    auto counts    = draft.append_counts.slice(0, 0, batch);
    ops::prepare_ragged_prefix(state.pending_features, frame.residency_slots.slice(0, 0, batch),
                               frame.backend_frontiers.slice(0, 0, batch),
                               frame.frontiers.slice(0, 0, batch), features, positions, counts,
                               device.execution_view());
    if (backend == SpeculativeBackend::Dspine) {
        // The DSpine drafter's context update: the target-tap field is projected and normed once
        // and each of its five layers then appends its own context key and value to its own paged
        // pages, which are the pages its layer's attention reads. It takes the drafter's paged
        // context and no cyclic State records at all - the drafter declares no sliding window
        // (planning/startup.cpp:113-120) - so it takes `backend_table_rows` for the page-table row
        // and no `state_destination_offsets`.
        if (!state.full)
            throw std::logic_error("DSpine draft context: the drafter has no paged context");
        execution::dspine_append_context(
            execution::DspineExecutionView{parameters, resources.work, device.execution_view(),
                                           &*state.full},
            features, positions, counts, frame.backend_table_rows.slice(0, 0, batch), envelope);
        return;
    }
    execution::draft_append_context(draft_view(*this), features, positions, counts,
                                    frame.state_destination_offsets.slice(0, 0, batch),
                                    frame.backend_table_rows.slice(0, 0, batch), envelope);
}

void RankProgramImpl::run_dflash_draft(std::uint32_t batch_value, std::uint32_t minimum,
                                       std::uint32_t maximum) {
    auto& frame      = resources.frame;
    auto& draft      = *frame.dflash;
    const int batch  = static_cast<int>(batch_value);
    const int drafts = static_cast<int>(draft_window), width = drafts + 1;
    execution::DraftInputs inputs{frame.anchors.slice(0, 0, batch),
                                  frame.frontiers.slice(0, 0, batch),
                                  (backend == SpeculativeBackend::DFlash2
                                       ? frame.proposal_valid_columns
                                       : frame.valid_columns)
                                      .slice(0, 0, batch),
                                  frame.state_destination_offsets.slice(0, 0, batch),
                                  frame.backend_table_rows.slice(0, 0, batch),
                                  frame.live_rows.slice(0, 0, batch),
                                  frame.sampling};
    execution::DraftOutputs outputs{
        dense_prefix(draft.proposal_ids, {width, batch}),
        dense_prefix(draft.proposal_positions, {width, batch}),
        dense_prefix(draft.drafts, {drafts, batch}),
        draft.candidate_ids.data
            ? dense_prefix(draft.candidate_ids, {draft.candidate_ids.ne[0], drafts, batch})
            : Tensor{},
        draft.proposal_q.data
            ? dense_prefix(draft.proposal_q, {draft.proposal_q.ne[0], drafts, batch})
            : Tensor{}};
    execution::DraftEnvelopes envelopes{{minimum, maximum}, {minimum, maximum}};
    execution::draft_propose(draft_view(*this), inputs, outputs, envelopes);
}

// The DSpine draft seam: the same shape of call the decode loop already makes for DFlash2
// (run_dflash_draft above), for the backend whose proposal tail lives in
// execution::dspine_propose_tail. It hands the round the last bridge's three streams, the
// normed proposal block, the row anchors and the draw positions, and receives drafts,
// candidate_ids and proposal_q in the frame's accept-layout streams - the same three streams
// decide_decode then hands `speculative_accept_sparse_drafts`, which is why the accepted-token
// accounting needs no DSpine arm at all (decision.cpp:204-211 reads the accept op's counts).
// The DSpine round: the five draft layers, then the proposal tail. Two executor calls, in the
// order the round's data flows:
//
//   execution::dspine_walk          the mask block, the embedding, the five layers over the
//                                  drafter's own paged context (the pages the per-round
//                                  append_draft_history filled), the code bridge after each of
//                                  them; it publishes the last write's stash, scale, message and
//                                  normed block into the frame's five DSpine streams.
//   execution::dspine_propose_tail  the tower head's top-16, the score table over the real tg and
//                                  tsv, the sampled tail, and the layout bridges into the accept
//                                  op's own contract.
//
// BATCH 1, WRITTEN DOWN RATHER THAN HIDDEN. The tail's `bridge_message` input is handed
// `draft.bridge_message` UNPREFIXED a few lines below (the frame's region is
// {dspine_rank, width, batch_capacity} at program/frames.cpp:196-203, so the Tensor the tail sees
// has ne[2] = the frame's capacity, not this round's batch). At batch 1 the two are the same
// grid; at batch > 1 the tail reads the message of a wider block than the walk wrote, which is a
// silent wrong-proposal bug, not a loud one. So this seam REFUSES a batch above one, in words, at
// the call site. Everything on the accept side of it (drafts, candidate_ids, proposal_q) and the
// walk's own geometry are already batch-shaped; the widening is one `slice` at the tail's input
// and a re-check of the tail's own shape gate - recorded here, and in D-45 where the same
// limitation is recorded, and deliberately not taken in this stage.
void RankProgramImpl::run_dspine_draft(std::uint32_t batch_value, std::uint32_t minimum,
                                        std::uint32_t maximum) {
    auto& frame = resources.frame;
    auto& draft = *frame.dflash;
    const int batch = static_cast<int>(batch_value);
    const auto& dspine = parameters.profile->config.dspine;
    if (!parameters.dspine || !dspine)
        throw std::logic_error("run_dspine_draft: the model carries no dspine drafter");
    if (batch != 1)
        throw std::logic_error(
            "DSpine drafts one sequence per round: the frame's bridge message region is read "
            "unprefixed by the batch (program/frames.cpp:196-203), so batch > 1 would share one "
            "row's proposal across the block. The DFlash2 arm of this loop is the batched one.");
    if (!resources.dflash || !resources.dflash->full)
        throw std::logic_error("run_dspine_draft: the drafter has no paged context");
    const auto width = static_cast<int>(draft_window) + 1;
    // The walk, on the drafter's own paged context. Its context lengths, page-table rows and
    // validity are the drafter half of the frame - the same trio the DFlash2 drafter's full
    // context reads - and its envelope is the one the decode loop computed for this round.
    const execution::DspineWalkInputs walk_inputs{
        frame.anchors.slice(0, 0, batch), frame.frontiers.slice(0, 0, batch),
        frame.valid_columns.slice(0, 0, batch), frame.backend_table_rows.slice(0, 0, batch)};
    execution::DspineWalkOutputs walk_outputs{
        dense_prefix(draft.proposal_ids, {width, batch}),
        dense_prefix(draft.proposal_positions, {width, batch}),
        dense_prefix(draft.bridge_raw, {frame.spec.hidden, width, batch}),
        dense_prefix(draft.bridge_rms, {width * batch}),
        dense_prefix(draft.bridge_message, {int(dspine->bridge.rank), width, batch}),
        dense_prefix(draft.proposal_hidden, {frame.spec.hidden, width, batch})};
    execution::dspine_walk(
        execution::DspineExecutionView{parameters, resources.work, device.execution_view(),
                                       &*resources.dflash->full},
        walk_inputs, walk_outputs,
        ops::ContextAttentionExecutionEnvelope{minimum, maximum});
    // The head comes from the drafter's own binding, i.e. from the head's `dspine/final_hidden`
    // use, not from the text tower's binding of the same object (which carries that arm's
    // `text/final_hidden` policy).
    const execution::DspineTailView view{parameters.dspine->output_head.weight,
                                         parameters.dspine->final_norm,
                                         parameters.dspine->gate_bias,
                                         parameters.dspine->tg,
                                         parameters.dspine->tsv,
                                         static_cast<int>(dspine->bridge.rank)};
    execution::DspineTailInputs inputs{
        dense_prefix(draft.bridge_raw, {frame.spec.hidden, int(draft_window) + 1, batch}),
        dense_prefix(draft.bridge_rms, {(int(draft_window) + 1) * batch}),
        draft.bridge_message,
        dense_prefix(draft.proposal_hidden, {frame.spec.hidden, int(draft_window) + 1, batch}),
        frame.anchors.slice(0, 0, batch), frame.sampling_positions.slice(0, 0, batch),
        frame.sampling};
    execution::DspineTailOutputs outputs{
        dense_prefix(draft.drafts, {static_cast<int>(draft_window), batch}),
        dense_prefix(draft.candidate_ids,
                     {draft.candidate_ids.ne[0], static_cast<int>(draft_window), batch}),
        dense_prefix(draft.proposal_q,
                     {draft.proposal_q.ne[0], static_cast<int>(draft_window), batch})};
    execution::dspine_propose_tail(view, inputs, outputs, resources.work,
                                   device.execution_view());
    CUDA_CHECK(cudaGetLastError());
}

void RankProgramImpl::begin_draft(ComputeOperation& operation) {
    auto& frame           = resources.frame;
    const int batch       = static_cast<int>(operation.shape.rows);
    const int drafts      = static_cast<int>(draft_window);
    const int width       = drafts + 1;
    std::uint32_t minimum = capacity, maximum = 0;
    for (const auto& row : operation.rows) {
        minimum = std::min(minimum, row.main_frontier);
        maximum = std::max(maximum, row.main_frontier);
    }
    if (!graphs) gpu_timer.begin_draft();
    if (graphs) {
        (void)graphs->start(GraphPhase::Draft, operation.shape.rows, maximum, 0,
                            device.execution_view());
    } else if (backend == SpeculativeBackend::Mtp) {
        begin_mtp_draft(operation);
    } else if (backend == SpeculativeBackend::Dspine) {
        run_dspine_draft(operation.shape.rows, minimum, maximum);
    } else {
        run_dflash_draft(operation.shape.rows, minimum, maximum);
    }
    if (!graphs) gpu_timer.end_draft();
    Tensor proposals = backend == SpeculativeBackend::Mtp
                           ? frame.mtp->proposals(batch)
                           : dense_prefix(frame.dflash->drafts, {drafts, batch});
    auto input       = frame.decode_view(width, batch);
    ops::prepare_verify_ids(frame.anchors.slice(0, 0, batch), proposals,
                            frame.extents.slice(0, 0, batch), input.ids, device.execution_view());
    const bool constrained = std::any_of(operation.sampling.begin(), operation.sampling.end(),
                                         [](const auto& row) { return row.constrained; });
    if (constrained) {
        if (backend == SpeculativeBackend::Mtp) {
            const auto row_bytes = std::size_t(batch) * sizeof(TokenId);
            CUDA_CHECK(cudaMemcpy2DAsync(frame.host_drafts.data, row_bytes, proposals.data,
                                         frame.mtp->drafts.nb[1], row_bytes, operation.drafts,
                                         cudaMemcpyDeviceToHost, device.stream()));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(frame.host_drafts.data, proposals.data, proposals.bytes(),
                                       cudaMemcpyDeviceToHost, device.stream()));
        }
        drafts_ready.record(device.stream());
    }
}

void RankProgramImpl::begin_target(ComputeOperation& operation) {
    if (!operation.width) return;
    auto& frame        = resources.frame;
    const bool forward = std::holds_alternative<ForwardWork>(operation.request.work.operation) ||
                         std::holds_alternative<ScoreWork>(operation.request.work.operation);
    const int batch = forward ? 1 : static_cast<int>(operation.shape.rows);
    const int width = static_cast<int>(operation.width);
    ++operation.attention.forwards;
    operation.attention.rows += forward ? 1 : operation.batch;
    operation.attention.scheduled_positions +=
        std::uint64_t(width) * (forward ? 1 : operation.batch);
    std::uint32_t active_tokens = forward ? width : 0;
    if (!forward)
        for (std::uint32_t row = 0; row < operation.batch; ++row)
            active_tokens += frame.host_ingress.valid_columns[row];
    operation.attention.valid_positions += active_tokens;
    auto input = forward ? frame.input_view(width, batch) : frame.decode_view(width, batch);
    if (!forward && graphs) {
        const auto phase       = operation.speculative ? GraphPhase::Verify : GraphPhase::Main;
        std::uint32_t frontier = 0;
        for (const auto& row : operation.rows) frontier = std::max(frontier, row.main_frontier);
        operation.graph_started = true;
        operation.projected     = true;
        operation.row_finished = graphs->start(phase, operation.shape.rows, frontier, active_tokens,
                                               device.execution_view());
        return;
    }
    execution::TextInputs inputs{input.ids,
                                 input.cache_positions,
                                 input.rope_positions,
                                 frame.valid_columns.slice(0, 0, batch),
                                 frame.main_table_rows.slice(0, 0, batch),
                                 frame.state_source_offsets.slice(0, 0, batch),
                                 frame.state_destination_offsets.slice(0, 0, batch),
                                 operation.input_embeddings};
    inputs.active_tokens              = active_tokens;
    execution::TextStateUpdate update = execution::StepUpdate{};
    if (forward) {
        const auto& row = operation.rows[operation.forward_row];
        update = execution::SequenceUpdate{row.state_offsets.source, row.state_offsets.destination};
    } else if (operation.speculative) {
        update.emplace<execution::ReplayRecords>(*resources.replay_records);
    }
    std::uint32_t minimum = capacity, maximum = 0;
    for (std::uint32_t row = 0; row < (forward ? 1 : operation.batch); ++row) {
        const auto end = static_cast<std::uint32_t>(frame.host_ingress.frontiers[row] +
                                                    frame.host_ingress.valid_columns[row]);
        minimum        = std::min(minimum, end);
        maximum        = std::max(maximum, end);
    }
    std::optional<execution::DFlashFeatureSink> features;
    if (resources.dflash) {
        auto& state = *resources.dflash;
        features.emplace();
        // The target taps the feature field is packed by. DFlash's and DFlash2's come out of
        // `config.draft`; DSpine's out of `config.dspine` (D-20: [1,16,31,46,61], NOT prod's
        // [5,19,33,47,61]) - the same tap list planning/startup.cpp:126 sized
        // `spec.dflash.feature_rows` from. The two configs are populated mutually exclusively
        // (models/common/dspine/config.h:96-97), so the single deref this replaces read a nullopt
        // for a DSpine plan.
        features->layers = parameters.profile->config.dspine
                               ? parameters.profile->config.dspine->target_layer_ids
                               : parameters.profile->config.draft->target_layer_ids;
        if (forward) {
            features->features  = state.prefill_features.slice(1, 0, width);
            features->positions = state.prefill_positions.slice(0, 0, width);
        } else {
            features->features        = state.pending_features.slice(1, 0, width);
            features->residency_slots = frame.residency_slots.slice(0, 0, batch);
        }
    }
    resources.work.reset();
    if (forward) prefill_timer.start();
    if (!forward) gpu_timer.begin_target();
    operation.text = std::make_unique<execution::TextRun>(
        execution::TextExecutionView{parameters, resources.work, device.execution_view(),
                                     resources.decoder.text_kv, resources.state_images.linear(),
                                     resources.moe.get()},
        std::move(inputs), update, ops::CausalAttentionExecutionEnvelope{minimum, maximum},
        input.hidden, features ? &*features : nullptr);
    operation.row_finished = operation.text->start();
}

bool RankProgramImpl::advance_target(ComputeOperation& operation) {
    if (!operation.width) return true;
    if (!operation.row_finished)
        operation.row_finished = operation.graph_started ? graphs->poll() : operation.text->poll();
    if (!operation.row_finished) return false;
    workspace_peak = std::max(workspace_peak, resources.work.peak_used());
    operation.text.reset();
    return true;
}

} // namespace ninfer::models::qwen3_5::detail
