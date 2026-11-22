#include "models/qwen3_5/program/internal.h"

#include "models/qwen3_5/execution/draft.h"
#include "models/qwen3_5/execution/dspine.h"
#include "ninfer/ops/sequence.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {

void RankProgramImpl::commit_draft_history(ComputeOperation& operation) {
    if (!resources.dflash) return;
    if (!operation.commit) throw std::logic_error("draft history commit has no accepted prefix");
    std::uint32_t maximum = 0;
    for (const auto& row : operation.commit->rows)
        if (row.terminal && !row.discarded) maximum = std::max(maximum, row.accepted_tokens);
    if (maximum == 0) return;

    auto& frame       = resources.frame;
    auto& draft       = *frame.dflash;
    auto& state       = *resources.dflash;
    const auto width  = static_cast<std::int32_t>(draft_window + 1);
    const auto batch  = static_cast<std::int32_t>(operation.shape.rows);
    const auto cuda   = device.execution_view();
    const auto prefix = [](const Tensor& storage, std::initializer_list<std::int32_t> shape) {
        return Tensor({storage.data, storage.bytes(), storage.device}, storage.dtype, shape);
    };
    auto scope = resources.work.scope();
    Tensor features =
        resources.work.alloc(DType::BF16, {state.pending_features.ne[0], width, batch});
    auto positions = prefix(draft.append_positions, {width, batch});
    auto counts    = draft.append_counts.slice(0, 0, batch);
    ops::prepare_ragged_prefix(state.pending_features, frame.residency_slots.slice(0, 0, batch),
                               frame.draft_append_starts.slice(0, 0, batch),
                               frame.draft_append_ends.slice(0, 0, batch), features, positions,
                               counts, cuda);
    // The commit half of the drafter's context update, the same pair of arms as the prefill half
    // (execution.cpp:74-104) and the decode half (decode.cpp:70-82). `draft_append_context` reads
    // `parameters.draft`, empty for a DSpine profile, so a DSpine round stops at its first commit.
    if (backend == SpeculativeBackend::Dspine) {
        if (!state.full) {
            throw std::logic_error("DSpine draft context: the drafter has no paged context");
        }
        execution::dspine_append_context(
            execution::DspineExecutionView{parameters, resources.work, cuda, &*state.full},
            features, positions, counts, frame.backend_table_rows.slice(0, 0, batch),
            {0, maximum});
        return;
    }
    execution::draft_append_context({parameters, resources.work, cuda, state.local,
                                     state.full ? &*state.full : nullptr, proposal_head},
                                    features, positions, counts,
                                    frame.commit_destination_offsets.slice(0, 0, batch),
                                    frame.backend_table_rows.slice(0, 0, batch), {0, maximum});
}

} // namespace ninfer::models::qwen3_5::detail
