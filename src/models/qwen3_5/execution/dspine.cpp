// P2 stage 4: the DSpine proposal tail in the engine's execution layer - the half of a DSpine
// proposal round from the tower head onward, publishing the drafts, the candidate rows and the
// FP32 proposal distribution in the layouts `speculative_accept_sparse_drafts` verifies them
// in. See dspine.h for the seam it attaches to and for the three layouts it bridges.
//
// Every step here is a call into a verified op: `linear_topk` over the tower's own head (P0b,
// D-37's Q8/NVFP4 gather gate), `dspine_serial_scores` over the real tg/tsv tables (P0b:
// `serial_scores.idx` BIT-EXACT against the Tencent reference), `dspine_serial_tail` (K1+K2:
// bit-exact against the CPU oracle at the production policy and at T=0, D-24/D-37). This file
// implements no proposal mathematics, takes no copy of a weight and quantizes nothing.

#include "models/qwen3_5/execution/dspine.h"

#include "models/common/dspine/dspine_serial_tail.h"
#include "models/qwen3_5/planning.h"
#include "ninfer/ops/dspine_bridge.h"
#include "ninfer/ops/linear_topk.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_5::execution {
namespace {

using ninfer::models::common::dspine::kSerialTailSlots;

// The head's valid row count: rows at or beyond it never participate, and the accept op names
// the same domain (src/ops/sampling/acceptance/sparse_drafts/entry.cpp:12).
constexpr std::int32_t kTokenDomain = 248077;

[[nodiscard]] std::int32_t extent(const Tensor& value, const char* what) {
    if (!value.data || value.ne[0] <= 0)
        throw std::invalid_argument(std::string("dspine proposal tail: ") + what +
                                     " is unavailable");
    return value.ne[0];
}

// One stream-sequenced crossing of the boundary between the device round and the host, in both
// directions: the bytes ride the round's own stream, and the crossing settles that stream, so
// what follows reads what the steps above wrote. This is the move the probe's
// `seed_tail_inputs` makes for the same reason (test_dspine_forward_real.cpp:863-890, D-40's
// second hidden probe defect: "unsequenced host copies").
template <class Value>
void settle_to_host(std::vector<Value>& destination, const void* source,
                    CudaExecutionView execution, const char* what) {
    CUDA_CHECK(cudaMemcpyAsync(destination.data(), source, destination.size() * sizeof(Value),
                                cudaMemcpyDeviceToHost, execution.stream()));
    CUDA_CHECK(cudaStreamSynchronize(execution.stream()));
    (void)what;
}

template <class Value>
void settle_to_device(void* destination, const std::vector<Value>& source,
                      CudaExecutionView execution, const char* what) {
    CUDA_CHECK(cudaMemcpyAsync(destination, source.data(), source.size() * sizeof(Value),
                                cudaMemcpyHostToDevice, execution.stream()));
    CUDA_CHECK(cudaStreamSynchronize(execution.stream()));
    (void)what;
}

// head: (candidate, slot, batch) at candidate + slots * (slot + drafts * batch)
// tail: (batch, slot, candidate) at batch_index + batch * (slot + drafts * candidate)
// The two are the same grid read with its axes exchanged, with the unequal 16-versus-K stride
// D-40's RED 2 was about; a relabel of one grid's words, never an interpolation of them.
template <class Value>
void bridge_head_to_tail(const Value* head, Value* tail, std::int32_t slots, std::int32_t drafts,
                         std::int32_t batch) {
    for (std::int32_t row = 0; row < batch; ++row)
        for (std::int32_t slot = 0; slot < drafts; ++slot)
            for (std::int32_t candidate = 0; candidate < slots; ++candidate)
                tail[std::size_t(row)
                     + std::int64_t(batch)
                               * (std::int64_t(slot) + std::int64_t(drafts) * candidate)] =
                    head[std::size_t(candidate)
                         + std::int64_t(slots)
                                   * (std::int64_t(slot) + std::int64_t(drafts) * row)];
}

template <class Value>
void bridge_tail_to_head(const Value* tail, Value* head, std::int32_t slots, std::int32_t drafts,
                         std::int32_t batch) {
    for (std::int32_t row = 0; row < batch; ++row)
        for (std::int32_t slot = 0; slot < drafts; ++slot)
            for (std::int32_t candidate = 0; candidate < slots; ++candidate)
                head[std::size_t(candidate)
                     + std::int64_t(slots)
                               * (std::int64_t(slot) + std::int64_t(drafts) * row)] =
                    tail[std::size_t(row)
                         + std::int64_t(batch)
                                   * (std::int64_t(slot) + std::int64_t(drafts) * candidate)];
}

} // namespace

void dspine_bridge_head_to_tail(const std::int32_t* head, std::int32_t* tail, std::int32_t slots,
                                 std::int32_t drafts, std::int32_t batch) {
    bridge_head_to_tail(head, tail, slots, drafts, batch);
}

void dspine_bridge_tail_to_head(const std::int32_t* tail, std::int32_t* head, std::int32_t slots,
                                 std::int32_t drafts, std::int32_t batch) {
    bridge_tail_to_head(tail, head, slots, drafts, batch);
}

void dspine_bridge_tail_to_head(const float* tail, float* head, std::int32_t slots,
                                 std::int32_t drafts, std::int32_t batch) {
    bridge_tail_to_head(tail, head, slots, drafts, batch);
}

void dspine_propose_tail(const DspineTailView& view, const DspineTailInputs& inputs,
                          DspineTailOutputs& outputs, MemoryArena& work,
                          CudaExecutionView execution) {
    if (!inputs.sampling || !inputs.bridge_raw.data || !inputs.bridge_rms.data ||
        !inputs.bridge_message.data || !inputs.proposal_hidden.data || !inputs.anchors.data ||
        !inputs.draw_positions.data || !outputs.drafts.data || !outputs.candidate_ids.data ||
        !outputs.proposal_q.data)
        throw std::invalid_argument("dspine proposal tail: an input is unavailable");

    const auto batch  = extent(inputs.anchors, "anchors");
    const auto hidden = extent(inputs.proposal_hidden, "proposal hidden rows");
    const auto width  = inputs.proposal_hidden.ne[1];
    const auto drafts = outputs.drafts.ne[0];
    const auto slots  = static_cast<std::int32_t>(kSerialTailSlots);
    if (batch < 1 || drafts < 1 || std::uint32_t(drafts) > kDFlashDecodeMaximumDrafts ||
        width != drafts + 1 || outputs.drafts.ne[1] != batch ||
        outputs.candidate_ids.ne[0] != slots || outputs.candidate_ids.ne[1] != drafts ||
        outputs.candidate_ids.ne[2] != batch || outputs.proposal_q.ne[0] != slots ||
        outputs.proposal_q.ne[1] != drafts || outputs.proposal_q.ne[2] != batch)
        throw std::invalid_argument("dspine proposal tail: the round geometry is out of range");
    if (inputs.bridge_message.ne[0] != view.rank ||
        inputs.bridge_raw.ne[0] != hidden || inputs.bridge_raw.ne[1] != width ||
        inputs.bridge_rms.ne[0] != width * batch)
        throw std::invalid_argument("dspine proposal tail: a bridge stream is out of shape");

    const auto& weights = view;
    // The proposal head is the TOWER's own output head, so it arrives in the representation the
    // tower declares (Q8_G32_FP16 on the prod V3 tower, D-37) and is gathered by candidate id
    // inside each op. No copy of it exists: D-37 sized that copy at 1.3 - 2.5 GiB of VRAM the
    // drafter is not provisioned for.
    const auto head = view.head;
    const auto cuda = execution;

    const std::size_t grid_words = std::size_t(std::int64_t(batch) * slots * drafts);
    auto scope                   = work.scope();

    // 1. The proposal columns: the block's columns 1..K. Column 0 is the anchor, which is not a
    //    proposal, so in the column-major [hidden, width, batch] form the proposals are the
    //    (width-1)*hidden wide, batch tall strided block behind it - the same 2-D gather form the
    //    DFlash2 proposal path uses (draft.cpp:452-463).
    auto proposals = work.alloc(DType::BF16, {hidden, drafts, batch});
    {
        const auto row_bytes    = std::size_t(std::int64_t(hidden) * drafts) * 2;
        const auto source_pitch = std::size_t(std::int64_t(hidden) * width) * 2;
        CUDA_CHECK(cudaMemcpy2DAsync(proposals.data, row_bytes,
                                     static_cast<const std::byte*>(inputs.proposal_hidden.data)
                                         + std::size_t(std::int64_t(hidden) * 2),
                                     source_pitch, row_bytes, std::size_t(batch),
                                     cudaMemcpyDeviceToDevice, cuda.stream()));
    }

    // 2. The head's top-`slots` per proposal slot, at the head layout (candidate rank fastest).
    //    That layout IS the accept op's `candidate_ids` layout (kernel.cuh:109), so this writes
    //    the frame's stream directly and no bridge is needed for it.
    auto scores = work.alloc(DType::FP32, {slots, drafts, batch});
    ops::linear_topk(proposals, head, kTokenDomain, outputs.candidate_ids, scores, work, cuda);

    // 3. The score table: the head's base scores and the last bridge's message over the real
    //    tg/tsv tables, at the head's candidate layout. `index` is serial_scores' argmax chain,
    //    the greedy pair's twin of the tail's `idx`.
    auto index = work.alloc(DType::I32, {slots, drafts, batch});
    auto table = work.alloc(DType::FP32, {batch, slots, slots, drafts});
    ops::dspine_serial_scores(outputs.candidate_ids, inputs.bridge_raw, inputs.bridge_rms,
                               inputs.bridge_message, inputs.anchors, view.final_norm,
                               view.gate_bias, head, view.tg, view.tsv, width, index,
                               &table, cuda.stream());

    // 4. The tail runs at its own layouts. Both crossings of that boundary are the bridges
    //    dspine.h names; each moves 16 * K words per row, which is the whole of a round's
    //    proposal grid (240 words, 15 KiB at a prod round). They settle the stream, so the route
    //    that takes them is the eager one - the route DSpine is on, since graphs.cpp:213 names no
    //    Graph profile for the backend. Recording it as a cost of this stage, not hiding it.
    auto tail_candidates = work.alloc(DType::I32, {batch, drafts, slots});
    auto tail_drafts     = work.alloc(DType::I32, {batch, drafts});
    auto tail_q          = work.alloc(DType::FP32, {batch, drafts, slots});
    auto tail_index      = work.alloc(DType::I32, {batch, drafts, slots});
    {
        std::vector<std::int32_t> head_words;
        head_words.resize(grid_words);
        settle_to_host(head_words, outputs.candidate_ids.data, cuda, "head candidates");
        std::vector<std::int32_t> tail_words;
        tail_words.resize(grid_words);
        bridge_head_to_tail(head_words.data(), tail_words.data(), slots, drafts, batch);
        settle_to_device(tail_candidates.data, tail_words, cuda, "tail candidates");
    }

    ops::dspine_serial_tail(table, tail_candidates, inputs.sampling, inputs.draw_positions,
                             tail_drafts, tail_q, tail_index, cuda.stream());
    CUDA_CHECK(cudaGetLastError());

    // 5. Publish the round in the accept op's own contract: drafts [K, batch] contiguous (the op
    //    reads it through `token_rows`, so it needs the contiguous form the tail's (batch, slot)
    //    is not), candidate_ids [16, K, batch] (step 2 wrote those) and proposal_q [16, K, batch]
    //    FP32 contiguous. Both crossings are relabels of the same words.
    {
        std::vector<float> q_words;
        q_words.resize(grid_words);
        settle_to_host(q_words, tail_q.data, cuda, "tail proposal distribution");
        std::vector<float> head_q;
        head_q.resize(grid_words);
        bridge_tail_to_head(q_words.data(), head_q.data(), slots, drafts, batch);
        settle_to_device(outputs.proposal_q.data, head_q, cuda, "proposal distribution");

        std::vector<std::int32_t> tail_ids;
        tail_ids.resize(std::size_t(batch) * std::size_t(drafts));
        settle_to_host(tail_ids, tail_drafts.data, cuda, "tail drafts");
        std::vector<std::int32_t> accept_ids(tail_ids.size());
        for (std::int32_t row = 0; row < batch; ++row)
            for (std::int32_t slot = 0; slot < drafts; ++slot)
                accept_ids[std::size_t(slot) + std::int64_t(drafts) * row] =
                    tail_ids[std::size_t(row) + std::int64_t(batch) * std::int64_t(slot)];
        settle_to_device(outputs.drafts.data, accept_ids, cuda, "drafts");
    }
    (void)scope;
}

} // namespace ninfer::models::qwen3_5::execution
