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

#include "models/common/dspine/config.h"
#include "models/common/dspine/dspine_serial_tail.h"
#include "models/qwen3_5/execution/linear.h"
#include "models/qwen3_5/planning.h"
#include "ninfer/ops/dspine_bridge.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_split.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/linear_topk.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/sequence.h"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
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
    // The round zero (D-42). `dspine_serial_tail` writes ONLY the sender row its walk reaches
    // per slot - src/models/common/dspine/dspine_serial_tail.cpp:258-268 - so the other
    // `slots - 1` rows of every slot are whatever this arena slot held. Step 5 reads the WHOLE
    // grid back and publishes all of it as the accept op's `proposal_q`, so without this the
    // round would publish the previous round's sender rows: a round would not be a function of
    // its inputs, which is the property D-42 proved over 32 rounds (`0 of 37 streams part`) and
    // the property that makes repeated decode rounds correct. The probe gets this by zeroing
    // the stream in its `reset_round`; this is the round's own copy of that zero, and the only
    // stream of the tail that needs it (the table, the candidate rows and the index are written
    // whole by steps 2 to 4).
    CUDA_CHECK(cudaMemsetAsync(tail_q.data, 0, tail_q.bytes(), cuda.stream()));
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

namespace {

// The drafter geometry, read out of the profile the plan was built from. Every dimension the
// walk names comes from the DSpine component - NOT from the tower's - the same rule the target
// tap list follows in program/decode.cpp:232 (D-20's taps are [1,16,31,46,61], not prod's).
struct WalkGeometry {
    std::int32_t hidden, taps, code, rank, intermediate, head_dim, query_heads, key_value_heads;
    std::int32_t query, key;
    float rms_norm_eps, rope_theta, bridge_eps;
};

[[nodiscard]] const models::dspine::Config& drafter(const Parameters& parameters) {
    if (!parameters.profile->config.dspine)
        throw std::logic_error("dspine draft: the model profile carries no dspine drafter");
    return *parameters.profile->config.dspine;
}

[[nodiscard]] const DspineParameters& drafter_parameters(const Parameters& parameters) {
    if (!parameters.dspine)
        throw std::logic_error("dspine draft: the model carries no dspine drafter");
    if (parameters.dspine->layers.empty())
        throw std::logic_error("dspine draft: the drafter carries no draft layers");
    return *parameters.dspine;
}

[[nodiscard]] WalkGeometry walk_geometry(const Parameters& parameters) {
    const auto& draft = drafter(parameters);
    return WalkGeometry{dimension(draft.hidden_size),
                         dimension(draft.target_layer_ids.size()),
                         dimension(draft.bridge.code_size),
                         dimension(draft.bridge.rank),
                         dimension(draft.intermediate_size),
                         dimension(draft.head_dim),
                         dimension(draft.num_attention_heads),
                         dimension(draft.num_key_value_heads),
                         dimension(draft.query_width()),
                         dimension(draft.key_width()),
                         draft.rms_norm_eps,
                         draft.rope_theta,
                         draft.bridge.eps};
}

[[nodiscard]] std::int32_t columns_of(const Tensor& stream, const char* what) {
    if (!stream.data || stream.ne[0] <= 0 || stream.ne[1] <= 0)
        throw std::invalid_argument(std::string("dspine draft: ") + what + " is unavailable");
    return stream.ne[1];
}

} // namespace

PagedKVBatchLayerView DspineExecutionView::context_layer(std::uint32_t layer) const {
    if (!context) throw std::logic_error("dspine draft: the drafter has no paged context");
    return context->batch_layer_view(layer);
}

// The DSpine draft-context append: the half of the drafter's per-round context update that the
// DFlash2 path runs in `append_context_impl` (execution/draft.cpp:41). The steps are the ones
// `test_dspine_forward_real.cpp` labels "feature projection of the target taps", "context norm"
// and "context key/value projections, key norm, RoPE, append to layer N's pages" - in that order,
// on the same ops, with the packed-parent split for the context key and value.
void dspine_append_context(DspineExecutionView view, const Tensor& features,
                           const Tensor& positions, const Tensor& counts,
                           const Tensor& table_rows,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    const auto& draft   = drafter(view.parameters);
    const auto& weights = drafter_parameters(view.parameters);
    const auto shape    = walk_geometry(view.parameters);
    if (!features.data || !positions.data || !counts.data || !table_rows.data)
        throw std::invalid_argument("dspine draft context: an input is unavailable");
    const auto width  = columns_of(features, "the target-tap field");
    const auto batch  = features.ne[2];
    const auto rows   = features.ne[0];
    if (rows != shape.hidden * shape.taps || batch < 1 ||
        positions.ne[0] != width || positions.ne[1] != batch || counts.ne[0] != batch ||
        table_rows.ne[0] != batch || features.dtype != DType::BF16)
        throw std::invalid_argument("dspine draft context: the round geometry is out of range");
    if (std::uint64_t(draft.num_hidden_layers) != weights.layers.size())
        throw std::logic_error("dspine draft context: the drafter's layers and its profile part");

    const auto columns = width * batch;
    auto scope         = view.work.scope();
    auto projected     = view.work.alloc(DType::BF16, {shape.hidden, columns});
    auto context       = view.work.alloc(DType::BF16, {shape.hidden, columns});
    project(features.view({rows, columns}), weights.feature_projection, projected, view.work,
            view.cuda);
    ops::rmsnorm(projected, weights.context_norm, shape.rms_norm_eps, false, context, view.cuda);

    // Every draft layer projects the SAME normed target features with ITS OWN context key and
    // value, and those are the K and V ranges of the layer's packed Q/K/V object. The split
    // family is the one compiled route to them (see dspine.h), and the Q rows it also computes
    // land in a scratch stream and are dropped: a measured cost of the missing context geometry,
    // the same one the probe records at its own "context key/value projections" step.
    for (std::size_t layer = 0; layer < weights.layers.size(); ++layer) {
        auto layer_scope = view.work.scope();
        const auto& block = weights.layers[layer];
        auto key_raw = view.work.alloc(DType::BF16, {shape.key, columns});
        auto value   = view.work.alloc(DType::BF16, {shape.key, columns});
        auto query_scratch = view.work.alloc(DType::BF16, {shape.query, columns});
        const std::array streams{query_scratch, key_raw, value};
        ops::linear_split(context, block.query_key_value.weight, streams, view.cuda);
        auto key = view.work.alloc(DType::BF16, {shape.key, columns});
        ops::rmsnorm(key_raw, block.key_norm, shape.rms_norm_eps, false, key, view.cuda);
        ops::rope(positions.view({columns}), shape.head_dim, shape.rope_theta, key, view.cuda);
        ops::kv_cache_append_prefix(
            key.view({shape.head_dim, shape.key_value_heads, width, batch}),
            value.view({shape.head_dim, shape.key_value_heads, width, batch}),
            positions, counts, table_rows, envelope, view.context_layer(std::uint32_t(layer)),
            view.cuda);
    }
}

namespace {

// One layer's code bridge, the seven steps the probe runs as `run_bridge`, in its order:
// the read (rms of the layer stream), the code projection, normalize_codes over the real
// codebook (the anchor column's row is the codebook's anchor row), the gate, predecessor and
// value projections of the codes into ONE packed [3R, width, batch] block, the message, the
// write projection, and the finish that refines the residual IN PLACE and fuses the next
// layer's input norm. `stash` is the pre-refinement residual the serial tail reads, and only
// the LAST layer's stash, rms and message leave the walk.
struct BridgeStreams {
    Tensor normalized, rms, codes, codes_out, packed, message, delta;
    // The write's fused-norm outputs, which the NEXT layer reads as its input norm and the
    // last layer's turn publishes to the tail: `normed` is the proposal hidden and `stash` is
    // the tail's raw stream. `anchors` and `width` are normalize_codes' anchor-column rule.
    Tensor normed, stash, anchors;
    std::int32_t width = 0;
};

void dspine_bridge(DspineExecutionView view, BridgeStreams& streams, Tensor& residual,
                   const WalkGeometry& shape, std::size_t layer, bool last) {
    const auto& weights = drafter_parameters(view.parameters);
    const auto& bridge  = weights.bridge;
    if (layer >= bridge.read.size() || layer >= bridge.write.size())
        throw std::logic_error("dspine draft: the bridge is missing a layer's read or write");
    const auto columns = residual.ne[0] * residual.ne[1];
    auto codes_in   = streams.codes.view({shape.code, columns});
    auto codes_out  = streams.codes_out.view({shape.code, columns});
    auto codes_post = streams.codes.view({shape.code, columns});
    auto delta      = streams.delta.view({shape.hidden, columns});
    auto normed     = streams.normed.view({shape.hidden, columns});
    auto stash      = streams.stash.view({shape.hidden, columns});
    auto message    = streams.message.view({shape.rank, columns});
    // The read: rms of the layer stream, the scale `dspine_finish` and `dspine_serial_scores`
    // both read back.
    ops::dspine_prepare_read(residual, shape.bridge_eps, streams.normalized, streams.rms,
                              view.cuda.stream());
    // The code projection, then normalize_codes over the real codebook - whose anchor column
    // (t = 0) is the codebook's row for the row's anchor token.
    // The per-layer read and write are the Weight forms: one row range of the bridge's stacked
    // object, issued with the drafter's own policy rather than a prepared projection's.
    ops::linear(codes_in, bridge.read[layer], codes_out, ops::LinearPolicy::A16Only, view.work,
                view.cuda);
    ops::dspine_normalize_codes(codes_out, streams.anchors, bridge.codebook, streams.width,
                                shape.bridge_eps, codes_post, view.cuda.stream());
    // The gate, the predecessor gate and the value projection of the codes, into ONE packed
    // [3R, columns] block - the three R-row slices of one arena allocation.
    {
        const auto plane = std::size_t(std::int64_t(shape.rank) * columns) * 2;
        const auto base  = static_cast<std::byte*>(streams.packed.data);
        auto gate = Tensor(base, view.cuda.device(), DType::BF16, {shape.rank, columns, 1, 1});
        auto predecessor =
            Tensor(base + plane, view.cuda.device(), DType::BF16, {shape.rank, columns, 1, 1});
        auto previous =
            Tensor(base + 2 * plane, view.cuda.device(), DType::BF16, {shape.rank, columns, 1, 1});
        project(codes_post, bridge.gate, gate, view.work, view.cuda);
        project(codes_post, bridge.predecessor_gate, predecessor, view.work, view.cuda);
        project(codes_post, bridge.prev, previous, view.work, view.cuda);
    }
    ops::dspine_message(streams.packed, bridge.gate_bias, message, view.cuda.stream());
    // The write projection, and the finish: it refines the residual IN PLACE (that is why the
    // walk hands the bridge the residual by reference) and fuses the NEXT layer's input norm,
    // which is why the next layer's input norm is taken from the write and not recomputed.
    ops::linear(message, bridge.write[layer], delta, ops::LinearPolicy::A16Only, view.work,
                view.cuda);
    const auto& next = last ? weights.final_norm : weights.layers[layer + 1].input_norm;
    ops::dspine_finish(residual, streams.rms, delta, next, shape.rms_norm_eps, normed, &stash,
                       view.cuda.stream());
}
} // namespace

// The five-layer walk. Step-for-step this is `run_forward` from the probe, with the context
// half lifted out into `dspine_append_context` (the engine runs the drafter's context once per
// round on the ACCEPTED columns, not once per round inside the block), and the head, score
// table and sampled tail left in `dspine_propose_tail`, which already owned them.
void dspine_walk(DspineExecutionView view, const DspineWalkInputs& inputs,
                 DspineWalkOutputs& outputs, ops::ContextAttentionExecutionEnvelope envelope) {
    const auto& weights = drafter_parameters(view.parameters);
    const auto& draft   = drafter(view.parameters);
    const auto shape    = walk_geometry(view.parameters);
    if (!inputs.anchors.data || !inputs.frontiers.data || !inputs.valid_columns.data ||
        !inputs.table_rows.data || !outputs.ids.data || !outputs.positions.data ||
        !outputs.bridge_raw.data || !outputs.bridge_rms.data || !outputs.bridge_message.data ||
        !outputs.proposal_hidden.data)
        throw std::invalid_argument("dspine draft: an input is unavailable");
    const auto batch  = inputs.anchors.ne[0];
    const auto width  = outputs.ids.ne[0];
    const auto drafts = width - 1;
    if (batch < 1 || width < 2 || std::uint32_t(drafts) > kDFlashDecodeMaximumDrafts ||
        std::uint64_t(width) != draft.block_size ||
        std::uint64_t(weights.layers.size()) != draft.num_hidden_layers ||
        outputs.ids.ne[1] != batch || outputs.positions.ne[0] != width ||
        outputs.positions.ne[1] != batch || outputs.proposal_hidden.ne[0] != shape.hidden ||
        outputs.proposal_hidden.ne[1] != width || outputs.proposal_hidden.ne[2] != batch ||
        outputs.bridge_raw.ne[0] != shape.hidden || outputs.bridge_raw.ne[1] != width ||
        outputs.bridge_raw.ne[2] != batch ||
        outputs.bridge_rms.ne[0] != width * batch ||
        outputs.bridge_message.ne[0] != shape.rank || outputs.bridge_message.ne[1] != width ||
        outputs.bridge_message.ne[2] != batch)
        throw std::invalid_argument("dspine draft: the round geometry is out of range");
    if (!view.context)
        throw std::logic_error("dspine draft: the drafter has no paged context to attend over");

    const auto columns = width * batch;
    // The round reset: the arena the walk's transients live in is rewound, exactly as the
    // engine's own masked-draft round opens (execution/draft.cpp:347). See dspine.h for what is
    // and is not part of it.
    view.work.reset();
    auto scope = view.work.scope();

    auto residual = view.work.alloc(DType::BF16, {shape.hidden, columns});
    ops::prepare_masked_block(inputs.anchors, inputs.frontiers, inputs.valid_columns,
                               dimension(draft.mask_token_id), outputs.ids, outputs.positions,
                               view.cuda);
    ops::embedding(outputs.ids.view({columns}), weights.token_embedding, residual, view.cuda);

    // The streams the layers share, allocated once for the round out of the round's own arena.
    auto hidden       = view.work.alloc(DType::BF16, {shape.hidden, columns});
    auto projected    = view.work.alloc(DType::BF16, {shape.hidden, columns});
    auto intermediate = view.work.alloc(DType::BF16, {shape.intermediate, columns});
    auto query_raw    = view.work.alloc(DType::BF16, {shape.query, columns});
    auto key_raw      = view.work.alloc(DType::BF16, {shape.key, columns});
    auto value        = view.work.alloc(DType::BF16, {shape.key, columns});
    auto query        = view.work.alloc(DType::BF16, {shape.query, columns});
    auto key          = view.work.alloc(DType::BF16, {shape.key, columns});
    auto attended     = view.work.alloc(DType::BF16, {shape.query, columns});
    BridgeStreams bridge{
        view.work.alloc(DType::BF16, {shape.hidden, columns}),   // normalized
        view.work.alloc(DType::FP32, {columns}),                 // rms
        view.work.alloc(DType::BF16, {shape.code, columns}),     // codes
        view.work.alloc(DType::BF16, {shape.code, columns}),     // codes_out
        view.work.alloc(DType::BF16, {3 * shape.rank, columns}), // packed
        view.work.alloc(DType::BF16, {shape.rank, columns}),     // message
        view.work.alloc(DType::BF16, {shape.hidden, columns})};  // delta
    bridge.anchors = inputs.anchors;
    bridge.width   = width;
    auto normed = view.work.alloc(DType::BF16, {shape.hidden, columns});
    auto stash  = view.work.alloc(DType::BF16, {shape.hidden, columns});
    bridge.normed = normed;
    bridge.stash  = stash;

    const ops::AttentionHeadGeometry geometry{shape.head_dim, shape.query_heads,
                                              shape.key_value_heads};
    const auto scale = float(1.0 / std::sqrt(double(shape.head_dim)));

    // The bridge of the previous layer hands the next layer a stream that is ALREADY normed
    // (its write fuses that norm), so the input norm runs only where the write did not.
    bool pre_normed = false;
    for (std::size_t layer = 0; layer < weights.layers.size(); ++layer) {
        auto layer_scope = view.work.scope();
        const auto& block = weights.layers[layer];
        if (pre_normed) {
            CUDA_CHECK(cudaMemcpyAsync(hidden.data, normed.data, hidden.bytes(),
                                       cudaMemcpyDeviceToDevice, view.cuda.stream()));
        } else {
            ops::rmsnorm(residual, block.input_norm, shape.rms_norm_eps, false, hidden,
                         view.cuda);
        }
        {
            const std::array streams{query_raw, key_raw, value};
            ops::linear_split(hidden, block.query_key_value.weight, streams, view.cuda);
        }
        ops::rmsnorm(query_raw, block.query_norm, shape.rms_norm_eps, false, query, view.cuda);
        ops::rmsnorm(key_raw, block.key_norm, shape.rms_norm_eps, false, key, view.cuda);
        ops::rope(outputs.positions.view({columns}), shape.head_dim, shape.rope_theta, query, key,
                  view.cuda);
        ops::context_softmax_attention(
            query.view({shape.head_dim, shape.query_heads, width, batch}),
            key.view({shape.head_dim, shape.key_value_heads, width, batch}),
            value.view({shape.head_dim, shape.key_value_heads, width, batch}), inputs.frontiers,
            inputs.valid_columns, inputs.table_rows, geometry, scale,
            ConstPagedKVBatchLayerView(view.context_layer(std::uint32_t(layer))), envelope,
            view.work, attended, view.cuda);
        // The attention output projection is [hidden, query_width], and this engine's linear_add
        // carries its own shape table, which has not that geometry - so the residual add is
        // taken as its own op. Same math as the DFlash2 path's `project_add`, one launch apart,
        // and the probe records the geometry gap at the same step rather than behind a new
        // kernel.
        project(attended, block.output, projected, view.work, view.cuda);
        ops::residual_add(projected, residual, view.cuda);
        ops::rmsnorm(residual, block.post_attention_norm, shape.rms_norm_eps, false, hidden,
                      view.cuda);
        project_swiglu(hidden, block.mlp.gate_up, intermediate, view.work, view.cuda);
        project_add(intermediate, block.mlp.down, residual, view.work, view.cuda);
        dspine_bridge(view, bridge, residual, shape, layer,
                      layer + 1 == weights.layers.size());
        pre_normed = true;
    }

    // The last fused write is the tail's input. Its stash is the tail's `raw`, its rms the
    // tail's read scale, its message the tail's packed Q, and its normed block the proposal
    // hidden the tower's head scores - published into the frame's own streams, which is where
    // `dspine_propose_tail` reads them from.
    CUDA_CHECK(cudaMemcpyAsync(outputs.proposal_hidden.data, normed.data,
                               std::size_t(std::int64_t(shape.hidden) * columns) * 2,
                               cudaMemcpyDeviceToDevice, view.cuda.stream()));
    CUDA_CHECK(cudaMemcpyAsync(outputs.bridge_raw.data, stash.data,
                               std::size_t(std::int64_t(shape.hidden) * columns) * 2,
                               cudaMemcpyDeviceToDevice, view.cuda.stream()));
    CUDA_CHECK(cudaMemcpyAsync(outputs.bridge_rms.data, bridge.rms.data,
                               std::size_t(std::int64_t(columns)) * 4, cudaMemcpyDeviceToDevice,
                               view.cuda.stream()));
    CUDA_CHECK(cudaMemcpyAsync(outputs.bridge_message.data, bridge.message.data,
                               std::size_t(std::int64_t(shape.rank) * columns) * 2,
                               cudaMemcpyDeviceToDevice, view.cuda.stream()));
    (void)scope;
}

// The round's peak, term by term, in the order the walk opens its streams. `columns` is the
// block's width by the batch, and every stream the walk opens for the WHOLE round is one
// `columns`-wide field, so the arithmetic is a sum of widths, not a scan of the layers: the layers
// run in scopes of their own, so only the widest of them is ever live.
std::size_t dspine_round_workspace_bytes(const models::dspine::Config& draft,
                                         std::int32_t batch, std::int32_t width,
                                         ops::ContextAttentionExecutionEnvelope envelope,
                                         std::int32_t multiprocessors) {
    const auto columns = std::size_t(dimension(batch) * dimension(width));
    const auto hidden  = dimension(draft.hidden_size);
    const auto code    = dimension(draft.bridge.code_size);
    const auto rank    = dimension(draft.bridge.rank);
    const auto query   = dimension(draft.query_width());
    const auto key     = dimension(draft.key_width());
    std::size_t wide = 0;
    // The walk's round-long streams: the residual and the five [hidden, columns] fields it keeps
    // alongside it (the hidden, the out projection, the bridge's write, its fused norm, its stash
    // and its read), the widened field of the SwiGLU mix, the three [query] and three [key] fields
    // of the attention, the code field in and out, and the packed gate, predecessor and value
    // block with its message out. Each is bfloat16, plus the read's per-column fp32 scale.
    wide += 7 * hidden + dimension(draft.intermediate_size) + 3 * query + 3 * key + 2 * code +
            4 * rank;
    // The append's round-long pair: the projected taps and the normed field they become, both
    // [hidden, columns], and the widest of its per-layer scopes - the context key and value and
    // the key the norm makes, plus the query rows the packed split also writes and drops.
    wide += 2 * hidden + 3 * key + query;
    // The A16 staging of the widest projection input the round feeds (the SwiGLU's widened field,
    // which is wider than the attention out's input and than the bridge's code field at every
    // shipped geometry), one byte a row - the same slack term the drafter's own round sizes.
    wide += dimension(draft.intermediate_size) / 2;
    auto bytes = columns * (wide * 2 + 4);
    // The one stream of the walk that is not a width times columns field: the layer's attention
    // over the drafter's paged context, whose workspace the op's own capacity names.
    const ops::AttentionHeadGeometry geometry{dimension(draft.head_dim),
                                              dimension(draft.num_attention_heads),
                                              dimension(draft.num_key_value_heads)};
    (void)multiprocessors;   // the paged context attention's capacity is a function of the
                             // envelope and the block, not of the machine's grid (the op's own
                             // capacity note says so); the parameter stays because the drafter's
                             // round sizing passes it and a local attention may need it.
    bytes += ops::context_softmax_attention_workspace_capacity_bytes(
        geometry, envelope, dimension(width), dimension(width), dimension(batch));
    return bytes;
}

} // namespace ninfer::models::qwen3_5::execution
