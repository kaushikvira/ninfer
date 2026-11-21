// P2 stage 5: the PORT-FIDELITY probe for the DSpine round in the executor (D-49's open item).
//
// THE GAP THIS CLOSES. `execution::dspine_walk` and `execution::dspine_append_context` exist in
// the executor (merged `dfb0bcbf`) and have never executed: no probe drove them, and the decode
// loop that does sits behind a full target tower. D-49 therefore graded the port STRUCTURAL
// (step order, op identity, shapes read out of the probe) and not bit-parity. This probe drives
// the executor's own `dspine_append_context` -> `dspine_walk` -> `dspine_propose_tail` over
// `parameters.dspine` for N >= 32 rounds on the real V3 drafter and compares, per stream, against
// the reviewed specification - the forward probe's walk (`run_forward`/`run_bridge`,
// test_dspine_forward_real.cpp) - on identical inputs. The result is a port-fidelity table:
// bit-exact / max delta in bf16 ulp / divergent, per published stream. The probe is the
// specification and passes 22/22: nothing here widens a tolerance, copies a weight, edits a kernel
// or touches a numeric.
//
// WHAT IS UNDER TEST ON THE EXECUTOR SIDE, AND WHY THE LOAD IS THE EXECUTOR'S OWN
//   binding    the drafter is loaded the way the engine loads it: `read_definition` ->
//              `execution::describe_model` -> `resolve_load` -> `materialize_model` ->
//              `execution::Parameters`, i.e. through `Prepare::dspine`
//              (execution/parameters.cpp:267-336) - the five layers' packed Q/K/V parent, the
//              per-layer READ and WRITE slices of the two STACKED bridge objects
//              (`weight_bank.cpp`'s matrix form), the aliased context K/V, the tower's own
//              embedding and head. The specification side builds the same objects the probe's
//              way (`Probe::packed_parent`/`Probe::stacked`/`Probe::parameter`,
//              test_dspine_forward_real.cpp:280-333). Every one of those pairs is printed with
//              its device address, n, k and representation: that table is where a stacked-object
//              slice would be seen to be wrong, and the two walks then run on THEIR OWN pairs.
//   the walk   `dspine_append_context` (feature projection, context norm, the per-layer packed
//              split into that layer's own paged pages, key norm, RoPE, append), then
//              `dspine_walk` (mask block, embedding, 5 x {input norm, split Q/K/V, query and key
//              head norms + RoPE, paged context attention, out + residual, post norm, SwiGLU,
//              down} + the 7-step code bridge after each layer), then `dspine_propose_tail`
//              (the proposal gather, the tower head's top-16, the score table, the sampled tail,
//              the layout bridges).
//   the arena  the walk's share is sized by `dspine_round_workspace_bytes` (the same call
//              program/planning/startup.cpp:435 makes at plan time) and its peak is printed, so
//              the plan-time sizing is exercised, not just asserted.
//   D-42       every round opens with the round reset, and both arms are checked for
//              round-to-round determinism on identical inputs.
//
// ONE SHARD, NOT TWO. The two arms share ONE materialization (the executor's own shard): the
// tower plus the drafter of this artifact is 24.115 GiB of objects and GPU 1 has ~25.4 GiB free,
// so a second copy of the drafter is not affordable and the probe would not open. The binding
// comparison is then a statement about the two BINDINGS of one set of bytes, which is the thing
// that must be under test; the streams the two arms write are separate allocations, as are their
// paged contexts and their arenas, so neither arm can read the other's bytes.
//
// THE SKELETON IS NOT INVENTED HERE. Every line of device bring-up - the `Probe` struct and its
// `Device`/`DeviceContext` use, the artifact `Reader` open, the `buffer` allocation, the
// `CudaDeviceScope` clearing and the `upload`/`download` transfer boundary - is copied verbatim
// from tests/models/qwen3_5/test_dspine_tail_cost.cpp, which copies it verbatim from the forward
// probe (#71: manifest 0.29 s, 55 device objects, 3.83186 GiB, 3514 walk steps). D-47 records two
// agents lost to a stop after "manifest parse ends" behind a freshly written device-bind setup;
// the LOCKED rule is that the working path is the only accepted setup, and a stop that recurs with
// a verbatim copy is a stop to report, not to debug. The paged drafter context is the engine's own
// `PagedKVCache` (state/decoder_state.h) built as tests/models/qwen3_5/test_mtp_history_real.cpp
// builds one, because the thing under test is the executor's context geometry.
//
// NINFER_TEST_ARTIFACT  the V3-tailored DSpine artifact (skip, exit 77, when unset)
// NINFER_TEST_DEVICE    the visible CUDA ordinal to run on; required, no default (D-29)
// NINFER_TEST_ROUNDS    rounds of identical inputs (default 32, floor 32)
// NINFER_FIDELITY_SLACK MiB of device headroom the load insists on (default 768)
//
// Run it pinned to the free card: CUDA_VISIBLE_DEVICES=1 NINFER_TEST_DEVICE=0 ...

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/placement.h"
#include "artifact/reader.h"
#include "artifact/views.h"
#include "core/arena.h"
#include "core/buffer.h"
#include "core/context_pool.h"
#include "core/device.h"
#include "core/layout.h"
#include "core/paged_kv_cache.h"
#include "core/paged_kv_geometry.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "core/weight_view.h"
#include "models/common/bindings.h"
#include "models/common/dspine/config.h"
#include "models/common/dspine/dspine_serial_tail.h"
#include "models/common/dspine/weights.h"
#include "models/qwen3_5/execution/dspine.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/execution/profile.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "ninfer/ops/dspine_bridge.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_split.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/linear_topk.h"
#include "ninfer/ops/projection_weights.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/sequence.h"
#include "ninfer/ops/softmax_attention.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::models;
namespace dspine    = ninfer::models::dspine;
namespace q35       = ninfer::models::qwen3_5;
namespace execution = q35::execution;

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
constexpr std::int32_t kSlots = 16;
// One production proposal round: batch 1, the artifact's 16-wide block, 15 proposal slots.
constexpr std::int32_t kBatch = 1;
// The drafter's paged context is carried over this many positions, one page. The block is 16 wide
// and the context is 16 wide, so every projection the round issues runs at T = 16, the shape
// gate's own second point, and the context attention sees 16 keys.
constexpr std::int32_t kContext = 16;
// The head's valid row domain (execution/dspine.cpp:38).
constexpr std::int32_t kTokenDomain = 248077;

void report_fault(const int signal) {
    std::puts("SIGNAL: a device-side fault in the harness itself");
    std::fflush(stdout);
    std::quick_exit(90 + signal);
}

std::string setting(std::string_view name, std::string_view fallback = {}) {
    const char* value = std::getenv(std::string(name).c_str());
    return value && *value ? std::string(value) : std::string(fallback);
}

void cuda_ok(const cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(error));
    }
}

std::uint32_t field(const artifact::Json& config, const char* name) {
    return static_cast<std::uint32_t>(artifact::require_u64(config.at(name), name));
}

[[nodiscard]] std::int32_t dimension(const std::uint64_t value) {
    return static_cast<std::int32_t>(value);
}

[[nodiscard]] std::int32_t dimension(const std::int32_t value) { return value; }

// A config dimension, in whatever unsigned width the config names it by.
template <class Value>
[[nodiscard]] std::int32_t dim(const Value value) {
    return static_cast<std::int32_t>(value);
}

// ---------------------------------------------------------------- the probe (the working probe's) ---

struct Probe {
    Probe(const std::string& path, const int ordinal)
        : reader(path), device(Device::cuda(ordinal)), context(ordinal) {
        cuda_ok(cudaSetDevice(ordinal), "set device");
        cudaDeviceProp properties{};
        cuda_ok(cudaGetDeviceProperties(&properties, ordinal), "device properties");
        std::printf("device: visible ordinal %d, name \"%s\", pci %02x:%02x, %llu MiB total\n",
                    ordinal, properties.name, properties.pciBusID, properties.pciDeviceID,
                    static_cast<unsigned long long>(properties.totalGlobalMem / 1024 / 1024));
        std::printf("        CUDA_VISIBLE_DEVICES=\"%s\"\n",
                    setting("CUDA_VISIBLE_DEVICES", "<unset>").c_str());
    }

    [[nodiscard]] std::pair<std::uint64_t, std::uint64_t> memory() const {
        std::uint64_t free_bytes = 0, total_bytes = 0;
        cuda_ok(cudaMemGetInfo(&free_bytes, &total_bytes), "memory info");
        return {free_bytes, total_bytes};
    }

    [[nodiscard]] Tensor buffer(const std::size_t bytes, const DType dtype,
                                 const std::initializer_list<std::int32_t> shape) {
        storage.push_back(std::make_unique<DeviceBuffer>(device, bytes));
        auto& slot = *storage.back();
        const CudaDeviceScope clearing(device);
        cuda_ok(cudaMemset(slot.data(), 0, slot.size()), "clear buffer");
        return Tensor(slot.span(), dtype, shape);
    }

    CudaExecutionView cuda() const { return context.execution_view(); }

    artifact::Reader reader;
    Device device;
    DeviceContext context;
    std::vector<std::unique_ptr<DeviceBuffer>> storage;
};

// The probe built by the running case. The device boundary helpers are free functions, so they
// need the live probe to reach its buffers; one process runs one probe (RUN_SERIAL).
namespace {
Probe* owner = nullptr;
Device work_device;
}  // namespace

// The buffer that owns a device pointer, so a copy goes through the transfer boundary the
// engine's own buffers provide (they name their device, order the stream, and settle the copy).
DeviceBuffer* owning_buffer(void* pointer) {
    for (const auto& slot : owner->storage) {
        const auto base      = static_cast<std::byte*>(slot->data());
        const auto target    = static_cast<std::byte*>(pointer);
        if (target >= base && target < base + slot->size()) {
            return slot.get();
        }
    }
    throw std::runtime_error("the probe copied to a buffer it does not own");
}

void upload(const void* destination, const void* source, const std::size_t bytes,
            const char* what) {
    (void)what;
    if (destination == nullptr || bytes == 0) {
        throw std::runtime_error(std::string("the probe uploaded ") + what +
                                 " to a destination it does not own");
    }
    const auto target = static_cast<std::byte*>(const_cast<void*>(destination));
    auto* buffer      = owning_buffer(destination ? const_cast<void*>(destination) : nullptr);
    buffer->copy_from_host(source, bytes, std::size_t(target - static_cast<std::byte*>(
                                                        buffer->data())));
}

void download(void* destination, const void* source, const std::size_t bytes, const char* what) {
    (void)what;
    const auto target = static_cast<std::byte*>(const_cast<void*>(source));
    auto* buffer      = owning_buffer(const_cast<void*>(source));
    buffer->copy_to_host(destination, bytes,
                          std::size_t(target - static_cast<std::byte*>(buffer->data())));
}

// A deterministic BF16 activation field, column-major: the same bytes for the same shape, on the
// host and therefore on the device. The round's activations are synthetic carriers by design
// (D-42: acceptance needs the target, which this probe does not load); its weights are the
// artifact's, which is what makes the comparison a measurement of the code and not of the data.
void seed(const Tensor& stream) {
    const auto columns = std::int64_t(stream.ne[1]) * stream.ne[2];
    const auto rows    = std::int64_t(stream.ne[0]);
    std::vector<std::uint16_t> host(std::size_t(rows * columns));
    for (std::int64_t column = 0; column < columns; ++column) {
        for (std::int64_t row = 0; row < rows; ++row) {
            const float value =
                1.7f * std::sin(0.00390621f * float(row + 1) + 0.217747f * float(column + 3));
            const auto bits = std::bit_cast<std::uint32_t>(value);
            host[std::size_t(column) * rows + row] =
                static_cast<std::uint16_t>((bits >> 16) + ((bits >> 16) & 1u));
        }
    }
    upload(stream.data, host.data(), host.size() * 2, "seed stream");
}

void upload_i32(const Tensor& stream, const std::vector<std::int32_t>& values, const char* what) {
    upload(stream.data, values.data(), values.size() * sizeof(std::int32_t), what);
}

[[nodiscard]] std::vector<std::int32_t> download_i32(const Tensor& stream) {
    std::vector<std::int32_t> values(std::size_t(stream.numel()));
    download(values.data(), stream.data, values.size() * sizeof(std::int32_t), "read i32 stream");
    return values;
}

[[nodiscard]] std::vector<std::uint16_t> download_bf16(const Tensor& stream) {
    std::vector<std::uint16_t> values(std::size_t(stream.numel()));
    download(values.data(), stream.data, values.size() * 2, "read bf16 stream");
    return values;
}

[[nodiscard]] std::vector<float> download_f32(const Tensor& stream) {
    std::vector<float> values(std::size_t(stream.numel()));
    download(values.data(), stream.data, values.size() * sizeof(float), "read fp32 stream");
    return values;
}

// A FNV-1a digest over bytes (the forward probe's fingerprint primitive, :1443).
[[nodiscard]] std::uint64_t digest(const void* bytes, const std::size_t count) {
    const auto* data = static_cast<const std::uint8_t*>(bytes);
    std::uint64_t hash = 14695981039346656037ULL;
    for (std::size_t index = 0; index < count; ++index) {
        hash ^= static_cast<std::uint64_t>(data[index]);
        hash *= 1099511628211ULL;
    }
    return hash;
}

// One step of the specification walk. In the forward probe a step the engine refuses is a GATE it
// counts and prints; here a refused step voids the parity claim, so it names itself and ends the
// run with the probe's exit status.
template <class Function>
void step(const char* label, Function&& body) {
    try {
        body();
        cuda_ok(cudaGetLastError(), "launch");
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string("GATE at ") + label + ": " + error.what());
    }
}

} // namespace
namespace {

// ---------------------------------------------------------------- the drafter's paged context ----
//
// The engine's own `PagedKVCache` over a device page arena and a fixed execution table, built the
// way tests/models/qwen3_5/test_mtp_history_real.cpp:72-93 builds one: `plan_paged_kv_cache` for a
// drafter with `layers` layers at the drafter's own head_dim and kv_heads, one physical page per
// logical page (the context is 16 tokens, the page is 64), identity-mapped. One instance per arm,
// so the two arms' pages are separate bytes; the geometry is the executor's, which is what the
// round's context attention is being asked about.
struct DrafterContext {
    DrafterContext(const std::uint32_t layers, const std::int32_t head_dim,
                   const std::int32_t kv_heads, const std::int32_t capacity) {
        LayoutBuilder builder;
        const auto layout =
            q35::plan_paged_kv_cache(builder, layers, dimension(capacity), kv_heads, head_dim,
                                       KvCacheStorage::BFloat16, kBatch);
        pages_ = layout.pages.page_bytes;
        tables_.emplace(owner->device, builder.finish(256, "dspine fidelity page tables"));
        records_.emplace(owner->device, pages_ + layout.pages.alignment);
        records_->fill(0);
        cache_ = std::make_unique<q35::PagedKVCache>(records_->span(), tables_->span(), layout);
        staging_.emplace(sizeof(std::int64_t) * std::size_t(layout.execution_tables.spec
                                                                .logical_page_capacity));
        offsets_.assign(std::size_t(layout.execution_tables.spec.logical_page_capacity), 0);
        cache_->tables().publish(0, 0, std::span<std::int64_t>(offsets_),
                                  staging_->span(), owner->cuda());
        owner->context.synchronize();
    }

    void clear() { records_->fill(0); }

    [[nodiscard]] PagedKVBatchLayerView layer(const std::size_t index) const {
        return cache_->batch_layer_view(std::uint32_t(index));
    }

    [[nodiscard]] ConstMemorySpan bytes() const { return records_->span(); }

    std::uint64_t pages_ = 0;
    std::optional<DeviceBuffer> records_, tables_;
    std::optional<PinnedHostBuffer> staging_;
    std::vector<std::int64_t> offsets_;
    std::unique_ptr<q35::PagedKVCache> cache_;
};

// ---------------------------------------------------------------- the round's streams ----------
//
// One instance per arm, at the shapes the specification names them by
// (test_dspine_forward_real.cpp:556-626) and the executor's contract names them by
// (execution/dspine.h:150-170, :196-215). `stash`, `rms`, `message` and `normed` are the four
// published frame streams: the last fused write's pre-refinement residual, its per-column read
// scale, its packed message and its normed block, i.e. `DspineWalkOutputs`'s bridge_raw,
// bridge_rms, bridge_message and proposal_hidden.
struct Round {
    Tensor ids, positions, anchors, lengths, valid, table_rows, context_positions, context_lengths,
        context_counts, taps;
    Tensor projected, context, context_key, context_value, context_key_raw, scratch_query;
    Tensor hidden, residual, normalized, rms, codes, codes_out, packed, message, delta, normed,
        stash, intermediate, query, key, value, attended, query_raw, key_raw;
    Tensor proposal, candidates, scores, idx_scores, table, tail_cand, tail_drafts, tail_q,
        tail_index, index, configs, draw_positions;
    Tensor drafts, candidate_ids, proposal_q;
};

[[nodiscard]] Round allocate(const std::int32_t h, const std::int32_t w, const std::int32_t b,
                              const std::int32_t ctx, const std::int32_t c, const std::int32_t r,
                              const std::int32_t taps, const std::int32_t q, const std::int32_t k,
                              const std::int32_t d, const std::int32_t inter,
                              const std::int32_t drafts) {
    Round out;
    const auto integers = [&](const std::int64_t count,
                               const std::initializer_list<std::int32_t> shape) {
        return owner->buffer(std::size_t(4) * count, DType::I32, shape);
    };
    const auto stream = [&](const std::int64_t count,
                             const std::initializer_list<std::int32_t> shape) {
        return owner->buffer(std::size_t(2) * count, DType::BF16, shape);
    };
    const auto reals = [&](const std::int64_t count,
                            const std::initializer_list<std::int32_t> shape) {
        return owner->buffer(std::size_t(4) * count, DType::FP32, shape);
    };
    out.ids               = integers(w * b, {w, b, 1, 1});
    out.positions         = integers(w * b, {w, b, 1, 1});
    out.context_positions = integers(ctx * b, {ctx, b, 1, 1});
    out.context_lengths   = integers(b, {b, 1, 1, 1});
    out.context_counts    = integers(b, {b, 1, 1, 1});
    out.lengths           = integers(b, {b, 1, 1, 1});
    out.valid             = integers(b, {b, 1, 1, 1});
    out.table_rows        = integers(b, {b, 1, 1, 1});
    out.anchors           = integers(b, {b, 1, 1, 1});
    out.taps              = stream(std::int64_t(h) * taps * ctx * b, {h, taps, ctx, b});
    out.projected         = stream(std::int64_t(h) * ctx * b, {h, ctx, b, 1});
    out.context           = stream(std::int64_t(h) * ctx * b, {h, ctx, b, 1});
    out.hidden            = stream(std::int64_t(h) * w * b, {h, w, b, 1});
    out.residual          = stream(std::int64_t(h) * w * b, {h, w, b, 1});
    out.normalized         = stream(std::int64_t(h) * w * b, {h, w, b, 1});
    out.delta             = stream(std::int64_t(h) * w * b, {h, w, b, 1});
    out.normed            = stream(std::int64_t(h) * w * b, {h, w, b, 1});
    out.stash             = stream(std::int64_t(h) * w * b, {h, w, b, 1});
    out.proposal          = stream(std::int64_t(h) * drafts * b, {h, drafts, b, 1});
    out.intermediate      = stream(std::int64_t(inter) * w * b, {inter, w, b, 1});
    out.rms               = reals(std::int64_t(w) * b, {w * b, 1, 1, 1});
    out.codes             = stream(std::int64_t(c) * w * b, {c, w, b, 1});
    out.codes_out         = stream(std::int64_t(c) * w * b, {c, w, b, 1});
    out.packed            = stream(std::int64_t(3 * r) * w * b, {3 * r, w, b, 1});
    out.message           = stream(std::int64_t(r) * w * b, {r, w, b, 1});
    out.query             = stream(std::int64_t(q) * w * b, {d, q / d, w, b});
    out.query_raw         = stream(std::int64_t(q) * w * b, {d, q / d, w, b});
    out.scratch_query     = stream(std::int64_t(q) * ctx * b, {q, ctx, b, 1});
    out.key               = stream(std::int64_t(k) * w * b, {d, k / d, w, b});
    out.key_raw           = stream(std::int64_t(k) * w * b, {d, k / d, w, b});
    out.value             = stream(std::int64_t(k) * w * b, {d, k / d, w, b});
    out.attended          = stream(std::int64_t(q) * w * b, {d, q / d, w, b});
    out.context_key       = stream(std::int64_t(k) * ctx * b, {d, k / d, ctx, b});
    out.context_key_raw   = stream(std::int64_t(k) * ctx * b, {d, k / d, ctx, b});
    out.context_value     = stream(std::int64_t(k) * ctx * b, {d, k / d, ctx, b});
    out.candidates        = integers(std::int64_t(kSlots) * drafts * b, {kSlots, drafts, b, 1});
    out.scores            = reals(std::int64_t(kSlots) * drafts * b, {kSlots, drafts, b, 1});
    out.idx_scores        = integers(std::int64_t(kSlots) * drafts * b, {kSlots, drafts, b, 1});
    out.table             = reals(std::int64_t(b) * kSlots * kSlots * drafts,
                                  {b, kSlots, kSlots, drafts});
    out.tail_cand         = integers(std::int64_t(b) * drafts * kSlots, {b, drafts, kSlots, 1});
    out.tail_drafts       = integers(std::int64_t(b) * drafts, {b, drafts, 1, 1});
    out.tail_q            = reals(std::int64_t(b) * drafts * kSlots, {b, drafts, kSlots, 1});
    out.tail_index        = integers(std::int64_t(b) * drafts * kSlots, {b, drafts, kSlots, 1});
    out.index             = integers(std::int64_t(b) * drafts * kSlots, {b, drafts, kSlots, 1});
    out.draw_positions    = integers(b, {b, 1, 1, 1});
    out.configs = owner->buffer(sizeof(ops::SamplingConfig) * std::size_t(b), DType::FP32,
                                 {b, 1, 1, 1});
    // The accept op's contract: the executor publishes these three (execution/dspine.h:203-215).
    out.drafts        = integers(std::int64_t(drafts) * b, {drafts, b, 1, 1});
    out.candidate_ids = integers(std::int64_t(kSlots) * drafts * b, {kSlots, drafts, b, 1});
    out.proposal_q    = reals(std::int64_t(kSlots) * drafts * b, {kSlots, drafts, b, 1});
    return out;
}

// The geometry the two arms read, in the units each of them names its streams by.
struct Geometry {
    std::int32_t h, w, b, ctx, c, r, taps, q, k, d, inter, drafts, qh, kvh;
    float rms_eps, rope_theta, bridge_eps;
    std::int32_t mask_token;
};

// ---------------------------------------------------------------- the round reset (D-42) -------
//
// The specification's reset is test_dspine_forward_real.cpp:1355-1435: zero the shared workspace
// and rewind its cursor, zero every stream the round writes in whole or in part, zero the draft
// layers' context pages, then establish the round's own inputs. Everything after the inputs is the
// round, and the round is a function of nothing but them. Both arms get their own copy, and the
// inputs are uploaded from the SAME host vectors, so the two arms start each round on identical
// bytes.
void reset_round(const Geometry& g, Round& round, DrafterContext& pages, MemoryArena& arena) {
    const CudaDeviceScope settling(owner->device);
    const auto scratch = arena.storage();
    cuda_ok(cudaMemset(scratch.data, 0, scratch.bytes), "reset the round's shared workspace");
    arena.reset();
    const std::vector<Tensor*> streams{
        &round.ids,           &round.positions,      &round.context_positions, &round.lengths,
        &round.valid,         &round.context_lengths, &round.context_counts,   &round.table_rows,
        &round.anchors,       &round.taps,           &round.projected,          &round.context,
        &round.hidden,        &round.residual,       &round.normalized,         &round.rms,
        &round.codes,         &round.codes_out,      &round.packed,             &round.message,
        &round.delta,         &round.normed,         &round.stash,              &round.intermediate,
        &round.query,         &round.key,            &round.value,              &round.attended,
        &round.context_key,   &round.context_value,  &round.scratch_query,      &round.query_raw,
        &round.key_raw,       &round.context_key_raw, &round.proposal,          &round.candidates,
        &round.scores,        &round.idx_scores,     &round.table,              &round.tail_cand,
        &round.tail_drafts,   &round.tail_q,         &round.tail_index,         &round.index,
        &round.draw_positions, &round.configs,       &round.drafts,             &round.candidate_ids,
        &round.proposal_q};
    for (auto* stream : streams) {
        cuda_ok(cudaMemset(stream->data, 0, stream->bytes()), "reset a round stream");
    }
    pages.clear();
    // The round's inputs: the block arrangement's scalars, the context geometry, the tail's draw
    // coordinate, the production sampling policy, and the tap field.
    const std::vector<std::int32_t> anchors{40901}, lengths{kContext}, valid{kSlots},
        context_lengths{kContext}, context_counts{kContext}, table_rows{0};
    upload_i32(round.anchors, anchors, "anchors");
    upload_i32(round.lengths, lengths, "block lengths");
    upload_i32(round.valid, valid, "valid columns");
    upload_i32(round.context_lengths, context_lengths, "context lengths");
    upload_i32(round.context_counts, context_counts, "context counts");
    upload_i32(round.table_rows, table_rows, "page table rows");
    std::vector<std::int32_t> context_positions(std::size_t(g.ctx * g.b));
    for (std::int32_t index = 0; index < g.ctx * g.b; ++index) {
        context_positions[std::size_t(index)] = index;
    }
    upload_i32(round.context_positions, context_positions, "context positions");
    std::vector<std::int32_t> draws(std::size_t(g.b));
    for (std::int32_t index = 0; index < g.b; ++index) {
        draws[std::size_t(index)] = 1024 + index;
    }
    upload_i32(round.draw_positions, draws, "tail draw positions");
    std::vector<ops::SamplingConfig> configs(std::size_t(g.b), ops::SamplingConfig{});
    for (auto& config : configs) {
        config.temperature = 0.9f;
        config.top_k       = 20;
        config.top_p       = 1.0f;
        config.min_p       = 0.05f;
        config.seed        = 0x50371153ULL;
    }
    upload(round.configs.data, configs.data(), configs.size() * sizeof(ops::SamplingConfig),
           "tail configs");
    seed(round.taps);
}

} // namespace
namespace {

// ---------------------------------------------------------- the specification's own binding ---
//
// The forward probe binds the drafter through `Bindings` + `dspine::bind_weights` and reaches a
// packed parent / a stacked slice with `over_object` (test_dspine_forward_real.cpp:303-333): the
// object's own view over the bytes the loader materialized once, with an element range, no copy
// and no second representation. Here the object views come from the engine's shard
// (`ModelShard::weight(id).view`), which is the same `bind_view` result, so the forms are the
// probe's verbatim and the bytes are the executor's. The executor reaches the SAME objects
// through `Prepare::dspine` (execution/parameters.cpp:267), which is the pair under test.
[[nodiscard]] Weight over_object(const WeightParent* parent, const std::string& name,
                                 const std::uint64_t begin, const std::uint64_t elements) {
    if (parent == nullptr || parent->data == nullptr) {
        throw std::runtime_error(name + ": its packed object is not resident on the device");
    }
    const auto& shape = parent->geometry.shape;
    if (shape.size() < 2) {
        throw std::runtime_error(name + ": its packed object is not a matrix");
    }
    const auto count = elements == 0 ? parent->geometry.elements : elements;
    if (begin + count > parent->geometry.elements) {
        throw std::runtime_error(name + ": the requested range leaves its packed object");
    }
    const std::vector<std::uint64_t> matrix{shape[shape.size() - 2], shape.back()};
    return native_weight({matrix, {{parent, begin, begin + count}}}, 1.0F);
}

[[nodiscard]] Weight packed_parent(const BoundWeight& bound) {
    const auto& view = bound.view;
    if (view.parts.size() != 1) {
        throw std::runtime_error(bound.name + ": a packed parent is one contiguous object range");
    }
    return over_object(view.parts.front().parent, bound.name, 0, 0);
}

[[nodiscard]] Weight stacked(const BoundWeight& bound, const std::uint64_t layer,
                              const std::uint64_t rows, const std::uint64_t columns) {
    const auto& view = bound.view;
    if (view.parts.size() != 1) {
        throw std::runtime_error(bound.name + ": a stacked parameter is one contiguous object "
                                  "range");
    }
    return over_object(view.parts.front().parent, bound.name, layer * rows * columns,
                       rows * columns);
}

[[nodiscard]] Weight as_weight(const BoundWeight& bound) {
    return native_weight(bound.view, 1.0F);
}

[[nodiscard]] Tensor as_parameter(const BoundWeight& bound) {
    const auto& shape = bound.view.shape;
    return weight_tensor(bound.view, {dimension(shape[0]), 1, 1, 1});
}

struct SpecLayer {
    Tensor input_norm, post_attention_norm, query_norm, key_norm;
    Weight packed_query_key_value;   // the packed parent the block AND the context split run on
    Weight output, down;
    Weight packed_gate_up;
};

// The specification's view of the drafter: every weight the walk names, in the form the
// specification's own calls take it in.
struct SpecWeights {
    Weight embedding, feature_projection, codebook, gate, predecessor_gate, prev, head, tg, tsv;
    Tensor final_norm, gate_bias, context_norm;
    std::vector<SpecLayer> layers;
    std::vector<Weight> read;    // [layers]: the stacked early-read slices, last = read_weight
    std::vector<Weight> write;   // [layers]: the stacked feature-write slices
};

// ---------------------------------------------------------------- the specification's walk ----
//
// `run_forward` and `run_bridge` from test_dspine_forward_real.cpp (:1004-1218 and :769-844),
// transcribed step for step on this file's streams, with the probe's `step()` gate folded into
// `step()` above (a gate here voids the parity claim and ends the run). The bridge of layer N is
// the seven steps at :782-843 in the probe's order. Nothing here is the executor's code: the
// executor's arm is driven in part 4 below, and the two arms meet only in the byte comparison.
struct SpecArm {
    const Geometry* geometry;
    SpecWeights* weights;
    Round* round;
    DrafterContext* pages;
    MemoryArena* arena;
    const dspine::Config* config;

    void bridge(const std::size_t layer) {
        const auto& g         = *geometry;
        const auto cuda       = owner->cuda();
        const auto columns    = std::int32_t(g.w) * g.b;
        const auto& probe     = *weights;
        const auto last = layer + 1 == probe.layers.size();
        auto flat_residual   = round->residual.view({g.h, columns, 1, 1});
        auto flat_normalized = round->normalized.view({g.h, columns, 1, 1});
        auto flat_rms        = round->rms.view({columns, 1, 1, 1});
        auto flat_codes      = round->codes.view({g.c, columns, 1, 1});
        step("bridge read: rms of the layer stream", [&] {
            ops::dspine_prepare_read(flat_residual, g.bridge_eps, flat_normalized, flat_rms,
                                      cuda.stream());
        });
        step("bridge code projection", [&] {
            const auto& read = probe.read[layer];
            auto codes_in   = round->codes.view({g.c, columns, 1, 1});
            auto codes_out  = round->codes_out.view({g.c, columns, 1, 1});
            ops::linear(codes_in, read, codes_out, ops::LinearPolicy::A16Only, *arena, cuda);
        });
        step("bridge normalize_codes over the real codebook", [&] {
            auto codes_pre  = round->codes_out.view({g.c, columns, 1, 1});
            auto codes_post = round->codes.view({g.c, columns, 1, 1});
            ops::dspine_normalize_codes(codes_pre, round->anchors, probe.codebook, g.w,
                                        g.bridge_eps, codes_post, cuda.stream());
        });
        step("bridge gate, predecessor and value projections into the packed block", [&] {
            const auto plane = std::int64_t(g.r) * columns * 2;
            const auto base  = static_cast<std::byte*>(round->packed.data);
            auto gate = Tensor(base, owner->device, DType::BF16, {g.r, columns, 1, 1});
            auto pred = Tensor(base + plane, owner->device, DType::BF16, {g.r, columns, 1, 1});
            auto prev = Tensor(base + 2 * plane, owner->device, DType::BF16, {g.r, columns, 1, 1});
            ops::linear(flat_codes, probe.gate, gate, ops::LinearPolicy::A16Only, *arena, cuda);
            ops::linear(flat_codes, probe.predecessor_gate, pred, ops::LinearPolicy::A16Only,
                        *arena, cuda);
            ops::linear(flat_codes, probe.prev, prev, ops::LinearPolicy::A16Only, *arena, cuda);
        });
        step("bridge message", [&] {
            ops::dspine_message(round->packed, probe.gate_bias, round->message, cuda.stream());
        });
        step("bridge write projection", [&] {
            auto message = round->message.view({g.r, columns, 1, 1});
            auto delta   = round->delta.view({g.h, columns, 1, 1});
            ops::linear(message, probe.write[layer], delta, ops::LinearPolicy::A16Only, *arena,
                        cuda);
        });
        step("bridge finish: refines the residual in place, fuses the next layer's norm", [&] {
            const auto next = last ? probe.final_norm
                                   : probe.layers[layer + 1].input_norm;
            auto delta  = round->delta.view({g.h, columns, 1, 1});
            auto normed = round->normed.view({g.h, columns, 1, 1});
            ops::dspine_finish(flat_residual, flat_rms, delta, next, g.rms_eps, normed,
                               &round->stash, cuda.stream());
        });
    }

    void run() {
        const auto& g       = *geometry;
        const auto cuda     = owner->cuda();
        const auto columns  = std::int32_t(g.w) * g.b;
        const auto flat     = std::int32_t(g.ctx * g.b);
        const auto& layers  = weights->layers;
        step("block arrangement: the mask block, the mask token in the draft positions", [&] {
            ops::prepare_masked_block(round->anchors, round->lengths, round->valid, g.mask_token,
                                       round->ids, round->positions, cuda);
        });
        step("token embedding of the block (the tower's own embedding)", [&] {
            auto ids      = round->ids.view({columns, 1, 1, 1});
            auto residual = round->residual.view({g.h, columns, 1, 1});
            ops::embedding(ids, weights->embedding, residual, cuda);
        });
        step("feature projection of the target taps", [&] {
            auto taps_in   = round->taps.view({g.h * g.taps, g.ctx * g.b, 1, 1});
            auto projected = round->projected.view({g.h, columns, 1, 1});
            ops::linear(taps_in, weights->feature_projection, projected,
                        ops::LinearPolicy::A16Only, *arena, cuda);
        });
        step("context norm", [&] {
            auto projected = round->projected.view({g.h, columns, 1, 1});
            auto context   = round->context.view({g.h, columns, 1, 1});
            ops::rmsnorm(projected, weights->context_norm, g.rms_eps, false, context, cuda);
        });
        for (std::size_t layer = 0; layer < layers.size(); ++layer) {
            const auto& block = layers[layer];
            step("context key/value projections, key norm, RoPE, append to the layer's pages",
                 [&] {
                     auto context     = round->context.view({g.h, flat, 1, 1});
                     auto key_out     = round->context_key_raw.view({g.k, flat, 1, 1});
                     auto value_out   = round->context_value.view({g.k, flat, 1, 1});
                     auto query_out   = round->scratch_query.view({g.q, flat, 1, 1});
                     const std::array split{std::move(query_out), std::move(key_out),
                                            std::move(value_out)};
                     ops::linear_split(context, block.packed_query_key_value, split, cuda);
                     ops::rmsnorm(round->context_key_raw, block.key_norm, g.rms_eps, false,
                                  round->context_key, cuda);
                     ops::rope(round->context_positions.view({flat, 1, 1, 1}), g.d, g.rope_theta,
                               round->context_key, cuda);
                     ops::kv_cache_append_prefix(
                         round->context_key, round->context_value, round->context_positions,
                         round->context_counts, round->table_rows,
                         ops::KVCacheAppendPrefixExecutionEnvelope{static_cast<std::uint32_t>(g.ctx),
                                                                   static_cast<std::uint32_t>(g.ctx)},
                         pages->layer(layer), cuda);
                 });
        }
        const ops::AttentionHeadGeometry heads{g.d, g.qh, g.kvh};
        const auto scale = float(1.0 / std::sqrt(double(g.d)));
        const auto context_envelope =
            ops::ContextAttentionExecutionEnvelope{static_cast<std::uint32_t>(g.ctx), static_cast<std::uint32_t>(g.ctx)};
        bool pre_normed = false;
        for (std::size_t layer = 0; layer < layers.size(); ++layer) {
            const auto& block = layers[layer];
            if (pre_normed) {
                step("input norm taken from the previous layer's fused write", [&] {
                    cuda_ok(cudaMemcpyAsync(round->hidden.data, round->normed.data,
                                            std::size_t(std::int64_t(g.h) * columns) * 2,
                                            cudaMemcpyDeviceToDevice, cuda.stream()),
                            "carry the pre-normed stream");
                });
            } else {
                step("input norm", [&] {
                    auto residual = round->residual.view({g.h, columns, 1, 1});
                    auto hidden   = round->hidden.view({g.h, columns, 1, 1});
                    ops::rmsnorm(residual, block.input_norm, g.rms_eps, false, hidden, cuda);
                });
            }
            step("split query/key/value projection over the packed parent", [&] {
                const std::array outputs{round->query_raw.view({g.q, columns, 1, 1}),
                                         round->key_raw.view({g.k, columns, 1, 1}),
                                         round->value.view({g.k, columns, 1, 1})};
                ops::linear_split(round->hidden.view({g.h, columns, 1, 1}),
                                  block.packed_query_key_value, outputs, cuda);
            });
            step("query and key head norms, then RoPE", [&] {
                ops::rmsnorm(round->query_raw, block.query_norm, g.rms_eps, false, round->query,
                             cuda);
                ops::rmsnorm(round->key_raw, block.key_norm, g.rms_eps, false, round->key, cuda);
                ops::rope(round->positions.view({columns, 1, 1, 1}), g.d, g.rope_theta,
                          round->query, round->key, cuda);
            });
            step("context attention over the layer's pages", [&] {
                ops::context_softmax_attention(
                    round->query, round->key, round->value, round->context_lengths, round->valid,
                    round->table_rows, heads, scale,
                    ConstPagedKVBatchLayerView(pages->layer(layer)), context_envelope, *arena,
                    round->attended, cuda);
            });
            step("attention output projection into the residual", [&] {
                auto attended  = round->attended.view({g.q, columns, 1, 1});
                auto residual  = round->residual.view({g.h, columns, 1, 1});
                auto projected = round->projected.view({g.h, columns, 1, 1});
                ops::linear(attended, block.output, projected, ops::LinearPolicy::A16Only, *arena,
                            cuda);
                ops::residual_add(projected, residual, cuda);
            });
            step("post-attention norm", [&] {
                auto residual = round->residual.view({g.h, columns, 1, 1});
                auto hidden   = round->hidden.view({g.h, columns, 1, 1});
                ops::rmsnorm(residual, block.post_attention_norm, g.rms_eps, false, hidden, cuda);
            });
            step("gate/up projection over the packed parent (SwiGLU)", [&] {
                auto hidden       = round->hidden.view({g.h, columns, 1, 1});
                auto intermediate = round->intermediate.view({g.inter, columns, 1, 1});
                ops::linear_swiglu(hidden, block.packed_gate_up, intermediate,
                                   ops::LinearPolicy::A16Only, *arena, cuda);
            });
            step("down projection into the residual", [&] {
                auto intermediate = round->intermediate.view({g.inter, columns, 1, 1});
                auto residual     = round->residual.view({g.h, columns, 1, 1});
                ops::linear_add(intermediate, block.down, residual,
                                ops::LinearPolicy::A16Only, *arena, cuda);
            });
            bridge(layer);
            pre_normed = true;
        }
        step("gather the proposal columns out of the wide block", [&] {
            const auto row      = std::size_t(std::int64_t(g.h) * g.drafts) * 2;
            const auto source   = static_cast<std::byte*>(round->normed.data)
                                + std::size_t(g.h) * 2;
            const auto spitch   = std::size_t(std::int64_t(g.h) * columns) * 2;
            cuda_ok(cudaMemcpy2DAsync(round->proposal.data, row, source, spitch, row,
                                      std::size_t(g.b), cudaMemcpyDeviceToDevice, cuda.stream()),
                    "gather the proposal columns");
        });
        step("proposal logits per slot: the tower head's top-k over the valid rows", [&] {
            auto proposal   = round->proposal.view({g.h, g.drafts, 1, 1});
            auto candidates = round->candidates.view({kSlots, g.drafts, 1, 1});
            auto scores     = round->scores.view({kSlots, g.drafts, 1, 1});
            ops::linear_topk(proposal, weights->head, kTokenDomain, candidates, scores, *arena,
                            cuda);
        });
        step("the tail's candidate layout: the head's grid to the tail's", [&] {
            seed_tail_inputs();
        });
        step("serial scores over the tower head, emitting the tail's FP32 score table", [&] {
            auto cand = round->candidates.view({kSlots, g.drafts, g.b, 1});
            auto idx  = round->idx_scores.view({kSlots, g.drafts, g.b, 1});
            ops::dspine_serial_scores(cand, round->stash, round->rms, round->message,
                                      round->anchors, weights->final_norm, weights->gate_bias,
                                      weights->head, weights->tg, weights->tsv, g.w, idx,
                                      &round->table, cuda.stream());
        });
    }

    // The probe's `seed_tail_inputs` (:863-890): the head's [candidate, slot, batch] grid to the
    // tail's [batch, slot, candidate] across a stream-settled host crossing.
    void seed_tail_inputs() {
        const auto slots = round->candidates.ne[1];
        const CudaDeviceScope settling(owner->device);
        cuda_ok(cudaDeviceSynchronize(), "the head's candidate rows before the tail's permute");
        const auto from = download_i32(round->candidates);
        std::vector<std::int32_t> to(std::size_t(std::int64_t(kBatch) * slots * kSlots));
        for (std::int32_t batch = 0; batch < kBatch; ++batch) {
            for (std::int32_t slot = 0; slot < slots; ++slot) {
                for (std::int32_t cand = 0; cand < kSlots; ++cand) {
                    to[std::size_t(batch + std::int64_t(kBatch)
                                        * (slot + std::int64_t(slots) * cand))] =
                        from[std::size_t(cand
                                         + std::int64_t(kSlots)
                                                   * (slot + std::int64_t(slots) * batch))];
                }
            }
        }
        upload(round->tail_cand.data, to.data(), to.size() * sizeof(std::int32_t),
               "tail candidates");
    }

    // The probe's `run_tail_case` (:1221-1258) at the production policy.
    void tail() {
        const auto cuda = owner->cuda();
        ops::dspine_serial_tail(round->table, round->tail_cand,
                                 static_cast<const ops::SamplingConfig*>(round->configs.data),
                                 round->draw_positions, round->tail_drafts, round->tail_q,
                                 round->tail_index, cuda.stream());
        cuda_ok(cudaGetLastError(), "serial tail launch");
        cuda_ok(cudaDeviceSynchronize(), "serial tail sync");
    }
};

} // namespace
namespace {

// ---------------------------------------------------------------- the binding table ----------
//
// One row per weight the round names by: the specification's form (the probe's `over_object`
// slice of the shard's object view) against the executor's (`Prepare::dspine`). The identity is
// the device address of the matrix's code plane plus its rows, its columns and its representation
// - a stacked slice that took the wrong row range, a packed parent that is not the object the
// context split needs, or a representation that resolved apart are all visible here, and the two
// walks then ran on these two pairs.
[[nodiscard]] std::string weight_identity(const Weight& weight) {
    char buffer[160];
    std::snprintf(buffer, sizeof(buffer), "%p n=%d k=%d qtype=%d layout=%d",
                  const_cast<void*>(std::bit_cast<void*>(weight.qdata)), weight.n, weight.k,
                  static_cast<std::int32_t>(weight.qtype),
                  static_cast<std::int32_t>(weight.layout));
    return buffer;
}

struct BindingRow {
    std::string label;
    Weight specification, executor;
};

struct BindingReport {
    std::int32_t rows = 0, parted = 0;
};

[[nodiscard]] BindingReport report_bindings(const std::vector<BindingRow>& rows) {
    BindingReport out;
    for (const auto& row : rows) {
        ++out.rows;
        const auto left = weight_identity(row.specification);
        const auto right = weight_identity(row.executor);
        const bool same  = left == right;
        out.parted += not same;
        std::printf("BINDING %-34s %s\n", row.label.c_str(), same ? "matched" : "PARTED");
        if (not same) {
            std::printf("        specification %s\n        executor      %s\n", left.c_str(),
                        right.c_str());
        } else {
            std::printf("        %s\n", right.c_str());
        }
    }
    std::printf("BINDING %d of %d weight pairs are the same device range, %d parted\n",
                out.rows - out.parted, out.rows, out.parted);
    std::fflush(stdout);
    return out;
}

[[nodiscard]] BindingReport report_parameters(const std::vector<std::pair<std::string, void*>>&
                                                  specification,
                                              const std::vector<std::pair<std::string, void*>>&
                                                  executor) {
    BindingReport out;
    for (std::size_t index = 0; index < specification.size(); ++index) {
        ++out.rows;
        const auto left  = specification[index].second;
        const auto right = executor[index].second;
        const bool same  = left == right;
        out.parted += not same;
        std::printf("BINDING %-34s %s  %p / %p\n", specification[index].first.c_str(),
                    same ? "matched" : "PARTED", left, right);
    }
    std::printf("BINDING %d of %d parameter tensors are the same device address, %d parted\n",
                out.rows - out.parted, out.rows, out.parted);
    std::fflush(stdout);
    return out;
}

// ---------------------------------------------------------------- the fidelity rows ----------
//
// The unit is the bf16 ulp: two finite bfloat16 values are adjacent when their ordered codes
// differ by one, so the ordered-code distance is the ulp count (the same mapping bfloat16 has
// under the truncation of a float32, with -0 and +0 sharing a code). The fp32 ulp is the same
// mapping over the 32-bit codes. An exact bit match is a zero distance, and bit-exactness is
// reported separately, because the distance does not see a sign-of-zero part.
struct Fidelity {
    std::string label;
    std::size_t elements = 0, differing = 0;
    std::int64_t maximum = 0, first = -1;
    std::string detail;
    std::int32_t arms = 0;
};

[[nodiscard]] std::int64_t bf16_code(const std::uint16_t bits) {
    return (bits & 0x8000u) ? std::int64_t(0x8000) - std::int64_t(bits & 0x7fffu)
                             : std::int64_t(0x8000) + std::int64_t(bits);
}

[[nodiscard]] std::int64_t f32_code(const float value) {
    const auto bits = std::bit_cast<std::int32_t>(value);
    return bits >= 0 ? std::int64_t(bits) : std::int64_t(0x7fffffff) - std::int64_t(bits);
}

template <class Value>
[[nodiscard]] Fidelity compare(const std::string& label, const std::vector<Value>& specification,
                               const std::vector<Value>& executor,
                               const std::int32_t ulp) {
    Fidelity out;
    out.label = label;
    out.elements = specification.size();
    out.arms     = ulp;
    for (std::size_t index = 0; index < specification.size(); ++index) {
        if (specification[index] == executor[index]) { continue; }
        ++out.differing;
        const auto distance = [&] {
            if (ulp == 0) { return std::int64_t(1); }
            if (ulp == 16) {
                return std::abs(bf16_code(specification[index]) - bf16_code(executor[index]));
            }
            return std::abs(f32_code(specification[index]) - f32_code(executor[index]));
        }();
        if (distance > out.maximum) { out.maximum = distance; }
        if (out.first < 0) {
            out.first = std::int64_t(index);
            char buffer[128];
            if (ulp == 32) {
                std::snprintf(buffer, sizeof(buffer), "at element %zu: specification %.9g, "
                                    "executor %.9g",
                              index, double(specification[index]), double(executor[index]));
            } else if (ulp == 16) {
                std::snprintf(buffer, sizeof(buffer),
                              "at element %zu: specification 0x%04x, executor 0x%04x",
                              index, static_cast<std::uint32_t>(specification[index]),
                              static_cast<std::uint32_t>(executor[index]));
            } else {
                std::snprintf(buffer, sizeof(buffer), "at element %zu: specification %d, "
                                    "executor %d",
                              index, static_cast<std::int32_t>(specification[index]),
                              static_cast<std::int32_t>(executor[index]));
            }
            out.detail = buffer;
        }
    }
    return out;
}

void report(const Fidelity& row) {
    std::printf("FIDELITY %-30s bit-exact %s  elements %zu  differing %zu  max %s  first %s\n",
                row.label.c_str(), row.differing == 0 ? "Y" : "N", row.elements, row.differing,
                row.maximum == 0 ? "0 ulp" : [&] {
                    char buffer[48];
                    std::snprintf(buffer, sizeof(buffer), "%lld %s",
                                  static_cast<long long>(row.maximum),
                                  row.arms == 32 ? "fp32-ulp" : "bf16-ulp");
                    return std::string(buffer);
                }()
                    .c_str(),
                row.detail.empty() ? "none" : row.detail.c_str());
    std::fflush(stdout);
}

[[nodiscard]] std::uint64_t digest_of(const Tensor& stream) {
    std::vector<std::byte> bytes(std::size_t(stream.bytes()));
    download(bytes.data(), stream.data, bytes.size(), "fingerprint a round stream");
    return digest(bytes.data(), bytes.size());
}

struct Finger {
    std::vector<std::pair<std::string, std::uint64_t>> parts;

    [[nodiscard]] std::size_t parted_against(const Finger& other) const {
        std::size_t parted = 0;
        for (std::size_t index = 0; index < parts.size(); ++index) {
            parted += parts[std::size_t(index)].second
                   != other.parts[std::size_t(index)].second;
        }
        return parted;
    }
};

} // namespace

// ---------------------------------------------------------------- the run ---------------------
int main(const int argc, char** argv) {
    (void)argc;
    (void)argv;
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::signal(SIGSEGV, report_fault);
    std::signal(SIGABRT, report_fault);
    std::signal(SIGFPE, report_fault);
    const auto artifact_path = setting("NINFER_TEST_ARTIFACT");
    if (artifact_path.empty()) {
        std::printf("skip: NINFER_TEST_ARTIFACT is not set\n");
        return 77;
    }
    const auto device_setting = setting("NINFER_TEST_DEVICE");
    if (device_setting.empty()) {
        std::printf("skip: NINFER_TEST_DEVICE is not set\n");
        return 77;
    }
    const int device = std::stoi(device_setting);
    int visible      = 0;
    if (cudaGetDeviceCount(&visible) != cudaSuccess || visible <= 0) {
        std::printf("skip: no CUDA device\n");
        return 77;
    }
    if (device >= visible) {
        std::printf("skip: NINFER_TEST_DEVICE=%d, %d visible\n", device, visible);
        return 77;
    }
    const auto asked  = std::atoi(setting("NINFER_TEST_ROUNDS", "32").c_str());
    const auto rounds = std::max(32, asked > 0 ? asked : 32);

    try {
        Probe probe(artifact_path, device);
        work_device = probe.device;
        owner       = &probe;

        // ---- the executor's own load: the definition, the profile, the shard, the Parameters ---
        models::LoadOptions options;
        options.purpose       = EnginePurpose::Generation;
        options.speculative   = SpeculativeBackend::Dspine;
        options.proposal_head = ProposalHead::Full;
        const auto definition = q35::read_definition(probe.reader, options);
        const auto profile    = execution::describe_model(*definition);
        const auto& weights   = definition->weights();
        if (!weights.dspine || !profile->config.dspine) {
            std::printf("STOP: the artifact carries no dspine drafter\n");
            return 1;
        }
        const auto& draft     = *profile->config.dspine;
        const auto& tower     = profile->config.text;
        auto plan = q35::resolve_load(probe.reader, definition, probe.device);
        const auto [free, total] = probe.memory();
        const auto need = plan.materialization().device_capacity_bytes;
        const auto slack = std::uint64_t(std::stoull(setting("NINFER_FIDELITY_SLACK", "768")))
                         * 1024u * 1024u;
        std::printf("load: the executor's shard needs %.5f GiB of the %.5f GiB free on the card "
                    "(%.5f GiB total); the probe insists on %.5f GiB of headroom\n",
                    need / kGiB, free / kGiB, total / kGiB, slack / kGiB);
        std::fflush(stdout);
        if (need + slack > free) {
            std::printf("REFUSAL: the load is %s  It is refused on the card's own numbers before "
                        "a byte is allocated: a device bind that cannot be satisfied is the stop "
                        "D-47 recorded, and this probe does not chase it.\n",
                        "wider than the card's free memory plus the headroom");
            return 3;
        }
        const auto began = std::chrono::steady_clock::now();
        auto model       = q35::materialize_model(std::move(plan), probe.context);
        const auto elapsed = std::chrono::duration_cast<
            std::chrono::duration<double, std::ratio<1, 1000>>>(std::chrono::steady_clock::now()
                                                                - began);
        const auto [after, unused] = probe.memory();
        (void)unused;
        const auto& stats = model->storage_stats();
        std::printf("load: %zu device objects, %.5f GiB of backing, h2d %.5f GiB in %.2f s "
                    "(%.2f s through materialize_model); device free %llu -> %llu MiB\n",
                    stats.device_object_count, stats.device_capacity_bytes / kGiB,
                    stats.h2d_bytes / kGiB, stats.upload_seconds, elapsed.count() / 1000.0,
                    static_cast<unsigned long long>(free / 1024 / 1024),
                    static_cast<unsigned long long>(after / 1024 / 1024));
        std::fflush(stdout);
        // The binding census. The executor builds its `DspineParameters` one projection at a
        // time through `Prepare::linear(WeightId)` and `Prepare::linear(WeightUseId)`
        // (src/models/qwen3_5/execution/parameters.cpp:28-36), and the `WeightId` form of
        // `weight_input` refuses a parameter whose declared uses are not exactly one
        // (src/models/common/weights.cpp:15-18) — a shared parameter is only usable through a
        // `WeightUseId`, which names the mathematical use as well as the object, because the use
        // is what carries the operation's activation policy and calibration. The census is that
        // rule, parameter by parameter, against the definition's own declared-use table
        // (`ModelDefinition::uses`) and the shard's, printed before the construction: a binding
        // that stops is then a row of a table, not only an exception, and a binding that starts
        // silently dropping a declared use is a RED here rather than a policy widening later.
        std::int32_t census_reds = 0;
        {
            // `gathered` marks a parameter the round reads as a TABLE - a row gather inside a
            // DSpine kernel, an embedding lookup, a norm vector - which the artifact therefore
            // declares no mathematical use for. Such a parameter is not in the rule's domain: it
            // is reached as the bytes of its object, not as a `WeightInput`. Every other row is a
            // matmul operand, and the executor reaches it through `Prepare::linear`, so the rule
            // applies to it and a row that fails the rule is a RED of this probe.
            struct Census {
                std::string label;
                WeightId id;
                bool gathered = false;
                bool shared_head = false;
            };
            std::vector<Census> projections;
            std::int32_t linear_bound = 0, table_rows = 0;
            try {
                // The definition's own id table for the drafter: the same
                // `dspine::bind_weights` result `execution::Parameters` binds through, with
                // `token_embedding`/`output_head` already aliased to the tower's
                // (src/models/qwen3_5/definition.cpp:93-100).
                if (not definition->weights().dspine) {
                    throw std::runtime_error("the definition carries no drafter id table");
                }
                const auto& ids = *definition->weights().dspine;
                projections.emplace_back("feature_projection", ids.feature_projection);
                for (std::size_t layer = 0; layer < ids.layers.size(); ++layer) {
                    const auto prefix = "layer/" + std::to_string(layer);
                    const auto& attention = ids.layers[layer].attention;
                    const auto& mlp       = ids.layers[layer].mlp;
                    projections.emplace_back(prefix + "/attention/query", attention.query);
                    projections.emplace_back(prefix + "/attention/key", attention.key);
                    projections.emplace_back(prefix + "/attention/value", attention.value);
                    projections.emplace_back(prefix + "/attention/context_key",
                                             attention.context_key);
                    projections.emplace_back(prefix + "/attention/context_value",
                                             attention.context_value);
                    projections.emplace_back(prefix + "/attention/output", attention.output);
                    projections.emplace_back(prefix + "/mlp/gate", mlp.gate);
                    projections.emplace_back(prefix + "/mlp/up", mlp.up);
                    projections.emplace_back(prefix + "/mlp/down", mlp.down);
                }
                projections.emplace_back("bridge/read_weight", ids.bridge.read_weight);
                projections.emplace_back("bridge/early_read_weights",
                                         ids.bridge.early_read_weights);
                projections.emplace_back("bridge/feature_write_weights",
                                         ids.bridge.feature_write_weights);
                projections.emplace_back("bridge/gate", ids.bridge.gate);
                projections.emplace_back("bridge/predecessor_gate", ids.bridge.predecessor_gate);
                projections.emplace_back("bridge/prev", ids.bridge.prev);
                projections.emplace_back("bridge/tg", ids.bridge.tg, true);
                projections.emplace_back("bridge/tsv", ids.bridge.tsv, true);
                projections.emplace_back("bridge/codebook", ids.bridge.codebook, true);
                projections.emplace_back("token_embedding", ids.token_embedding, true);
                projections.emplace_back("output_head", ids.output_head, false, true);
            } catch (const std::exception& failure) {
                std::printf("census: the drafter's id table is not bindable: %s\n",
                            failure.what());
            }
            std::printf("census: every drafter projection and table `execution::Parameters` "
                        "binds. `declared` is the use\n            list the definition carries "
                        "for the parameter and `carried` what the loaded parameter\n            "
                        "has. The rule the binding runs on is the one at the right of the "
                        "header\n");
            std::printf("%-44s %12s %10s %6s %9s %8s  %s\n",
                        "parameter", "rows", "columns", "parts", "declared", "carried",
                        "the use the executor binds it by (weights.cpp:15 needs exactly one)");
            for (const auto& row : projections) {
                const auto& id   = row.id;
                const auto& view = model->weight(id).view;
                const auto  rank = view.shape.size();
                const auto  rows = rank == 0 ? std::size_t(0) : std::size_t(view.shape[rank - 2]);
                const auto  cols = rank == 0 ? std::size_t(0) : std::size_t(view.shape[rank - 1]);
                const auto  carried = model->weight(id).uses.size();
                std::string verdict;
                if (row.gathered) {
                    verdict = carried == 0
                                  ? "table: no use is declared for it, and none is needed"
                                  : "table that unexpectedly carries a use";
                    ++table_rows;
                    census_reds += carried != 0;
                } else if (row.shared_head) {
                    // The one parameter of this drafter that is genuinely SHARED by several
                    // arms: four declared uses, so the id form is refused by design and the
                    // round must name the one it multiplies under.
                    const auto& uses = definition->uses(id);
                    const auto  found = std::find_if(
                        uses.begin(), uses.end(), [](const auto& use) {
                            return use.input == "dspine/final_hidden";
                        });
                    verdict =
                        found == uses.end()
                            ? "REFUSED: the manifest declares no dspine use of the head"
                            : "by WeightUseId @" + found->input + " (policy "
                                  + (found->policy == ops::LinearPolicy::AllowA8
                                         ? "AllowA8"
                                         : found->policy == ops::LinearPolicy::AllowA4
                                               ? "AllowA4"
                                               : "A16Only")
                                  + "), " + std::to_string(carried) + " uses declared";
                    census_reds +=
                        found == uses.end() || carried != uses.size() || carried < 2;
                    ++linear_bound;
                } else if (carried == 1) {
                    verdict = "by its one declared use @" + model->weight(id).uses[0].input;
                    ++linear_bound;
                } else {
                    verdict = "REFUSED: the parameter carries no single declared use, so "
                              "linear(WeightId) cannot bind it";
                    ++census_reds;
                }
                std::printf("%-44s %12zu %10zu %6zu %9zu %8zu  %s\n",
                            model->weight(id).name.c_str(),
                            rows,
                            cols,
                            view.parts.size(),
                            definition->uses(id).size(),
                            carried,
                            verdict.c_str());
            }
            std::printf("census: %zu drafter parameters, %d of them linear operands that bind, "
                        "%d read as tables, %d RED\n",
                        projections.size(),
                        linear_bound,
                        table_rows,
                        census_reds);
            std::fflush(stdout);
        }
        // The construction is the executor's own step, and on this artifact it is where the
        // DSpine round stops. The pair, printed before the call so a stop in it names both
        // sides of itself:
        //
        //   the producer  src/models/common/dspine/weights.cpp:176,179,181,182 binds the four
        //                 bridge projections the bridge multiplies through `Bindings::direct`,
        //                 the input-less form (`binds.cpp:114` calls
        //                 `parameter(name, shape, {}, format)`), so the artifact's own declared
        //                 use for each of them - the manifest lists
        //                 `dspine/bridge/gate` on input `dspine/bridge/packed_input`, and the
        //                 same for `predecessor_gate`, `prev`, `read_weight` on
        //                 `dspine/bridge/read_input`, `early_read_weights` on
        //                 `dspine/bridge/early_read_input`, `feature_write_weights` on
        //                 `dspine/bridge/message_input` - never reaches the parameter. The
        //                 drafter's layer projections, bound a few lines earlier through
        //                 `b.parameter(name, shape, {p + "mlp_input"})` and friends
        //                 (`weights.cpp:158,166`), do carry theirs: the `declared`/`carried`
        //                 columns above are 1/1 for every one of them, 0/0 for the bridge's six.
        //   the consumer  src/models/qwen3_5/execution/parameters.cpp:280-282 and :299 bind the
        //                 same parameters with `linear(WeightId)`, whose rule is
        //                 `src/models/common/weights.cpp:17-18`: a parameter that does not carry
        //                 exactly one declared use is refused, because a shared projection is
        //                 usable only through a named mathematical use.
        std::printf("bind: building execution::Parameters (the executor's DspineParameters)\n");
        std::fflush(stdout);
        const execution::Parameters parameters(*model, profile);
        if (!parameters.dspine || parameters.dspine->layers.empty()) {
            std::printf("STOP: the executor bound no dspine layers\n");
            return 1;
        }

        // ---- the geometry, in the units each arm names its streams by --------------------------
        Geometry g;
        g.h      = dim(draft.hidden_size);
        g.w      = dim(draft.block_size);
        g.b      = kBatch;
        g.ctx    = kContext;
        g.c      = dim(draft.bridge.code_size);
        g.r      = dim(draft.bridge.rank);
        g.taps   = dim(draft.target_layer_ids.size());
        g.q      = dim(draft.query_width());
        g.k      = dim(draft.key_width());
        g.d      = dim(draft.head_dim);
        g.inter  = dim(draft.intermediate_size);
        g.drafts = dim(draft.block_size) - 1;
        g.rms_eps = draft.rms_norm_eps;
        g.rope_theta = draft.rope_theta;
        g.bridge_eps = draft.bridge.eps;
        g.mask_token = dim(draft.mask_token_id);
        g.qh     = dim(draft.num_attention_heads);
        g.kvh    = dim(draft.num_key_value_heads);
        std::printf("drafter: hidden=%d layers=%d block=%d taps=%d code=%d rank=%d q=%d k=%d "
                    "heads=%d/%d head_dim=%d inter=%d mask=%d sliding_window=%u; tower: "
                    "hidden=%d vocab=%d layers=%d\n",
                    g.h, dim(draft.num_hidden_layers), g.w, g.taps, g.c, g.r, g.q, g.k, g.qh,
                    g.kvh, g.d, g.inter, g.mask_token, draft.sliding_window,
                    dim(tower.hidden_size), dim(tower.vocab_size),
                    dim(tower.num_hidden_layers));
        std::printf("round: batch=%d width=%d drafts=%d context=%d\n", g.b, g.w, g.drafts, g.ctx);
        std::fflush(stdout);

        // ---- the specification's binding of the executor's shard ------------------------------
        const auto& shard = model->definition().weights();
        const auto& bound = *shard.dspine;
        SpecWeights spec;
        spec.embedding          = as_weight(model->weight(shard.text.token_embedding));
        spec.head               = as_weight(model->weight(shard.text.output_head));
        spec.feature_projection = as_weight(model->weight(bound.feature_projection));
        spec.codebook           = as_weight(model->weight(bound.bridge.codebook));
        spec.gate               = as_weight(model->weight(bound.bridge.gate));
        spec.predecessor_gate   = as_weight(model->weight(bound.bridge.predecessor_gate));
        spec.prev               = as_weight(model->weight(bound.bridge.prev));
        spec.tg                 = as_weight(model->weight(bound.bridge.tg));
        spec.tsv                = as_weight(model->weight(bound.bridge.tsv));
        spec.final_norm         = as_parameter(model->weight(bound.final_norm));
        spec.gate_bias          = as_parameter(model->weight(bound.bridge.gate_bias));
        spec.context_norm       = as_parameter(model->weight(bound.context_norm));
        const auto read_shape   = model->weight(bound.bridge.read_weight).view.shape;
        const auto write_shape  = model->weight(bound.bridge.feature_write_weights).view.shape;
        // Layer i of the bridge reads row i of the stacked `early_read_weights` and writes row i
        // of the stacked `feature_write_weights`; the LAST layer reads the artifact's own
        // `read_weight`. The row range is the slice under test: the executor takes
        // (rows, columns) = (read[0], read[1]) and (write[2], write[1])
        // (execution/parameters.cpp:290-300), which is the probe's `stacked` at the same product.
        for (std::size_t layer = 0; layer + 1 < bound.layers.size(); ++layer) {
            spec.read.push_back(stacked(model->weight(bound.bridge.early_read_weights), layer,
                                        read_shape[0], read_shape[1]));
            spec.write.push_back(stacked(model->weight(bound.bridge.feature_write_weights),
                                         layer, write_shape[2], write_shape[1]));
        }
        spec.read.push_back(as_weight(model->weight(bound.bridge.read_weight)));
        for (std::size_t layer = bound.layers.size() - 1; layer < bound.layers.size(); ++layer) {
            spec.write.push_back(stacked(model->weight(bound.bridge.feature_write_weights),
                                         layer, write_shape[2], write_shape[1]));
        }
        for (const auto& layer : bound.layers) {
            SpecLayer row;
            row.input_norm          = as_parameter(model->weight(layer.input_norm));
            row.post_attention_norm = as_parameter(model->weight(layer.post_attention_norm));
            row.query_norm          = as_parameter(model->weight(layer.attention.query_norm));
            row.key_norm            = as_parameter(model->weight(layer.attention.key_norm));
            row.packed_query_key_value = packed_parent(model->weight(layer.attention.query));
            row.output              = as_weight(model->weight(layer.attention.output));
            row.packed_gate_up      = packed_parent(model->weight(layer.mlp.gate));
            row.down                = as_weight(model->weight(layer.mlp.down));
            spec.layers.push_back(row);
        }

        // ---- the binding table: the two bindings of one set of bytes ---------------------------
        std::printf("binding: the specification's forms (the probe's over_object slices of the "
                    "shard's object views) against the executor's (Prepare::dspine)\n");
        std::vector<BindingRow> rows;
        const auto pair = [&](const std::string& label, Weight left, Weight right) {
            rows.push_back({label, std::move(left), std::move(right)});
        };
        pair("tower token embedding", spec.embedding, parameters.dspine->token_embedding);
        // The head: the specification binds the tower's object directly, the executor binds the
        // head's `dspine/final_hidden` USE of the same object. Same bytes, two declared uses.
        pair("tower output head", spec.head, parameters.dspine->output_head.weight);
        pair("dspine feature_projection", spec.feature_projection,
             parameters.dspine->feature_projection.weight);
        pair("dspine bridge codebook", spec.codebook, parameters.dspine->bridge.codebook);
        pair("dspine bridge gate", spec.gate, parameters.dspine->bridge.gate.weight);
        pair("dspine bridge predecessor_gate", spec.predecessor_gate,
             parameters.dspine->bridge.predecessor_gate.weight);
        pair("dspine bridge prev", spec.prev, parameters.dspine->bridge.prev.weight);
        pair("dspine bridge tg", spec.tg, parameters.dspine->tg);
        pair("dspine bridge tsv", spec.tsv, parameters.dspine->tsv);
        for (std::size_t layer = 0; layer < spec.layers.size(); ++layer) {
            const auto tag = "layer " + std::to_string(layer);
            pair(tag + " packed query_key_value",
                 spec.layers[layer].packed_query_key_value,
                 parameters.dspine->layers[layer].query_key_value.weight);
            pair(tag + " attention output", spec.layers[layer].output,
                 parameters.dspine->layers[layer].output.weight);
            pair(tag + " bridge read (stacked)", spec.read[layer],
                 parameters.dspine->bridge.read[layer]);
            pair(tag + " bridge write (stacked)", spec.write[layer],
                 parameters.dspine->bridge.write[layer]);
        }
        const auto bindings = report_bindings(rows);
        const auto parameters_bound = report_parameters(
            {{"final_norm", spec.final_norm.data}, {"gate_bias", spec.gate_bias.data},
             {"context_norm", spec.context_norm.data},
             {"layer 0 input_norm", spec.layers[0].input_norm.data},
             {"layer 0 query_norm", spec.layers[0].query_norm.data},
             {"layer 0 key_norm", spec.layers[0].key_norm.data},
             {"layer 0 post_attention_norm", spec.layers[0].post_attention_norm.data}},
            {{"final_norm", parameters.dspine->final_norm.data},
             {"gate_bias", parameters.dspine->gate_bias.data},
             {"context_norm", parameters.dspine->context_norm.data},
             {"layer 0 input_norm", parameters.dspine->layers[0].input_norm.data},
             {"layer 0 query_norm", parameters.dspine->layers[0].query_norm.data},
             {"layer 0 key_norm", parameters.dspine->layers[0].key_norm.data},
             {"layer 0 post_attention_norm", parameters.dspine->layers[0].post_attention_norm
                                                    .data}});

        // ---- the two arms' streams, page caches and arenas -----------------------------------
        auto spec_round = allocate(g.h, g.w, g.b, g.ctx, g.c, g.r, g.taps, g.q, g.k, g.d,
                                    g.inter, g.drafts);
        auto exec_round = allocate(g.h, g.w, g.b, g.ctx, g.c, g.r, g.taps, g.q, g.k, g.d,
                                    g.inter, g.drafts);
        const std::uint32_t layers = std::uint32_t(spec.layers.size());
        DrafterContext spec_pages(layers, g.d, dim(draft.num_key_value_heads), g.ctx);
        DrafterContext exec_pages(layers, g.d, dim(draft.num_key_value_heads), g.ctx);
        const auto envelope = ops::ContextAttentionExecutionEnvelope{static_cast<std::uint32_t>(g.ctx),
                                                                    static_cast<std::uint32_t>(g.ctx)};
        // The walk's share of the round, by the function the plan sizes it with
        // (program/planning/startup.cpp:435), and the tail's share, by the terms startup.cpp:386
        // -408 sums. One arena carries both, as the engine's round arena does: the walk rewinds it
        // on entry and the tail runs after it.
        const auto walk_bytes = execution::dspine_round_workspace_bytes(
            draft, g.b, g.w, envelope, 1);
        const auto topk_bytes = ops::linear_topk_workspace_capacity_bytes(
            spec.head.qtype, spec.head.n, spec.head.k, g.drafts * g.b, g.drafts * g.b);
        const auto tail_bytes = topk_bytes
            + std::size_t(g.h) * g.drafts * g.b * 2 + std::size_t(kSlots) * g.drafts * g.b * 4
            + std::size_t(kSlots) * g.drafts * g.b * 4
            + std::size_t(g.b) * kSlots * kSlots * g.drafts * 4
            + std::size_t(g.b) * g.drafts * kSlots * (4 + 4 + 4)
            + std::size_t(g.b) * g.drafts * 4;
        const auto arena_bytes = std::max(walk_bytes, tail_bytes) + (32u << 20);
        DeviceBuffer spec_scratch(probe.device, arena_bytes), exec_scratch(probe.device,
                                                                          arena_bytes);
        MemoryArena spec_arena(spec_scratch.span()), exec_arena(exec_scratch.span());
        std::printf("arena: the walk's own sizing is %zu B, the tail's share %zu B, the round "
                    "arena is %zu B (the head's top-k term alone is %zu B)\n",
                    walk_bytes, tail_bytes, arena_bytes, topk_bytes);
        std::printf("head: %d x %d, the tail's top-%d over %d valid rows; the drafter's "
                    "sliding_window reads %u\n",
                    spec.head.n, spec.head.k, kSlots, kTokenDomain, draft.sliding_window);
        std::fflush(stdout);

        // ---- the rounds ---------------------------------------------------------------------
        std::vector<Fidelity> totals;
        Finger spec_previous, exec_previous;
        std::int32_t spec_parted_rounds = 0, exec_parted_rounds = 0, compared = 0;
        std::int32_t peak_walk = 0, peak_tail = 0;
        const auto sampling = static_cast<const ops::SamplingConfig*>(exec_round.configs.data);
        const auto cuda     = probe.cuda();
        SpecArm specification{&g, &spec, &spec_round, &spec_pages, &spec_arena, &draft};
        std::printf("rounds: %d of identical inputs behind a round reset, after %d warmup pass; "
                    "the specification arm is the forward probe's walk, the executor arm is "
                    "dspine_append_context + dspine_walk + dspine_propose_tail\n",
                    rounds, 1);
        std::fflush(stdout);
        for (std::int32_t round = 0; round <= rounds; ++round) {
            reset_round(g, spec_round, spec_pages, spec_arena);
            reset_round(g, exec_round, exec_pages, exec_arena);
            specification.run();
            specification.tail();

            const execution::DspineExecutionView view{parameters, exec_arena, cuda,
                                                      exec_pages.cache_.get()};
            step("executor: dspine_append_context", [&] {
                execution::dspine_append_context(
                    view, exec_round.taps.view({g.h * g.taps, g.ctx, g.b, 1}),
                    exec_round.context_positions, exec_round.context_counts,
                    exec_round.table_rows,
                    ops::KVCacheAppendPrefixExecutionEnvelope{static_cast<std::uint32_t>(g.ctx),
                                                              static_cast<std::uint32_t>(g.ctx)});
                cuda_ok(cudaGetLastError(), "dspine_append_context launch");
            });
            const execution::DspineWalkInputs walk_inputs{exec_round.anchors,
                                                          exec_round.lengths,
                                                          exec_round.valid,
                                                          exec_round.table_rows};
            execution::DspineWalkOutputs walk_outputs{
                exec_round.ids, exec_round.positions, exec_round.stash, exec_round.rms,
                exec_round.message, exec_round.normed};
            step("executor: dspine_walk", [&] {
                execution::dspine_walk(view, walk_inputs, walk_outputs, envelope);
                cuda_ok(cudaGetLastError(), "dspine_walk launch");
            });
            peak_walk = std::max(peak_walk, std::int32_t(exec_arena.peak_used() >> 20));
            exec_arena.reset_peak();
            const execution::DspineTailView tail_view{
                parameters.dspine->output_head.weight, parameters.dspine->final_norm,
                parameters.dspine->gate_bias, parameters.dspine->tg, parameters.dspine->tsv, g.r};
            const execution::DspineTailInputs tail_inputs{
                exec_round.stash, exec_round.rms, exec_round.message, exec_round.normed,
                exec_round.anchors, exec_round.draw_positions, sampling};
            execution::DspineTailOutputs tail_outputs{
                exec_round.drafts, exec_round.candidate_ids, exec_round.proposal_q};
            step("executor: dspine_propose_tail", [&] {
                execution::dspine_propose_tail(tail_view, tail_inputs, tail_outputs, exec_arena,
                                               cuda);
                cuda_ok(cudaGetLastError(), "dspine_propose_tail launch");
            });
            peak_tail = std::max(peak_tail, std::int32_t(exec_arena.peak_used() >> 20));
            // The first pass of a process is the module-load pass: it is not a measurement (D-39's
            // red 1, the forward probe's warmup walk).
            if (round == 0) {
                std::printf("  warmup pass complete: the executor's walk ran\n");
                std::fflush(stdout);
                continue;
            }
            ++compared;
            const auto take = [&](const char* label, const Tensor& left, const Tensor& right,
                                  const std::int32_t ulp) {
                Fidelity row;
                if (ulp == 32) {
                    row = compare(label, download_f32(left), download_f32(right), ulp);
                } else if (ulp == 16) {
                    row = compare(label, download_bf16(left), download_bf16(right), ulp);
                } else {
                    row = compare(label, download_i32(left), download_i32(right), 0);
                }
                totals.push_back(row);
                return row;
            };
            // The tail's proposal distribution: the tail publishes [batch, slot, candidate],
            // the accept op's contract is [candidate, slot, batch]; the bridge is the documented
            // relabel of the same words (execution/dspine.h:57-77, D-40's RED 2).
            const auto take_q = [&] {
                const auto tail_q = download_f32(spec_round.tail_q);
                std::vector<float> head_q(
                    std::size_t(std::int64_t(kSlots) * g.drafts * g.b));
                for (std::int32_t lane = 0; lane < g.b; ++lane)
                    for (std::int32_t slot = 0; slot < g.drafts; ++slot)
                        for (std::int32_t candidate = 0; candidate < kSlots; ++candidate)
                            head_q[std::size_t(candidate)
                                   + std::int64_t(kSlots) * (slot + std::int64_t(g.drafts) * lane)] =
                                tail_q[std::size_t(lane) + std::int64_t(g.b)
                                                     * (slot + std::int64_t(g.drafts) * candidate)];
                auto row = compare("proposal_q (accept layout)", head_q,
                                   download_f32(exec_round.proposal_q), 32);
                totals.push_back(row);
                return row;
            };
            if (round == 1 || round == 2 || round == rounds) {
                std::printf("  round %d\n", round);
                report(take("proposal_hidden (normed block)", spec_round.normed,
                            exec_round.normed, 16));
                report(take("bridge_raw (last write stash)", spec_round.stash, exec_round.stash,
                            16));
                report(take("bridge_message (packed Q)", spec_round.message, exec_round.message,
                            16));
                report(take("bridge_rms (read scale)", spec_round.rms, exec_round.rms, 32));
                report(take("block ids", spec_round.ids, exec_round.ids, 0));
                report(take("block positions", spec_round.positions, exec_round.positions, 0));
                report(take("head candidate rows", spec_round.candidates,
                            exec_round.candidate_ids, 0));
                report(take("tail drafts", spec_round.tail_drafts, exec_round.drafts, 0));
                report(take_q());
            } else {
                take("proposal_hidden (normed block)", spec_round.normed, exec_round.normed, 16);
                take("bridge_raw (last write stash)", spec_round.stash, exec_round.stash, 16);
                take("bridge_message (packed Q)", spec_round.message, exec_round.message, 16);
                take("bridge_rms (read scale)", spec_round.rms, exec_round.rms, 32);
                take("block ids", spec_round.ids, exec_round.ids, 0);
                take("block positions", spec_round.positions, exec_round.positions, 0);
                take("head candidate rows", spec_round.candidates, exec_round.candidate_ids, 0);
                take("tail drafts", spec_round.tail_drafts, exec_round.drafts, 0);
                take_q();
            }
            // Round-to-round determinism: this round's published bytes against the last one's, on
            // identical inputs, per arm.
            const auto fingerprint = [&](Round& round_streams) {
                Finger out;
                for (const auto& [name, stream] : std::vector<std::pair<std::string, Tensor*>>{
                         {"proposal_hidden", &round_streams.normed},
                         {"bridge_raw", &round_streams.stash},
                         {"bridge_message", &round_streams.message},
                         {"bridge_rms", &round_streams.rms},
                         {"ids", &round_streams.ids}, {"positions", &round_streams.positions},
                         {"candidates", &round_streams.candidates},
                         {"drafts", &round_streams.drafts},
                         {"proposal_q", &round_streams.proposal_q}}) {
                    out.parts.emplace_back(name, digest_of(*stream));
                }
                return out;
            };
            const auto spec_taken = fingerprint(spec_round);
            const auto exec_taken = fingerprint(exec_round);
            if (spec_previous.parts.size() == spec_taken.parts.size()) {
                spec_parted_rounds += spec_taken.parted_against(spec_previous) > 0;
            }
            if (exec_previous.parts.size() == exec_taken.parts.size()) {
                exec_parted_rounds += exec_taken.parted_against(exec_previous) > 0;
            }
            spec_previous = spec_taken;
            exec_previous = exec_taken;
            if (round < 8 || round + 3 > rounds) {
                std::printf("  round %3d compared\n", round);
                std::fflush(stdout);
            }
        }

        // ---- the report ---------------------------------------------------------------------
        std::int32_t reds = 0;
        struct Aggregate {
            std::int32_t rounds = 0, clean = 0;
            std::int64_t maximum = 0;
            std::string detail;
        };
        std::vector<std::pair<std::string, Aggregate>> by_stream;
        for (const auto& row : totals) {
            auto found = std::find_if(by_stream.begin(), by_stream.end(),
                                      [&](const auto& item) { return item.first == row.label; });
            if (found == by_stream.end()) {
                by_stream.emplace_back(row.label, Aggregate{});
                found = by_stream.end() - 1;
            }
            ++found->second.rounds;
            found->second.clean += row.differing == 0;
            if (row.maximum > found->second.maximum) {
                found->second.maximum = row.maximum;
                found->second.detail  = row.detail;
            }
        }
        std::printf("fidelity over %d compared rounds, the executor's walk against the "
                    "specification's walk on identical inputs:\n", compared);
        for (const auto& [label, row] : by_stream) {
            const bool exact = row.clean == row.rounds;
            reds += not exact;
            std::printf("FIDELITY %-30s %s  rounds bit-exact %d/%d  max %lld ulp  first %s\n",
                        label.c_str(), exact ? "BIT-EXACT" : "PARTS", row.clean, row.rounds,
                        static_cast<long long>(row.maximum),
                        row.detail.empty() ? "none" : row.detail.c_str());
        }
        std::printf("determinism: %d of %d rounds part from the previous round on the "
                    "specification arm, %d on the executor arm (identical inputs, each behind its "
                    "own round reset)\n",
                    spec_parted_rounds, compared, exec_parted_rounds);
        reds += spec_parted_rounds + exec_parted_rounds;
        std::printf("arena: the round's peak through the executor's walk is %d MiB and %d MiB "
                    "through the tail of a %zu B (%.1f MiB) arena sized by "
                    "dspine_round_workspace_bytes\n",
                    peak_walk, peak_tail, arena_bytes, arena_bytes / (1024.0 * 1024.0));
        reds += census_reds;
        std::printf("binding: the drafter's binding census is %s\n",
                    census_reds == 0 ? "green" : "RED");
        std::printf("binding: %d of %d weight pairs and %d of %d parameter tensors are the same "
                    "device range\n",
                    bindings.rows - bindings.parted, bindings.rows,
                    parameters_bound.rows - parameters_bound.parted, parameters_bound.rows);
        reds += bindings.parted + parameters_bound.parted;
        std::printf("CHECKS %zu, REDS %d\n", totals.size() + bindings.rows + 2, reds);
        std::printf("VERDICT %s\n",
                    reds == 0
                        ? "the ported executor walk publishes the specification's bytes"
                        : "the ported executor walk and the specification's walk part; the "
                          "probe is the specification (22/22) - see the FIDELITY/BINDING rows");
        std::fflush(stdout);
        return reds == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::printf("STOP: %s\n", error.what());
        std::fflush(stdout);
        return 1;
    }
}
