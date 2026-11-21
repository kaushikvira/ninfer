// P2 stage 3a: the load side of the DSpine draft forward, measured on a real V3 drafter artifact
// on a CUDA device. The drafter binds fully (D-31: 53 objects, 1.31568 GiB); this answers the two
// questions a proposal path has to answer before any step of it can run, and answers them on the
// real weights rather than on a model of them:
//
//   load     the drafter's objects, the target output head and the target token embedding,
//            materialized on one device: the object count, the bytes, and the footprint the load
//            really costs. The 25 GiB target payload is never allocated - only these objects are
//            requested - which is what makes a drafter-only forward measurable at all (D-20).
//   shapes   every projection the draft layers, the code bridge and the proposal head name,
//            resolved against the compiled geometry of the linear family that would run it. Each
//            linear family resolves (representation, rows, columns, token interval) against a
//            registered shape table and refuses an uncompiled geometry on the host, before any
//            launch. So this is the load-side gate read out of the compiled binary, at the real
//            geometries and at T of both one token (a context append) and a whole draft block.
//
// The walk itself - the block arrangement, the embedding, the tap feature projection, a draft
// layer's split projections and SwiGLU MLP, the code bridge's read/codes/gates/message/write/
// finish - is the shape this probe measures, not a claim that it runs: a refused geometry is a
// step that cannot be issued, and the proposal path names three of them (D-26 criterion 3).
//
// The artifact is the one D-14/D-20 sized: …_dspine-nvfp4tables.ninfer.
//
// NINFER_TEST_ARTIFACT   the artifact (skip, exit 77, when unset)
// NINFER_TEST_DEVICE     the visible CUDA device ordinal to run on. Required, with no default: a
//                        device probe never guesses a card (D-29), and the card it landed on is
//                        printed (ordinal, name, PCI address) as part of the evidence.
// NINFER_TEST_STAGE      load | shapes | forward | all (default all)
//
// `forward` walks the proposal path on the real weights: the mask block, the tap feature
// projection and context norm, the five draft layers over the split projections and the paged
// context KV, the code bridge after each layer, the tower head over the proposal hidden, the
// score table, and the sampled serial tail at the production policy and at T=0 - then determinism
// (the same inputs twice), teeth (a rotated tap axis must move the drafts), the T=0 arm against
// the greedy pair, and a device-free-bytes check around the whole walk (it allocates nothing of
// its own).
//
// Run it pinned to the free card: CUDA_VISIBLE_DEVICES=1 NINFER_TEST_DEVICE=0 …

#include "artifact/binder.h"
#include "artifact/formats.h"
#include "artifact/materializer.h"
#include "artifact/placement.h"
#include "artifact/reader.h"
#include "artifact/schema.h"
#include "artifact/views.h"
#include "core/buffer.h"
#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "core/weight_view.h"
#include "models/common/bindings.h"
#include "models/common/dspine/config.h"
#include "models/common/dspine/dspine_serial_tail.h"
#include "models/common/dspine/weights.h"
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
#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "core/paged_kv_geometry.h"
#include "../../../tools/support/paged_kv_fixture.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <execinfo.h>
#include <exception>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::models;
namespace dspine = ninfer::models::dspine;

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
constexpr std::int32_t kSlots = 16;
// One production proposal round: batch 1, a 16-wide masked block, 15 proposal slots.
constexpr std::int32_t kBatch = 1;
// The target-tap context is carried over this many synthetic prompt positions. The block is
// 16 wide and the context is 16 wide, so every projection the walk issues runs at T = 16 - the
// shape gate's own second point - and the draft's context attention sees 16 keys.
constexpr std::int32_t kContext = 16;
// The paged context cache's page width, and the scale of the production draft's context profile.
constexpr std::int32_t kPage = 64;

// A harness that dies mid-stage must not lose the lines it already measured, and a device
// harness that dies in a driver call must name the call. SIGSEGV is the fault the D-34 forward
// walk parked on, so the probe installs its own reporter: a fault prints the frame it happened
// in and exits non-zero, instead of the shell reporting 139 with no line after the last
// measurement.
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

struct Counters {
    std::size_t steps    = 0;
    std::size_t gates    = 0;
    std::size_t checks   = 0;
    std::size_t failures = 0;
    std::size_t uploads  = 0;
};
Counters counters;

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

    void bind() {
        const auto& directory = reader.directory();
        const auto& text      = directory.component("text").config;
        target                = dspine::TargetGeometry{field(text, "hidden_size"),
                                                       field(text, "vocab_size"),
                                                       field(text, "num_hidden_layers")};
        config                = dspine::parse_config(directory.component("dspine"), target);
        binder                = std::make_unique<artifact::Binder>(reader);
        bindings              = std::make_unique<Bindings>(*binder);
        weights               = dspine::bind_weights(*bindings, config, target, "dspine");
        const std::vector<QType> declared{QType::Q8_G32_FP16, QType::FP8_E4M3FN_ROW_BF16,
                                          QType::BF16};
        head = bindings->direct_as_declared("text/output_head",
                                             artifact::Shape{target.vocabulary,
                                                             target.hidden_size},
                                             declared);
        embedding = bindings->direct_as_declared("text/token_embedding",
                                                 artifact::Shape{target.vocabulary,
                                                                  target.hidden_size},
                                                 declared);
        std::printf("dspine: hidden=%u layers=%u block=%u taps=%zu code=%u rank=%u head_topk=%u "
                    "mask=%u vocab=%u\n",
                    config.hidden_size, config.num_hidden_layers, config.block_size,
                    config.target_layer_ids.size(), config.bridge.code_size, config.bridge.rank,
                    config.head_topk, config.mask_token_id, target.vocabulary);
    }

    std::vector<WeightId> drafter() const {
        std::vector<WeightId> out{weights.feature_projection, weights.context_norm,
                                   weights.final_norm, weights.bridge.codebook,
                                   weights.bridge.early_read_weights, weights.bridge.read_weight,
                                   weights.bridge.feature_write_weights, weights.bridge.gate,
                                   weights.bridge.gate_bias, weights.bridge.predecessor_gate,
                                   weights.bridge.prev, weights.bridge.tg, weights.bridge.tsv,
                                   head, embedding};
        for (const auto& layer : weights.layers) {
            out.insert(out.end(),
                       {layer.input_norm, layer.post_attention_norm, layer.attention.query,
                        layer.attention.key, layer.attention.value, layer.attention.context_key,
                        layer.attention.context_value, layer.attention.query_norm,
                        layer.attention.key_norm, layer.attention.output, layer.mlp.gate,
                        layer.mlp.up, layer.mlp.down});
        }
        return out;
    }

    void materialize() {
        std::set<std::size_t> objects;
        std::vector<artifact::PlacementRequest> requests;
        for (const auto id : drafter()) {
            for (const auto& part : bindings->at(id).reference.binding.parts) {
                if (objects.insert(part.object.index).second) {
                    requests.push_back({part.object, device, artifact::HostAccess::CpuOnly});
                }
            }
        }
        std::uint64_t declared = 0;
        for (const auto index : objects) {
            declared += artifact::object_bytes(reader.directory().object(
                artifact::ObjectHandle{static_cast<std::size_t>(index)}));
        }
        std::printf("load: %zu objects, %llu bytes (%.5f GiB) requested on the device\n",
                    objects.size(), static_cast<unsigned long long>(declared), declared / kGiB);
        const auto [free_before, total] = memory();
        auto plan = artifact::plan_materialization(reader, requests);
        backing   = std::make_unique<artifact::MaterializedArtifact>(
            artifact::materialize(reader, std::move(plan), context.transfer_view()));
        context.synchronize();
        const auto [free_after, total_after] = memory();
        (void)total_after;
        const auto& stats = backing->stats();
        std::printf("load: %zu device objects, %.5f GiB of backing, h2d %.5f GiB in %.2f s\n",
                    stats.device_object_count, stats.device_capacity_bytes / kGiB,
                    stats.h2d_bytes / kGiB, stats.upload_seconds);
        std::printf("load: device free %llu -> %llu MiB (the load costs %llu MiB of %llu MiB)\n",
                    static_cast<unsigned long long>(free_before / 1024 / 1024),
                    static_cast<unsigned long long>(free_after / 1024 / 1024),
                    static_cast<unsigned long long>((free_before - free_after) / 1024 / 1024),
                    static_cast<unsigned long long>(total / 1024 / 1024));
    }

    [[nodiscard]] std::pair<std::uint64_t, std::uint64_t> memory() const {
        std::uint64_t free_bytes = 0, total_bytes = 0;
        cuda_ok(cudaMemGetInfo(&free_bytes, &total_bytes), "memory info");
        return {free_bytes, total_bytes};
    }

    [[nodiscard]] WeightView view(const WeightId id) const {
        return artifact::bind_view(bindings->at(id).reference, *backing, device);
    }

    [[nodiscard]] Weight weight(const WeightId id) const { return native_weight(view(id), 1.0F); }

    // An elementwise parameter (a norm vector, a gate bias) as the Tensor the ops take it in.
    [[nodiscard]] Tensor parameter(const WeightId id) const {
        const auto& shape = bindings->at(id).reference.shape;
        return weight_tensor(view(id), {std::int32_t(shape[0]), 1, 1, 1});
    }

    // A projection group the artifact binds apart over ONE packed object (a Q/K/V group, a
    // gate/up group) has its fused linear issued over that object. This builds the object's own
    // view over the bytes the loader already materialized once: no copy, no second
    // materialization, no re-quantization. The split names stay the authority on the geometry -
    // the packed rows are read back out of the object, not typed in.
    [[nodiscard]] Weight packed_parent(const WeightId id) const {
        const auto view = artifact::bind_view(bindings->at(id).reference, *backing, device);
        if (view.parts.size() != 1) {
            throw std::runtime_error(bindings->at(id).reference.name +
                                     ": a packed parent is one contiguous object range");
        }
        return over_object(view.parts.front().parent, bindings->at(id).reference.name, 0, 0);
    }

    // A stacked parameter ([layers, rows, columns]) projected one layer at a time: that layer's
    // row range of the one materialized object, as the [rows, columns] matrix its linear names.
    [[nodiscard]] Weight stacked(const WeightId id, const std::uint64_t layer,
                                  const std::uint64_t rows, const std::uint64_t columns) const {
        const auto view = artifact::bind_view(bindings->at(id).reference, *backing, device);
        if (view.parts.size() != 1) {
            throw std::runtime_error(bindings->at(id).reference.name +
                                     ": a stacked parameter is one contiguous object range");
        }
        return over_object(view.parts.front().parent, bindings->at(id).reference.name,
                            layer * rows * columns, rows * columns);
    }

    [[nodiscard]] Weight over_object(const WeightParent* parent, const std::string& name,
                                      const std::uint64_t begin,
                                      const std::uint64_t elements) const {
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

    // The manifest's own geometry for a bound parameter: the rows and columns it names and the
    // representation the artifact declares for it. No device backing is needed to read it, and a
    // linear family's shape gate is a host-side lookup, so this is what the gate is measured on.
    struct Declared {
        std::int32_t n, k;
        QType qtype;
    };

    // A stacked parameter ([layers, rows, columns]) is projected one slice at a time, so the
    // geometry a linear family is asked for is its last two axes.
    [[nodiscard]] Declared declared(const WeightId id) const {
        const auto& shape = bindings->at(id).reference.shape;
        return Declared{std::int32_t(shape[shape.size() - 2]),
                        std::int32_t(shape[shape.size() - 1]),
                        binder->declared_format(bindings->at(id).reference.name)};
    }

    [[nodiscard]] std::string geometry(const WeightId id) const {
        const auto& shape = bindings->at(id).reference.shape;
        return std::to_string(shape[0]) + "x" + std::to_string(shape[1]);
    }

    // The whole packed object a split projection is a row range of: what the fused linear family
    // would have to compile, as one geometry.
    [[nodiscard]] std::int32_t packed_rows(const WeightId id) const {
        const auto& parts = bindings->at(id).reference.binding.parts;
        return std::int32_t(reader.directory().tensor(parts.front().object).shape[0]);
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
    dspine::TargetGeometry target{};
    dspine::Config config;
    std::unique_ptr<artifact::Binder> binder;
    std::unique_ptr<Bindings> bindings;
    dspine::Weights weights;
    WeightId head{}, embedding{};
    std::unique_ptr<artifact::MaterializedArtifact> backing;
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
    ++counters.uploads;
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

// --- the geometries the proposal path names, against the compiled shape tables ---------------

void report_shapes(Probe& probe) {
    struct Request {
        std::string family;
        std::string label;
        WeightId parameter;
        std::int32_t rows = 0;  // 0 = the bound shape; otherwise the packed parent's rows
    };
    const auto& layers = probe.weights.layers;
    const auto& front  = layers.front();
    const auto& bridge = probe.weights.bridge;
    const std::vector<Request> requests{
        {"linear", "feature projection (the taps)", probe.weights.feature_projection, 0},
        {"linear_split", "attention query/key/value, packed", front.attention.query,
         probe.packed_rows(front.attention.query)},
        {"linear", "attention output", front.attention.output, 0},
        {"linear_swiglu", "mlp gate/up, packed", front.mlp.gate,
         probe.packed_rows(front.mlp.gate)},
        {"linear_add", "mlp down", front.mlp.down, 0},
        {"linear", "bridge read", bridge.read_weight, 0},
        {"linear", "bridge gate", bridge.gate, 0},
        {"linear", "bridge write", bridge.feature_write_weights, 0},
        {"linear_topk", "proposal head", probe.head, 0},
    };
    for (const std::int32_t tokens : {1, kSlots}) {
        for (const auto& request : requests) {
            const auto bound   = probe.declared(request.parameter);
            const auto rows    = request.rows != 0 ? request.rows : bound.n;
            const auto label   = request.label + " [" + std::to_string(rows) + ", " +
                                 std::to_string(bound.k) + "] " +
                                 std::string(artifact::format_name(bound.qtype));
            const auto policy = ops::LinearPolicy::A16Only;
            const auto attempt = [&]() -> std::size_t {
                if (request.family == "linear") {
                    return ops::linear_workspace_capacity_bytes(bound.qtype, rows, bound.k, policy,
                                                           tokens, tokens);
                }
                if (request.family == "linear_add") {
                    return ops::linear_add_workspace_capacity_bytes(bound.qtype, rows, bound.k, policy,
                                                               tokens, tokens);
                }
                if (request.family == "linear_swiglu") {
                    return ops::linear_swiglu_workspace_capacity_bytes(bound.qtype, rows, bound.k,
                                                                  policy, tokens, tokens);
                }
                if (request.family == "linear_topk") {
                    return ops::linear_topk_workspace_capacity_bytes(bound.qtype, rows, bound.k, tokens,
                                                                tokens);
                }
                const ops::ProjectionProfile profile{
                    ops::LinearProfile{bound.qtype, rows, bound.k, policy, QuantLayout::Contiguous},
                    {}};
                return ops::linear_split_workspace_capacity_bytes(
                    profile,
                    std::vector<std::int32_t>{std::int32_t(probe.config.query_width()),
                                              std::int32_t(probe.config.key_width()),
                                              std::int32_t(probe.config.key_width())},
                    tokens, tokens);
            };
            ++counters.steps;
            try {
                const auto bytes = attempt();
                std::printf("  COMPILED T=%-3d %s (workspace %zu B)\n", tokens, label.c_str(),
                            bytes);
            } catch (const std::exception& error) {
                ++counters.gates;
                std::printf("  REFUSED  T=%-3d %s : %s\n", tokens, label.c_str(), error.what());
            }
        }
    }
}

// --- the proposal path, step by step, on the real weights -----------------------------------
//
// The block arrangement, the target-tap context, the five draft layers over the split
// projections, the code bridge that follows each of them, the tower head over the proposal
// hidden, and the sampled serial tail. Every weight is the artifact's own; every activation is
// a deterministic synthetic field. The layout mirrors the DFlash2 draft forward
// (src/models/qwen3_5/execution/draft.cpp), which is the path the DSpine manifest's Uses name:
// query_projection_input feeds the split Q/K/V, context_input feeds the context projections that
// write the draft's KV pages, mlp_input and mlp_product feed the SwiGLU pair and the down
// projection.

// The gate count a run of the walk started at: the walk stops at a gate of ITS OWN, not at one
// an earlier stage of the same process reached.
std::size_t gate_base = 0;

// The last bridge's input residual: prepare_read and finish read and write the residual stream,
// so the reference needs the state the chain started from. Captured only for the reference arm.
Tensor* bridge_snapshot = nullptr;

// The code projection's output before normalize_codes overwrites it in place: the reference's
// normalize_codes input.
Tensor* codes_snapshot = nullptr;

void require(const bool condition, const char* message) {
    ++counters.checks;
    if (!condition) {
        ++counters.failures;
        std::printf("  RED   %s\n", message);
    }
    std::fflush(stdout);
}

// One step of the walk. A step the engine refuses (an uncompiled geometry, a rejected
// representation) is a GATE: printed with the engine's own words and counted, not a crash, so the
// probe reports how far the walk got and what stopped it. A step that faults is not caught here -
// the fault reporter owns that, and a gate is never allowed to be read as a measurement.
void step(const std::string& label, const std::function<void()>& body) {
    ++counters.steps;
    try {
        body();
        cuda_ok(cudaGetLastError(), "launch");
        std::printf("  RUN   %s\n", label.c_str());
    } catch (const std::exception& error) {
        ++counters.gates;
        std::printf("  GATE  %s : %s\n", label.c_str(), error.what());
    }
    std::fflush(stdout);
}

// A deterministic BF16 activation field, column-major: the same bytes for the same shape, on the
// host and therefore on the device. The walk's activations are synthetic by design; its weights
// are the artifact's, which is what makes the walk a measurement.
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

[[nodiscard]] std::int32_t dimension(const std::uint64_t value) {
    return static_cast<std::int32_t>(value);
}

struct Work {
    // The streams the path writes, at the shapes its steps name them by.
    Tensor ids, positions, context_positions, context_lengths, context_counts, lengths, valid,
        table_rows, anchors, taps, projected, context, hidden, residual, normalized, rms, codes,
        packed, message, delta, normed, stash, intermediate, query, key, value, attended,
        context_key, context_value, proposal, scratch_query, query_raw, key_raw,
        context_key_raw, codes_out;
    Tensor candidates, scores, table, tail_cand, drafts, proposal_q, index, idx_scores,
        positions_rng, configs, chain_drafts;
    std::vector<std::unique_ptr<tool::PackedKVFixture>> caches;
    std::unique_ptr<DeviceBuffer> page_table;
};

// The draft's context pages: one layer, one cache. The paged form is the one the draft's
// context attention reads (the registered D128/Hq32/Hkv8 profile); its record is the D128/Hkv8
// BF16-K / FP16-V plane the ops name.
[[nodiscard]] KVPageGeometry context_page_geometry(const std::int32_t head_dim,
                                                   const std::int32_t kv_heads) {
    return {std::uint32_t(kPage),
            {KVPlaneGeometry{DType::BF16, head_dim, kv_heads},
             KVPlaneGeometry{DType::FP16, head_dim, kv_heads}}};
}

// The walk's transient footprint, read out of the ops' own capacity queries at the geometries
// and column counts the walk issues. Nothing is allocated mid-step: this is sized once, here.
[[nodiscard]] std::size_t walk_workspace_bytes(Probe& probe) {
    const auto h    = dimension(probe.target.hidden_size);
    const auto taps = dimension(probe.config.target_layer_ids.size());
    const auto c    = dimension(probe.config.bridge.code_size);
    const auto r    = dimension(probe.config.bridge.rank);
    const auto inter = dimension(probe.config.intermediate_size);
    const auto head  = dimension(probe.target.vocabulary);
    const auto q    = dimension(probe.config.query_width());
    const auto k    = dimension(probe.config.key_width());
    const auto policy = ops::LinearPolicy::A16Only;
    const auto split_bytes = [&] {
        return ops::linear_split_workspace_capacity_bytes(
            ops::ProjectionProfile{ops::LinearProfile{
                probe.declared(probe.weights.layers.front().attention.query).qtype,
                std::int32_t(probe.packed_rows(probe.weights.layers.front().attention.query)),
                dimension(probe.config.key_width()), policy, QuantLayout::Contiguous},
                {}},
            std::vector<std::int32_t>{q, k, k}, kContext, kContext);
    };
    const auto at = [&](const WeightId id, const std::size_t rows, const std::size_t tokens) {
        const auto bound = probe.declared(id);
        return ops::linear_workspace_capacity_bytes(bound.qtype, std::int32_t(rows), bound.k,
                                                    policy, std::int32_t(tokens),
                                                    std::int32_t(tokens));
    };
    std::size_t bytes = 0;
    const auto every = [&](const char* label, const std::function<std::size_t()>& one) {
        try {
            bytes = std::max(bytes, one());
        } catch (const std::exception& error) {
            std::printf("  GATE  the capacity plan at %s : %s\n", label, error.what());
            ++counters.gates;
        }
    };
    every("feature projection", [&] {
        return at(probe.weights.feature_projection, std::size_t(h) * taps, kContext);
    });
    every("attention output", [&] {
        return at(probe.weights.layers.front().attention.output, h, kContext);
    });
    every("mlp down", [&] {
        return at(probe.weights.layers.front().mlp.down, h, kContext);
    });
    every("bridge gate", [&] { return at(probe.weights.bridge.gate, r, kContext); });
    every("bridge read", [&] { return at(probe.weights.bridge.read_weight, c, kContext); });
    every("bridge write", [&] {
        return at(probe.weights.bridge.feature_write_weights, h, kContext);
    });
    // The context key and value reach the device through the packed parent's split form
    // (see run_forward), so the split capacity is the one the walk needs.
    every("context key/value split", [&] { return split_bytes(); });
    every("gate/up swiglu split", [&] {
        return ops::linear_swiglu_workspace_capacity_bytes(
            probe.declared(probe.weights.layers.front().mlp.gate).qtype,
            std::int32_t(probe.packed_rows(probe.weights.layers.front().mlp.gate)), h, policy,
            kContext, kContext);
    });
    every("block q/k/v split", [&] { return split_bytes(); });
    every("proposal head topk", [&] {
        const auto bound = probe.declared(probe.head);
        return ops::linear_topk_workspace_capacity_bytes(bound.qtype, head, h, kContext - 1,
                                                         kContext - 1);
    });
    every("mlp down add", [&] {
        return ops::linear_add_workspace_capacity_bytes(
            probe.declared(probe.weights.layers.front().mlp.down).qtype, h,
            dimension(probe.config.intermediate_size), policy, kContext, kContext);
    });
    every("context attention", [&] {
        return ops::context_softmax_attention_workspace_capacity_bytes(
            ops::AttentionHeadGeometry{dimension(probe.config.head_dim),
                                       dimension(probe.config.num_attention_heads),
                                       dimension(probe.config.num_key_value_heads)},
            ops::ContextAttentionExecutionEnvelope{std::uint32_t(kContext),
                                                  std::uint32_t(kContext)},
            kContext, kContext, kBatch);
    });
    return bytes + (16 << 20);
}

[[nodiscard]] Work allocate(Probe& probe, const std::size_t workspace_bytes) {
    const auto h     = dimension(probe.target.hidden_size);
    const auto w     = dimension(probe.config.block_size);
    const auto b     = kBatch;
    const auto ctx   = kContext;
    const auto c     = dimension(probe.config.bridge.code_size);
    const auto r     = dimension(probe.config.bridge.rank);
    const auto taps  = dimension(probe.config.target_layer_ids.size());
    const auto q     = dimension(probe.config.query_width());
    const auto k     = dimension(probe.config.key_width());
    const auto d     = dimension(probe.config.head_dim);
    const auto inter = dimension(probe.config.intermediate_size);
    const auto slots = std::int32_t(w * b - 1);
    Work out;
    const auto integers = [&](const std::int64_t count,
                               const std::initializer_list<std::int32_t> shape) {
        return probe.buffer(std::size_t(4) * count, DType::I32, shape);
    };
    const auto stream = [&](const std::int64_t count,
                             const std::initializer_list<std::int32_t> shape) {
        return probe.buffer(std::size_t(2) * count, DType::BF16, shape);
    };
    const auto reals = [&](const std::int64_t count,
                            const std::initializer_list<std::int32_t> shape) {
        return probe.buffer(std::size_t(4) * count, DType::FP32, shape);
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
    out.proposal          = stream(std::int64_t(h) * slots * b, {h, slots, b, 1});
    out.intermediate      = stream(std::int64_t(inter) * w * b, {inter, w, b, 1});
    out.rms               = reals(std::int64_t(w) * b, {w, b, 1, 1});
    out.codes             = stream(std::int64_t(c) * w * b, {c, w, b, 1});
    out.packed            = stream(std::int64_t(3 * r) * w * b, {3 * r, w, b, 1});
    out.message           = stream(std::int64_t(r) * w * b, {r, w, b, 1});
    out.query             = stream(std::int64_t(q) * w * b, {d, q / d, w, b});
    out.key               = stream(std::int64_t(k) * w * b, {d, k / d, w, b});
    out.value             = stream(std::int64_t(k) * w * b, {d, k / d, w, b});
    out.attended          = stream(std::int64_t(q) * w * b, {d, q / d, w, b});
    out.context_key       = stream(std::int64_t(k) * ctx * b, {d, k / d, ctx, b});
    out.scratch_query     = stream(std::int64_t(q) * ctx * b, {q, ctx, b, 1});
    out.query_raw         = stream(std::int64_t(q) * w * b, {d, q / d, w, b});
    out.key_raw           = stream(std::int64_t(k) * w * b, {d, k / d, w, b});
    out.context_key_raw   = stream(std::int64_t(k) * ctx * b, {d, k / d, ctx, b});
    out.codes_out         = stream(std::int64_t(c) * w * b, {c, w, b, 1});
    out.context_value     = stream(std::int64_t(k) * ctx * b, {d, k / d, ctx, b});
    out.candidates        = integers(std::int64_t(kSlots) * slots * b, {kSlots, slots, b, 1});
    out.scores            = reals(std::int64_t(kSlots) * slots * b, {kSlots, slots, b, 1});
    out.idx_scores        = integers(std::int64_t(kSlots) * slots * b, {kSlots, slots, b, 1});
    out.table             = reals(std::int64_t(b) * kSlots * kSlots * slots,
                                  {b, kSlots, kSlots, slots});
    out.tail_cand         = integers(std::int64_t(b) * slots * kSlots, {b, slots, kSlots, 1});
    // The greedy pair's drafts, at the layout serial_chain publishes them in: [drafts, batch].
    // Sized with the rest of the walk's streams so the greedy pair costs the walk no allocation.
    out.chain_drafts      = integers(std::int64_t(slots) * b, {slots, b, 1, 1});
    out.drafts            = integers(std::int64_t(b) * slots, {b, slots, 1, 1});
    out.proposal_q        = reals(std::int64_t(b) * slots * kSlots, {b, slots, kSlots, 1});
    out.index             = integers(std::int64_t(b) * slots * kSlots, {b, slots, kSlots, 1});
    out.positions_rng     = integers(b, {b, 1, 1, 1});
    out.configs = probe.buffer(sizeof(ops::SamplingConfig) * std::size_t(b), DType::FP32,
                                {b, 1, 1, 1});
    const auto geometry =
        context_page_geometry(d, dimension(probe.config.num_key_value_heads));
    for (std::size_t layer = 0; layer < probe.weights.layers.size(); ++layer) {
        out.caches.push_back(
            std::make_unique<tool::PackedKVFixture>(probe.device, geometry, std::int32_t(1)));
    }
    // The I64 [logical_pages, table_rows] arena-relative byte-offset table: one logical page,
    // identity-mapped onto physical page 0.
    out.page_table = std::make_unique<DeviceBuffer>(probe.device, sizeof(std::int64_t));
    std::int64_t offset = 0;
    out.page_table->copy_from_host(&offset, sizeof(offset));
    std::printf("forward: the walk's whole transient footprint is one %zu B arena, sized once "
                "at allocation from the ops' own capacity queries\n",
                workspace_bytes);
    return out;
}

// The paged view of one layer's context cache.
[[nodiscard]] PagedKVBatchLayerView context_view(Work& work, const std::size_t layer,
                                                 const std::int32_t head_dim,
                                                 const std::int32_t kv_heads) {
    const Tensor table(work.page_table->span(), DType::I64, {1, 1, 1, 1});
    return {.k_pages      = work.caches[layer]->plane(0, table),
            .v_pages      = work.caches[layer]->plane(1, table),
            .head_dim     = head_dim,
            .num_kv_heads = kv_heads,
            .storage      = KvCacheStorage::BFloat16};
}

// The bridge that follows one draft layer: read, codes over the real codebook, the three packed
// gate projections into one [3R, width, batch] block, the message, the write, and the finish that
// refines the residual in place and fuses the next layer's norm. The LAST layer's `stash`, `rms`
// and `message` are the serial tail's.
void run_bridge(Probe& probe, Work& work, MemoryArena& arena, const std::size_t layer,
                const std::string& suffix) {
    const auto cuda      = probe.cuda();
    const auto& bridge   = probe.weights.bridge;
    const auto& layers   = probe.weights.layers;
    const auto last      = layer + 1 == layers.size();
    const auto h         = dimension(probe.target.hidden_size);
    const auto w         = dimension(probe.config.block_size);
    const auto b         = kBatch;
    const auto columns   = std::int32_t(w) * b;
    const auto c         = dimension(probe.config.bridge.code_size);
    const auto r         = dimension(probe.config.bridge.rank);
    const auto plane     = std::int64_t(r) * columns * 2;
    auto flat_residual = work.residual.view({h, columns, 1, 1});
    auto flat_normalized = work.normalized.view({h, columns, 1, 1});
    auto flat_rms = work.rms.view({columns, 1, 1, 1});
    auto flat_codes = work.codes.view({c, columns, 1, 1});
    if (last && bridge_snapshot != nullptr) {
        cuda_ok(cudaMemcpy(bridge_snapshot->data, flat_residual.data, flat_residual.bytes(),
                            cudaMemcpyDeviceToDevice),
                "snapshot the chain's input residual");
    }
    step("bridge read: rms of the layer stream" + suffix, [&] {
        ops::dspine_prepare_read(flat_residual, probe.config.bridge.eps, flat_normalized,
                                  flat_rms, cuda.stream());
    });
    step("bridge code projection" + suffix, [&] {
        const auto read =
            last ? probe.weight(bridge.read_weight)
                 : probe.stacked(bridge.early_read_weights, layer, std::uint64_t(c),
                                 std::uint64_t(h));
        auto codes_in  = work.codes.view({c, columns, 1, 1});
        auto codes_out = work.codes_out.view({c, columns, 1, 1});
        ops::linear(flat_normalized, read, codes_out, ops::LinearPolicy::A16Only, arena, cuda);
        codes_in = codes_out;
    });
    if (last && codes_snapshot != nullptr) {
        cuda_ok(cudaMemcpy(codes_snapshot->data, work.codes_out.data, work.codes_out.bytes(),
                            cudaMemcpyDeviceToDevice),
                "snapshot the code projection's output");
    }
    step("bridge normalize_codes over the real codebook" + suffix, [&] {
        auto codes_pre = work.codes_out.view({c, columns, 1, 1});
        auto codes_post = work.codes.view({c, columns, 1, 1});
        ops::dspine_normalize_codes(codes_pre, work.anchors, probe.weight(bridge.codebook), w,
                                    probe.config.bridge.eps, codes_post, cuda.stream());
    });
    step("bridge gate, predecessor and value projections into the packed block" + suffix, [&] {
        const auto base = static_cast<std::byte*>(work.packed.data);
        auto gate = Tensor(base, probe.device, DType::BF16, {r, columns, 1, 1});
        auto pred = Tensor(base + plane, probe.device, DType::BF16, {r, columns, 1, 1});
        auto prev = Tensor(base + 2 * plane, probe.device, DType::BF16, {r, columns, 1, 1});
        ops::linear(flat_codes, probe.weight(bridge.gate), gate, ops::LinearPolicy::A16Only, arena,
                    cuda);
        ops::linear(flat_codes, probe.weight(bridge.predecessor_gate), pred,
                    ops::LinearPolicy::A16Only, arena, cuda);
        ops::linear(flat_codes, probe.weight(bridge.prev), prev, ops::LinearPolicy::A16Only, arena,
                    cuda);
    });
    step("bridge message" + suffix, [&] {
        ops::dspine_message(work.packed, probe.parameter(bridge.gate_bias), work.message,
                            cuda.stream());
    });
    step("bridge write projection" + suffix, [&] {
        const auto write = probe.stacked(bridge.feature_write_weights, layer, std::uint64_t(h),
                                         std::uint64_t(r));
        auto message  = work.message.view({r, columns, 1, 1});
        auto delta    = work.delta.view({h, columns, 1, 1});
        ops::linear(message, write, delta, ops::LinearPolicy::A16Only, arena, cuda);
    });
    step("bridge finish: refines the residual in place, fuses the next layer's norm" + suffix,
         [&] {
             const auto next = last ? probe.parameter(probe.weights.final_norm)
                                    : probe.parameter(layers[layer + 1].input_norm);
             auto delta  = work.delta.view({h, columns, 1, 1});
             auto normed = work.normed.view({h, columns, 1, 1});
             ops::dspine_finish(flat_residual, flat_rms, delta, next,
                                probe.config.rms_norm_eps, normed, &work.stash, cuda.stream());
         });
}

// The head writes [candidate, slot, batch] - the shape serial_scores and serial_chain are handed
// (include/ninfer/ops/dspine_bridge.h:107-111: element (candidate, slot, batch) at
// candidate + 16*(slot + drafts*batch)). The tail is handed [batch, slot, candidate]
// (dspine_bridge.h:98-104, serial_tail_kernel's cand[b + batch*(slot + drafts*choice)]).
//
// The two ops declare TRANSPOSED shapes for one logical table, so the bridge is a real transpose
// and not a relabel: the element the tail reads for (batch, slot, candidate) is the head's
// (candidate, slot, batch), and the strides differ twice over - the slot and candidate axes swap,
// and the candidate axis carries batch*drafts (15 at the prod shape) where the head's carries the
// 16 candidates. Writing the head's bytes out under the head's own strides therefore hands the
// tail other slots' ids: real vocabulary ids, from the wrong slot, and the T=0 arm of the tail
// publishes ids the greedy pair never nominated. D-39's RED 2. The walk's own candidate-stream
// check below is what sees it.
void seed_tail_inputs(Probe& probe, Work& work) {
    const auto slots = work.candidates.ne[1];
    // The permute reads a stream the head wrote one step ago ON THE WALK'S STREAM. A host copy is
    // not sequenced against that stream, so without this the bridge can permute the PREVIOUS
    // round's candidate rows - the second half of D-39's RED 2, and the reason round 2 of a walk
    // carried ids round 1 had published. run_tail_case settles its launches the same way.
    const CudaDeviceScope settling(probe.device);
    cuda_ok(cudaDeviceSynchronize(), "the head's candidate rows before the tail's permute");
    const auto from  = download_i32(work.candidates);
    std::vector<std::int32_t> to(std::size_t(std::int64_t(kBatch) * slots * kSlots));
    for (std::int32_t batch = 0; batch < kBatch; ++batch) {
        for (std::int32_t slot = 0; slot < slots; ++slot) {
            for (std::int32_t cand = 0; cand < kSlots; ++cand) {
                to[std::size_t(batch + std::int64_t(kBatch)
                                            * (slot + std::int64_t(slots) * cand))] =
                    from[std::size_t(cand
                                     + std::int64_t(kSlots) * (slot + std::int64_t(slots) * batch))];
            }
        }
    }
    upload(work.tail_cand.data, to.data(), to.size() * sizeof(std::int32_t), "tail candidates");
}

// Reads both candidate streams back and counts the (batch, slot, candidate) triples where the two
// declared layouts do not say the same thing. 0 is the only answer a composed round can ship; see
// seed_tail_inputs for why the two layouts are not the same bytes. The caller reads it only on a
// pass that publishes (the first pass of a process publishes nothing), so the number is the number
// of a real round.
std::int32_t measure_tail_cand_layout(Probe& probe, Work& work) {
    const auto slots = std::int32_t(work.candidates.ne[1]);
    const CudaDeviceScope settling(probe.device);
    cuda_ok(cudaDeviceSynchronize(), "the candidate streams before the layout check");
    const auto head  = download_i32(work.candidates);
    const auto tail  = download_i32(work.tail_cand);
    std::int32_t misplaced = 0, first_slot = -1, first_cand = -1, read_id = 0, wrote_id = 0;
    for (std::int32_t batch = 0; batch < kBatch; ++batch) {
        for (std::int32_t slot = 0; slot < slots; ++slot) {
            for (std::int32_t cand = 0; cand < kSlots; ++cand) {
                const auto at_head = std::size_t(
                    cand + std::int64_t(kSlots) * (slot + std::int64_t(slots) * batch));
                const auto at_tail = std::size_t(
                    batch + std::int64_t(kBatch) * (slot + std::int64_t(slots) * cand));
                if (tail[at_tail] == head[at_head]) { continue; }
                if (misplaced++ == 0) {
                    first_slot = slot;
                    first_cand = cand;
                    read_id    = tail[at_tail];
                    wrote_id   = head[at_head];
                }
            }
        }
    }
    std::printf("        the tail's candidate stream: %d of %d ids are not where the tail's reader "
                "looks",
                int(misplaced), std::int32_t(kBatch) * kSlots * slots);
    if (misplaced > 0) {
        std::printf(" (first at slot %d, candidate %d: the tail reads %d, the head wrote %d)",
                    int(first_slot), int(first_cand), int(read_id), int(wrote_id));
    }
    std::printf("\n");
    std::fflush(stdout);
    return misplaced;
}

// The intermediate the two acceptance arms share: serial_scores writes the FP32 score table and
// the argmax of its own rows into idx, and serial_tail re-derives the argmax from the table. The
// composed round only agrees if the table the tail reads is the row the argmax collapsed, so the
// walk reads BOTH of its own streams back and compares them element by element: at every
// (batch, sender, slot), the argmax over the table's candidate axis (the tail's rule, first max
// wins ties) against the idx serial_scores published (the chain's rule). The first index where
// they part is where the acceptance path parts, and it says whether it is the table, the argmax of
// the table, or the walk that parts.
std::int32_t measure_table_against_idx(Probe& probe, Work& work) {
    const auto slots = std::int32_t(work.table.ne[3]);
    const CudaDeviceScope settling(probe.device);
    cuda_ok(cudaDeviceSynchronize(), "the score table and the idx before the comparison");
    const auto table = [&] {
        std::vector<float> values(std::size_t(std::int64_t(kBatch) * kSlots * kSlots * slots));
        download(values.data(), work.table.data, values.size() * sizeof(float), "score table");
        return values;
    }();
    const auto idx = download_i32(work.idx_scores);
    std::int32_t flips = 0, at_slot = -1, at_sender = -1, wanted = 0, published = 0;
    for (std::int32_t batch = 0; batch < kBatch; ++batch) {
        for (std::int32_t slot = 0; slot < slots; ++slot) {
            for (std::int32_t sender = 0; sender < kSlots; ++sender) {
                std::int32_t best = 0;
                for (std::int32_t cand = 1; cand < kSlots; ++cand) {
                    const auto at = std::size_t((((std::int64_t(slot) * kSlots + sender) * kSlots
                                                  + cand)) * kBatch + batch);
                    if (table[at] > table[std::size_t(
                            ((std::int64_t(slot) * kSlots + sender) * kSlots + best) * kBatch
                            + batch)]) {
                        best = cand;
                    }
                }
                const auto published_at = std::size_t(
                    (std::int64_t(batch) * slots + slot) * kSlots + sender);
                if (idx[published_at] == best) { continue; }
                if (flips++ == 0) {
                    at_slot   = slot;
                    at_sender = sender;
                    wanted    = best;
                    published = idx[published_at];
                }
            }
        }
    }
    std::printf("        argmax(the round's table) against the round's own idx: %d of %d flips",
                int(flips), std::int32_t(kBatch) * kSlots * slots);
    if (flips > 0) {
        std::printf(" (first at slot %d, sender %d: the table peaks at candidate %d, serial_scores"
                    " published %d)",
                    int(at_slot), int(at_sender), int(wanted), int(published));
    }
    std::printf("\n");
    std::fflush(stdout);
    return flips;
}

// The greedy pair over the streams of the round that just ran: dspine_serial_chain over the
// head's [16, drafts, batch] candidate rows and serial_scores' idx.
std::vector<std::int32_t> run_greedy_pair(Probe& probe, Work& work) {
    const auto slots = std::int32_t(work.candidates.ne[1]);
    ops::dspine_serial_chain(work.candidates.view({kSlots, slots, kBatch, 1}),
                             work.idx_scores.view({kSlots, slots, kBatch, 1}),
                             work.chain_drafts, probe.cuda().stream());
    cuda_ok(cudaGetLastError(), "serial chain launch");
    // The read-back of a stream the walk's stream produced is not sequenced against it: without
    // this the round reads the PREVIOUS round's chain and the comparison is a round-skew artifact
    // (which is what made the pair look a round behind the tail). run_tail_case settles the same
    // way.
    {
        const CudaDeviceScope settling(probe.device);
        cuda_ok(cudaDeviceSynchronize(), "the greedy pair before its readback");
    }
    return download_i32(work.chain_drafts);
}

// The whole proposal leg, on the real weights. It stops at the first gate: every later step
// consumes the gated one's output.
void run_forward(Probe& probe, Work& work, MemoryArena& arena) {
    const auto cuda    = probe.cuda();
    const auto h       = dimension(probe.target.hidden_size);
    const auto w       = dimension(probe.config.block_size);
    const auto b       = kBatch;
    const auto ctx     = kContext;
    const auto columns = std::int32_t(w) * b;
    const auto slots   = std::int32_t(columns - 1);
    const auto taps    = dimension(probe.config.target_layer_ids.size());
    const auto d       = dimension(probe.config.head_dim);
    const auto qh      = dimension(probe.config.num_attention_heads);
    const auto kvh     = dimension(probe.config.num_key_value_heads);
    const auto q       = dimension(probe.config.query_width());
    const auto k       = dimension(probe.config.key_width());
    const auto inter   = dimension(probe.config.intermediate_size);
    const auto& layers = probe.weights.layers;

    // The round's inputs - the block arrangement's scalars, the context geometry, the block's
    // positions and the tap field - are established by reset_round, which is the round reset. What
    // runs below is the walk proper, and it is a function of nothing but those bytes.
    gate_base = counters.gates;
    const auto gated = [&] { return counters.gates != gate_base; };

    step("block arrangement: the 16-slot mask block, mask token " +
             std::to_string(probe.config.mask_token_id) + " in the draft positions",
         [&] {
             ops::prepare_masked_block(work.anchors, work.lengths, work.valid,
                                        dimension(probe.config.mask_token_id), work.ids,
                                        work.positions, cuda);
         });
    step("token embedding of the block (the tower's own embedding)", [&] {
        auto ids     = work.ids.view({columns, 1, 1, 1});
        auto residual = work.residual.view({h, columns, 1, 1});
        ops::embedding(ids, probe.weight(probe.embedding), residual, cuda);
    });
    step("feature projection of the target taps: " + std::to_string(taps) + " taps x " +
             std::to_string(ctx) + " context positions", [&] {
        auto taps_in  = work.taps.view({std::int32_t(h * taps), std::int32_t(ctx * b), 1, 1});
        auto projected = work.projected.view({h, columns, 1, 1});
        ops::linear(taps_in, probe.weight(probe.weights.feature_projection), projected,
                    ops::LinearPolicy::A16Only, arena, cuda);
    });
    step("context norm", [&] {
        auto projected = work.projected.view({h, columns, 1, 1});
        auto context   = work.context.view({h, columns, 1, 1});
        ops::rmsnorm(projected, probe.parameter(probe.weights.context_norm),
                     probe.config.rms_norm_eps, false, context, cuda);
    });
    if (gated()) { return; }

    // The draft's context: each layer projects the normed target features with its own context
    // key and value (the packed ranges the artifact aliases onto that layer's K and V) and
    // appends the result to that layer's own KV pages. Those pages are what the layer's
    // attention then reads.
    for (std::size_t layer = 0; layer < layers.size(); ++layer) {
        const auto& block  = layers[layer];
        const auto  suffix = " [layer " + std::to_string(layer) + "]";
        step("context key/value projections, key norm, RoPE, append to layer " +
                 std::to_string(layer) + "'s pages",
             [&] {
                 const auto flat = std::int32_t(ctx * b);
                 auto context   = work.context.view({h, flat, 1, 1});
                 auto key_out   = work.context_key_raw.view({k, flat, 1, 1});
                 auto value_out = work.context_value.view({k, flat, 1, 1});
                 // The context key and value are the K and V RANGES of the layer's packed
                 // Q/K/V object, and this engine has no NVFP4 linear compiled at their
                 // apart geometry (1024 x 5120), nor at the paired one the DFlash path uses
                 // (2048 x 5120, linear_pair). The split family, which the drafter's own
                 // packed parent reaches (D-35), is the one compiled route to those bytes:
                 // the projection is issued over the parent and the Q rows it also computes
                 // are dropped. At a 16-wide context that is 4096 of 6144 output rows thrown
                 // away - a measured cost of the missing context geometry, recorded, not
                 // hidden behind a new kernel.
                 auto query_out = work.scratch_query.view({q, flat, 1, 1});
                 const std::array split{
                     std::move(query_out), std::move(key_out), std::move(value_out)};
                 ops::linear_split(context, probe.packed_parent(block.attention.query), split,
                                   cuda);
                 ops::rmsnorm(work.context_key_raw, probe.parameter(block.attention.key_norm),
                              probe.config.rms_norm_eps, false, work.context_key, cuda);
                 ops::rope(work.context_positions.view({flat, 1, 1, 1}),
                           dimension(probe.config.head_dim), probe.config.rope_theta,
                           work.context_key, cuda);
                 ops::kv_cache_append_prefix(
                     work.context_key, work.context_value, work.context_positions,
                     work.context_counts, work.table_rows,
                     ops::KVCacheAppendPrefixExecutionEnvelope{std::uint32_t(ctx),
                                                               std::uint32_t(ctx)},
                     context_view(work, layer, d, kvh), cuda);
             });
    }
    if (gated()) { return; }

    const ops::AttentionHeadGeometry geometry{d, qh, kvh};
    const auto scale            = float(1.0 / std::sqrt(double(d)));
    const auto context_envelope = ops::ContextAttentionExecutionEnvelope{std::uint32_t(ctx),
                                                                        std::uint32_t(ctx)};

    // The five draft layers. The bridge of the previous layer hands this layer a stream that is
    // already normed (the fused write), so the reference's `input_normed` flag runs the input norm
    // only where the write did not.
    bool pre_normed = false;
    for (std::size_t layer = 0; layer < layers.size(); ++layer) {
        const auto& block  = layers[layer];
        const auto  suffix = " [layer " + std::to_string(layer) + "]";
        if (!pre_normed) {
            step("input norm" + suffix, [&] {
                auto residual = work.residual.view({h, columns, 1, 1});
                auto hidden   = work.hidden.view({h, columns, 1, 1});
                ops::rmsnorm(residual, probe.parameter(block.input_norm),
                             probe.config.rms_norm_eps, false, hidden, cuda);
            });
        } else {
            step("input norm" + suffix + " taken from the previous layer's fused write", [&] {
                cuda_ok(cudaMemcpy(work.hidden.data, work.normed.data,
                                   std::size_t(std::int64_t(h) * columns) * 2,
                                   cudaMemcpyDeviceToDevice),
                        "carry the pre-normed stream");
            });
        }
        step("split query/key/value projection over the packed parent" + suffix, [&] {
            const std::array outputs{work.query_raw.view({q, columns, 1, 1}),
                                     work.key_raw.view({k, columns, 1, 1}),
                                     work.value.view({k, columns, 1, 1})};
            ops::linear_split(work.hidden.view({h, columns, 1, 1}),
                              probe.packed_parent(block.attention.query), outputs, cuda);
        });
        step("query and key head norms, then RoPE" + suffix, [&] {
            ops::rmsnorm(work.query_raw, probe.parameter(block.attention.query_norm),
                         probe.config.rms_norm_eps, false, work.query, cuda);
            ops::rmsnorm(work.key_raw, probe.parameter(block.attention.key_norm),
                         probe.config.rms_norm_eps, false, work.key, cuda);
            ops::rope(work.positions.view({columns, 1, 1, 1}), dimension(probe.config.head_dim),
                      probe.config.rope_theta, work.query, work.key, cuda);
        });
        step("context attention over the layer's pages" + suffix, [&] {
            ops::context_softmax_attention(work.query, work.key, work.value, work.context_lengths,
                                           work.valid, work.table_rows, geometry, scale,
                                           ConstPagedKVBatchLayerView(
                                               context_view(work, layer, d, kvh)),
                                           context_envelope, arena, work.attended, cuda);
        });
        step("attention output projection into the residual" + suffix, [&] {
            auto attended = work.attended.view({q, columns, 1, 1});
            auto residual = work.residual.view({h, columns, 1, 1});
            auto projected = work.projected.view({h, columns, 1, 1});
            // The attention output projection is [5120, 4096] (D-35 instantiated it for the
            // plain linear family), and this engine's linear_add carries its own shape table,
            // which has not that geometry - so the add is taken as its own op. Same math as the
            // DFlash2 path's project_add, one launch apart, and the geometry gap is reported
            // here rather than behind a new kernel.
            ops::linear(attended, probe.weight(block.attention.output), projected,
                        ops::LinearPolicy::A16Only, arena, cuda);
            ops::residual_add(projected, residual, cuda);
        });
        step("post-attention norm" + suffix, [&] {
            auto residual = work.residual.view({h, columns, 1, 1});
            auto hidden   = work.hidden.view({h, columns, 1, 1});
            ops::rmsnorm(residual, probe.parameter(block.post_attention_norm),
                         probe.config.rms_norm_eps, false, hidden, cuda);
        });
        step("gate/up projection over the packed parent (SwiGLU)" + suffix, [&] {
            auto hidden       = work.hidden.view({h, columns, 1, 1});
            auto intermediate = work.intermediate.view({inter, columns, 1, 1});
            ops::linear_swiglu(hidden, probe.packed_parent(block.mlp.gate), intermediate,
                               ops::LinearPolicy::A16Only, arena, cuda);
        });
        step("down projection into the residual" + suffix, [&] {
            auto intermediate = work.intermediate.view({inter, columns, 1, 1});
            auto residual    = work.residual.view({h, columns, 1, 1});
            ops::linear_add(intermediate, probe.weight(block.mlp.down), residual,
                            ops::LinearPolicy::A16Only, arena, cuda);
        });
        if (gated()) { return; }
        run_bridge(probe, work, arena, layer, suffix);
        if (gated()) { return; }
        pre_normed = true;
    }

    // The proposal hidden is the last fused write's normed stream, columns 1..15: column 0 is
    // the anchor, which is not a proposal. The head is the tower's own; the tail reads the last
    // write's stash - the pre-refinement residual, its rms, and the message as the packed Q.
    step("gather the 15 proposal columns out of the 16-wide block", [&] {
        // [H, width, batch] column-major: one batch row of the proposal is the H*(width-1)
        // elements behind the anchor, so the copy is (width-1)*H wide, batch tall - the same
        // 2-D form the DFlash2 proposal gather uses.
        const auto row  = std::size_t(std::int64_t(h) * slots) * 2;
        const auto src  = static_cast<std::byte*>(work.normed.data) + std::size_t(h) * 2;
        const auto spitch = std::size_t(std::int64_t(h) * columns) * 2;
        cuda_ok(cudaMemcpy2DAsync(work.proposal.data, row, src, spitch, row, std::int64_t(b),
                                  cudaMemcpyDeviceToDevice, cuda.stream()),
                "gather the proposal columns");
    });
    if (gated()) { return; }
    step("proposal logits per slot: the tower head's top-16 over 248077 valid rows", [&] {
        auto proposal  = work.proposal.view({h, slots, 1, 1});
        auto candidates = work.candidates.view({kSlots, slots, 1, 1});
        auto scores    = work.scores.view({kSlots, slots, 1, 1});
        ops::linear_topk(proposal, probe.weight(probe.head), 248077, candidates, scores, arena,
                         cuda);
    });
    if (gated()) { return; }
    step("the tail's candidate layout: the head's [candidate, slot] to the tail's [slot, candidate]",
         [&] { seed_tail_inputs(probe, work); });
    step("serial scores over the tower head, emitting the tail's FP32 score table", [&] {
        auto cand = work.candidates.view({kSlots, slots, b, 1});
        auto idx  = work.idx_scores.view({kSlots, slots, b, 1});
        ops::dspine_serial_scores(
            cand, work.stash, work.rms, work.message, work.anchors,
            probe.parameter(probe.weights.final_norm),
            probe.parameter(probe.weights.bridge.gate_bias), probe.weight(probe.head),
            probe.weight(probe.weights.bridge.tg), probe.weight(probe.weights.bridge.tsv), w, idx,
            &work.table, cuda.stream());
    });
}

// The sampled tail at a policy, and the drafts it publishes.
std::vector<std::int32_t> run_tail_case(Probe& probe, Work& work, const std::string& label,
                                        const float temperature, const std::int32_t top_k,
                                        const float top_p, const float min_p) {
    const auto cuda = probe.cuda();
    std::vector<ops::SamplingConfig> configs(std::size_t(kBatch), ops::SamplingConfig{});
    for (auto& config : configs) {
        config.temperature = temperature;
        config.top_k       = top_k;
        config.top_p       = top_p;
        config.min_p       = min_p;
        config.seed        = 0x50371153ULL;
    }
    upload(work.configs.data, configs.data(), configs.size() * sizeof(ops::SamplingConfig),
           "tail configs");
    std::vector<std::int32_t> positions(std::size_t(kBatch), 0);
    for (std::int32_t index = 0; index < kBatch; ++index) {
        positions[std::size_t(index)] = 1024 + index;
    }
    upload_i32(work.positions_rng, positions, "tail draw positions");
    {
        const CudaDeviceScope clearing(probe.device);
        cuda_ok(cudaMemset(work.drafts.data, 0, work.drafts.bytes()), "clear drafts");
    }
    ops::dspine_serial_tail(work.table, work.tail_cand,
                             static_cast<const ops::SamplingConfig*>(work.configs.data),
                             work.positions_rng, work.drafts, work.proposal_q, work.index,
                             cuda.stream());
    cuda_ok(cudaGetLastError(), "serial tail launch");
    cuda_ok(cudaDeviceSynchronize(), "serial tail sync");
    auto drafts = download_i32(work.drafts);
    std::printf("  RUN   %s: %zu drafts =", label.c_str(), drafts.size());
    for (std::size_t index = 0; index < drafts.size(); ++index) {
        std::printf(" %d", drafts[index]);
    }
    std::printf("\n");
    std::fflush(stdout);
    return drafts;
}

// --- the reference dump ----------------------------------------------------------------------
//
// The torch reference (/tmp/dspine-ref/sglang_dspine/models/dspine.py, CPU torch) recomputes the
// bridge chain from the chain's own inputs. To make the comparison non-circular it has to be fed
// the same bytes, so the probe writes them out: the chain's inputs, its outputs, and the
// synthetic-free values the reference needs (the norms, the gate bias, the rms vector). The
// quantized tables the chain indexes (the codebook rows, tg, tsv, the head rows) are NOT dumped:
// a dump of them is the device's own dequantization, and comparing that against the device is
// the vacuity the reference exists to exclude. The parity of those gather paths is P0b's and
// D-37's, and it is reported as inherited, not re-measured.
struct Dumper {
    std::FILE* file = nullptr;

    void put(const char* name, const Tensor& stream, const char* dtype) const {
        const auto elements = stream.numel();
        const auto width    = std::string(dtype) == "f4" ? 4u
                              : std::string(dtype) == "i4" ? 4u
                                                           : 2u;
        std::vector<std::byte> bytes(std::size_t(elements) * width);
        download(bytes.data(), stream.data, bytes.size(), name);
        std::fprintf(file, "%s %s %lld\n", name, dtype,
                     static_cast<long long>(elements));
        std::fwrite(bytes.data(), 1, bytes.size(), file);
    }

    void put_value(const char* name, const double value) const {
        std::fprintf(file, "%s f8 1\n", name);
        std::fwrite(&value, sizeof(double), 1, file);
    }

    void put_i32(const char* name, const std::vector<std::int32_t>& values) const {
        std::fprintf(file, "%s i4 %lld\n", name, static_cast<long long>(values.size()));
        std::fwrite(values.data(), sizeof(std::int32_t), values.size(), file);
    }
};

// The permuted-taps arm of the teeth test: the same synthetic tap field with its target-layer
// axis rotated by one, i.e. every context position is handed the concatenated feature vector of
// the WRONG set of target layers. The tap index is the only thing that moved.
void seed_taps_rotated(const Tensor& taps) {
    seed(taps);
    const auto h      = std::int64_t(taps.ne[0]);
    const auto layers = std::int64_t(taps.ne[1]);
    const auto width  = std::int64_t(taps.ne[2]);
    const auto batch  = std::int64_t(taps.ne[3]);
    if (layers < 2) { return; }
    std::vector<std::uint16_t> plane(std::size_t(std::int64_t(h) * width * batch));
    download(plane.data(), taps.data, plane.size() * 2, "read tap plane 0");
    const auto stride = std::size_t(std::int64_t(h) * width * batch);
    std::vector<std::uint16_t> all(std::size_t(stride * layers));
    download(all.data(), taps.data, all.size() * 2, "read taps");
    std::vector<std::uint16_t> moved(all.size());
    for (std::int64_t layer = 0; layer < layers; ++layer) {
        const auto from = std::size_t((layer + 1) % layers) * stride;
        const auto to   = std::size_t(layer) * stride;
        std::copy_n(all.begin() + from, stride, moved.begin() + to);
    }
    upload(taps.data, moved.data(), moved.size() * 2, "write rotated taps");
    (void)plane;
}

// --- the round's state, and the reset that owns it -------------------------------------------
//
// A proposal round is a pure function of the weights and of its own inputs ONLY if every stream
// it touches starts from a state the round itself establishes. These are the streams and regions
// the walk mutates in place, that is to say that survive a round and enter the next one:
//
//   the residual stream      `dspine_finish` refines it in place (include/ninfer/ops/
//                            dspine_bridge.h:41-47) and the two adds accumulate into it
//                            (`residual_add`, `linear_add`, run_forward's layer loop);
//   the bridge's read state  `rms` (dspine_prepare_read's per-column scale, read back by
//                            `dspine_finish` and by `dspine_serial_scores`) and the last bridge's
//                            `stash` + `message` - the tail's `raw`/`packed_q` inputs;
//   the 16-wide block         `ids`/`positions` plus every stream the five draft layers write
//                            (hidden, projected, context, the q/k/v and attended streams,
//                            intermediate, normalized, normed, delta, codes, packed);
//   the draft layers' pages  the paged context each layer's attention reads, one fixture per layer
//                            (written by `kv_cache_append_prefix` at the context positions);
//   the head and the table   the top-16 candidate rows and scores, the FP32 K x K score table and
//                            `serial_scores`' idx;
//   the tail's own state     `drafts`, `index`, the greedy pair's `chain_drafts`, and
//                            `proposal_q` - of which the tail writes ONLY the sender row its walk
//                            reaches per slot (src/models/common/dspine/dspine_serial_tail.cpp:
//                            258-268), so the other fifteen sender rows are the PREVIOUS round's;
//   the ops' scratch         the one arena the whole walk shares: the linear families'
//                            quantizer/workspace, `linear_topk`'s partial planes and its
//                            `group_done` arrival counters, which the merge kernel zeroes only at
//                            the counters' own scope (src/ops/linear/topk/merge.cu:158,129).
//
// The engine's own round opens exactly this way: `propose_batch_impl` begins with
// `execution.work.reset()` and re-derives its residual from `prepare_masked_block` + `embedding`
// (src/models/qwen3_5/execution/draft.cpp:347-353), so a decode round never reads the previous
// round's scratch, and the per-round buffers it hands out are the frame's own
// (src/models/qwen3_5/execution/draft.h:36-43). What follows is the DSpine round's equivalent of
// that reset, over the streams the DSpine walk owns.
void reset_round(Probe& probe, Work& work, MemoryArena& arena, const bool rotate_taps) {
    ++counters.steps;
    const CudaDeviceScope settling(probe.device);
    // The scratch the linear families and the top-k merge share, at the capacity the walk sized
    // it to. Zeroed, then the arena's cursor is rewound, so round N+2 cannot read round N's
    // partial planes, arrival counters or quantizer bounds.
    const auto scratch = arena.storage();
    cuda_ok(cudaMemset(scratch.data, 0, scratch.bytes), "reset the walk's shared workspace");
    arena.reset();
    // The streams the round writes, in whole or in part. Zeroing them is what turns "the same
    // inputs twice" into a statement: a stream the round writes only in part (proposal_q) is
    // defined over the whole of it after the reset, not over last round's sender rows.
    const std::vector<Tensor*> streams{
        &work.ids,           &work.positions,       &work.context_positions, &work.lengths,
        &work.valid,         &work.context_lengths, &work.context_counts,   &work.table_rows,
        &work.anchors,       &work.taps,            &work.projected,         &work.context,
        &work.hidden,        &work.residual,        &work.normalized,        &work.rms,
        &work.codes,         &work.codes_out,       &work.packed,            &work.message,
        &work.delta,         &work.normed,          &work.stash,             &work.intermediate,
        &work.query,         &work.key,             &work.value,             &work.attended,
        &work.context_key,   &work.context_value,   &work.scratch_query,     &work.query_raw,
        &work.key_raw,       &work.context_key_raw, &work.proposal,          &work.candidates,
        &work.scores,        &work.idx_scores,      &work.table,             &work.tail_cand,
        &work.drafts,        &work.proposal_q,      &work.index,             &work.chain_drafts,
        &work.positions_rng,  &work.configs};
    for (auto* stream : streams) {
        cuda_ok(cudaMemset(stream->data, 0, stream->bytes()), "reset a round stream");
    }
    // The draft layers' paged context: the whole record set, so a page the round does not append
    // to cannot carry last round's key or value into this round's attention.
    for (const auto& cache : work.caches) {
        const auto pages = cache->storage();
        cuda_ok(cudaMemset(pages.data, 0, pages.bytes), "reset a layer's context pages");
    }
    // The round's own inputs: the block arrangement and its positions, the context geometry, the
    // tail's draw coordinates, and the tap field. Everything after this point is the walk.
    const std::vector<std::int32_t> anchors{40901}, lengths{kContext}, valid{kSlots},
        context_lengths{kContext}, context_counts{kContext}, table_rows{0};
    upload_i32(work.anchors, anchors, "anchors");
    upload_i32(work.lengths, lengths, "block lengths");
    upload_i32(work.valid, valid, "valid columns");
    upload_i32(work.context_lengths, context_lengths, "context lengths");
    upload_i32(work.context_counts, context_counts, "context counts");
    upload_i32(work.table_rows, table_rows, "page table rows");
    std::vector<std::int32_t> context_positions(std::size_t(kContext * kBatch));
    for (std::int32_t index = 0; index < kContext * kBatch; ++index) {
        context_positions[std::size_t(index)] = index;
    }
    upload_i32(work.context_positions, context_positions, "context positions");
    std::vector<std::int32_t> block_positions(std::size_t(kSlots * kBatch));
    for (std::int32_t index = 0; index < kSlots * kBatch; ++index) {
        block_positions[std::size_t(index)] = kContext + index;
    }
    upload_i32(work.positions, block_positions, "block positions");
    std::vector<std::int32_t> draws(std::size_t(kBatch));
    for (std::int32_t index = 0; index < kBatch; ++index) {
        draws[std::size_t(index)] = 1024 + index;
    }
    upload_i32(work.positions_rng, draws, "tail draw positions");
    if (rotate_taps) { seed_taps_rotated(work.taps); } else { seed(work.taps); }
}

// A FNV-1a digest over a device stream. "The same bytes" has to be a statement about every byte of
// the round, not about the fifteen ids the probe happens to print: two rounds whose drafts agree
// while their score tables part are two rounds that got lucky.
[[nodiscard]] std::uint64_t digest(const void* bytes, const std::size_t count) {
    const auto* data = static_cast<const std::uint8_t*>(bytes);
    std::uint64_t hash = 14695981039346656037ULL;  // FNV-1a 64-bit offset basis
    for (std::size_t index = 0; index < count; ++index) {
        hash ^= static_cast<std::uint64_t>(data[index]);
        hash *= 1099511628211ULL;  // the FNV-1a 64-bit prime
    }
    return hash;
}

[[nodiscard]] std::uint64_t digest_stream(const Tensor& stream) {
    std::vector<std::byte> bytes(std::size_t(stream.bytes()));
    download(bytes.data(), stream.data, bytes.size(), "fingerprint a round stream");
    return digest(bytes.data(), bytes.size());
}

// Every byte the round owns, digested by name: the streams, the five layers' pages, and the ids.
struct Fingerprint {
    std::vector<std::pair<std::string, std::uint64_t>> parts;

    [[nodiscard]] std::vector<std::string> parted_against(const Fingerprint& other) const {
        std::vector<std::string> parted;
        for (std::size_t index = 0; index < parts.size(); ++index) {
            if (parts[std::size_t(index)].second
                != other.parts[std::size_t(index)].second) {
                parted.push_back(parts[std::size_t(index)].first);
            }
        }
        return parted;
    }
};

[[nodiscard]] Fingerprint fingerprint_round(Probe& probe, Work& work) {
    const CudaDeviceScope settling(probe.device);
    cuda_ok(cudaDeviceSynchronize(), "the round's bytes before the fingerprint");
    Fingerprint out;
    const std::vector<std::pair<std::string, Tensor*>> streams{
        {"block ids", &work.ids},
        {"block positions", &work.positions},
        {"taps", &work.taps},
        {"feature projection", &work.projected},
        {"context norm", &work.context},
        {"hidden", &work.hidden},
        {"residual (refined in place)", &work.residual},
        {"normalized", &work.normalized},
        {"rms (the bridge's read scale)", &work.rms},
        {"codes", &work.codes},
        {"packed gate/predecessor/value", &work.packed},
        {"message (the tail's packed_q)", &work.message},
        {"delta", &work.delta},
        {"normed (the proposal hidden)", &work.normed},
        {"stash (the tail's raw)", &work.stash},
        {"mlp intermediate", &work.intermediate},
        {"query", &work.query},
        {"key", &work.key},
        {"value", &work.value},
        {"attended", &work.attended},
        {"context key", &work.context_key},
        {"context value", &work.context_value},
        {"proposal columns", &work.proposal},
        {"head candidate rows", &work.candidates},
        {"head score rows", &work.scores},
        {"serial_scores idx", &work.idx_scores},
        {"serial_scores table", &work.table},
        {"tail candidate stream", &work.tail_cand},
        {"tail drafts", &work.drafts},
        {"tail proposal_q", &work.proposal_q},
        {"tail idx", &work.index},
        {"greedy pair drafts", &work.chain_drafts}};
    for (const auto& [name, stream] : streams) {
        out.parts.emplace_back(name, digest_stream(*stream));
    }
    // The fixtures own their page storage outside the probe's own buffer list, so the readback is
    // a plain device-to-host copy of the region the append writes, not a transfer through
    // owning_buffer.
    for (std::size_t layer = 0; layer < work.caches.size(); ++layer) {
        const auto pages = work.caches[layer]->storage();
        std::vector<std::byte> bytes(std::size_t(pages.bytes));
        cuda_ok(cudaMemcpy(bytes.data(), pages.data, bytes.size(), cudaMemcpyDeviceToHost),
                "fingerprint a layer's pages");
        out.parts.emplace_back("context pages of layer " + std::to_string(layer),
                               digest(bytes.data(), bytes.size()));
    }
    return out;
}


// One full pass: the block walk, the head, the score table, then the tail at both policies.
struct Round {
    // The round's own published drafts, its own greedy pair, and the head's candidate rows it
    // published out of: every identity is a statement about ONE round's bytes.
    std::vector<std::int32_t> sampled, greedy, chain, head;
    std::int32_t table_flips = -1;
    std::int32_t misplaced_cand = -1;
    bool complete = false;
    // Every byte the round owns, digested. Taken only on the rounds the identity rows judge.
    std::optional<Fingerprint> bytes;
};

[[nodiscard]] Round walk_and_draft(Probe& probe, Work& work, MemoryArena& arena,
                                   const bool rotate_taps, const bool identity = false) {
    // Every round is its own reset plus the walk: the shared workspace, the streams the round
    // writes in part, the layers' context pages, and the round's own inputs. Without it, "the same
    // inputs twice" is not what the determinism row feeds the engine.
    reset_round(probe, work, arena, rotate_taps);
    const auto gates = counters.gates;
    run_forward(probe, work, arena);
    Round out;
    out.complete = counters.gates == gates;
    if (!out.complete) { return out; }
    const auto tail = [&](const char* label, const float temperature, const std::int32_t top_k,
                         const float top_p, const float min_p) {
        try {
            ++counters.steps;
            return run_tail_case(probe, work, label, temperature, top_k, top_p, min_p);
        } catch (const std::exception& error) {
            ++counters.gates;
            std::printf("  GATE  the sampled tail at %s : %s\n", label, error.what());
            return std::vector<std::int32_t>{};
        }
    };
    out.sampled = tail("T=0.9 top_k=20 top_p=1.0 min_p=0.05 (the production policy)", 0.9f, 20,
                        1.0f, 0.05f);
    out.greedy  = tail("T=0 (the greedy degenerate of the same op)", 0.0f, 20, 1.0f, 0.0f);
    // Every identity below is a statement about ONE round: the greedy pair runs over this round's
    // idx, and the candidate-layout count reads this round's cand streams. Taking them after a
    // LATER round (the teeth round rotates the taps) compares one round's drafts to another's
    // inputs - which is how D-39's "15 real ids vs the tail's ids" was produced.
    const auto chain = [&](const char* label) {
        try {
            ++counters.steps;
            return run_greedy_pair(probe, work);
        } catch (const std::exception& error) {
            ++counters.gates;
            std::printf("  GATE  the greedy pair at %s : %s\n", label, error.what());
            return std::vector<std::int32_t>{};
        }
    };
    out.chain = chain("this round");
    out.head  = download_i32(work.candidates);
    std::printf("  the round's own greedy pair =");
    for (const auto one : out.chain) { std::printf(" %d", one); }
    std::printf("\n");
    std::fflush(stdout);
    out.misplaced_cand = measure_tail_cand_layout(probe, work);
    out.table_flips    = measure_table_against_idx(probe, work);
    if (identity && out.bytes == std::nullopt) { out.bytes = fingerprint_round(probe, work); }
    return out;
}

// --- the multi-round rows ---------------------------------------------------------------------
//
// WHAT THIS HARNESS CAN AND CANNOT MEASURE ABOUT ACCEPTANCE. The engine's accept entry
// (`ops::speculative_accept_sparse_drafts`, include/ninfer/ops/speculative_accept.h:65-110) verifies
// a round against the TARGET's logits over the drafted columns - `logits [248320, K+1, B]` plus
// `target_tokens`, the target's own argmax over them - exactly as prod wires it
// (src/models/qwen3_5/program/decision.cpp:76-92: `project_decode` puts the TARGET hidden through
// the tower head, `argmax` collapses it, `decide_decode` hands both to the acceptor). This probe
// loads the drafter, the tower head and the token embedding and deliberately never materializes
// the 25 GiB target payload (the load contract at the head of this file), and its activations are
// the synthetic tap field `seed` writes, not a real prompt's target features. So there is no target
// forward here, hence no target logits, hence NO ACCEPTED-DRAFTS NUMBER THAT MEANS WHAT PROD'S
// 2.998 MEANS. Every rate below is the drafter measured against the streams that produced it, and
// is labelled as such: the tail's own q checked against the tail's own draws, the tail's ids
// checked against the tower head's own argmax. They are drafting-quality diagnostics, not
// acceptance, and none of them is the P3 criterion-5 measurement.
struct RoundRate {
    std::int32_t published  = 0;  // proposal slots that published a draft id
    std::int32_t contract   = 0;  // slots whose (q, candidates, draft) break the accept contract
    std::int32_t against_head_top1 = 0;  // drafts equal to the head's own rank-0 of that slot
    std::int32_t head_prefix       = 0;  // ... as the longest matching prefix (an accept prefix)
    std::int32_t arms_agree        = 0;  // slots where T=0.9 and T=0 published the same id
    std::int32_t arm_prefix        = 0;  // ... as the longest matching prefix
    float q_drawn             = 0.0F;  // the q the tail put on the id it drew, mean over slots
    float q_rank0             = 0.0F;  // the q on the slot's rank-0 candidate, mean over slots
    const char* contract_reason = nullptr;
};

// One publishing round's rates. The accept-contract check is the engine's own
// `serial_tail_accept_q_error` over the device streams, at the layouts the accept entry declares;
// it is a VALIDITY check on the round (q normalised, the drawn id one of its candidates with
// positive mass, no repeated ids), which is what `speculative_accept_sparse_drafts` requires of a
// proposal - it is NOT a comparison of the drafts against any target.
[[nodiscard]] RoundRate rate_round(Probe& probe, Work& work, const Round& round) {
    const CudaDeviceScope settling(probe.device);
    cuda_ok(cudaDeviceSynchronize(), "the round's tail streams before the rates");
    const auto slots = std::int32_t(work.candidates.ne[1]);
    RoundRate out;
    const auto q     = [&] {
        std::vector<float> values(std::size_t(std::int64_t(kBatch) * slots * kSlots));
        download(values.data(), work.proposal_q.data, values.size() * sizeof(float),
                 "the tail's proposal q");
        return values;
    }();
    const auto tail_cand = download_i32(work.tail_cand);
    std::vector<std::int32_t> drawn = round.sampled;
    const char* reason              = common::dspine::serial_tail_accept_q_error(
        q.data(), tail_cand.data(), drawn.data(), kSlots, slots, kBatch, 1e-4F);
    out.contract_reason = reason;
    out.contract        = reason == nullptr ? 0 : kBatch * slots;
    float q_drawn_total = 0.0F, q_rank0_total = 0.0F;
    std::int32_t prefix = 0, arm_prefix = 0;
    bool prefix_open = true, arm_open = true;
    for (std::int32_t batch = 0; batch < kBatch; ++batch) {
        for (std::int32_t slot = 0; slot < slots; ++slot) {
            const auto at_cand = [&](const std::int32_t candidate) {
                return tail_cand[std::size_t(batch
                                             + std::int64_t(kBatch) * (slot + slots * candidate))];
            };
            const auto at_q = [&](const std::int32_t candidate) {
                return q[std::size_t(batch
                                     + std::int64_t(kBatch) * (slot + slots * candidate))];
            };
            // The head's own ranking of the same slot, [candidate, slot, batch] with the candidate
            // axis first: rank 0 is the head's argmax of the tower head over this slot's proposal
            // hidden. The tail draws out of THAT set, so agreement with it is self-referential.
            const auto at_head = [&](const std::int32_t candidate) {
                return round.head[std::size_t(candidate
                                               + std::int64_t(kSlots) * (slot + slots * batch))];
            };
            const auto draft = drawn[std::size_t(batch + std::int64_t(kBatch) * slot)];
            out.published += draft != 0;
            std::int32_t chosen = -1;
            for (std::int32_t candidate = 0; candidate < kSlots; ++candidate) {
                if (at_cand(candidate) == draft) { chosen = candidate; break; }
            }
            if (chosen >= 0) { q_drawn_total += at_q(chosen); }
            q_rank0_total += at_q(0);
            const auto hits_head = at_head(0) == draft;
            out.against_head_top1 += hits_head;
            prefix_open = prefix_open && hits_head;
            prefix += prefix_open;
            const auto agrees = round.greedy[std::size_t(batch + std::int64_t(kBatch) * slot)]
                           == draft;
            out.arms_agree += agrees;
            arm_open = arm_open && agrees;
            arm_prefix += arm_open;
        }
    }
    out.head_prefix = prefix;
    out.arm_prefix  = arm_prefix;
    out.q_drawn     = q_drawn_total / float(std::max(1, out.published));
    out.q_rank0     = q_rank0_total / float(kBatch * slots);
    return out;
}

// The multi-round run: a warmup pass, then N rounds of the SAME inputs, each with its own round
// reset, and the mean/range of the rates. `NINFER_TEST_ROUNDS` raises N; the floor is the task's 32.
void measure_rounds(Probe& probe, Work& work, MemoryArena& arena) {
    const auto wanted = setting("NINFER_TEST_ROUNDS", "32");
    const auto asked  = std::atoi(wanted.c_str());
    const auto rounds = std::max(32, asked > 0 ? asked : 32);
    const auto slots  = std::int32_t(work.candidates.ne[1]);
    std::printf("multi-round: %d rounds of identical inputs, each behind its own round reset\n",
                rounds + 1);
    std::fflush(stdout);
    // The warmup pass: the first pass of a process publishes nothing (the module-load pass, D-39).
    auto warm = walk_and_draft(probe, work, arena, false);
    if (!warm.complete) {
        std::printf("  GATE  the warmup round did not complete: no multi-round rows\n");
        return;
    }
    struct Series {
        std::vector<std::int32_t> published, contract, top1, head_prefix, agree, arm_prefix;
        std::vector<float> q_drawn, q_rank0;
    };
    Series series;
    std::int32_t parted_rounds = 0, same_as_warmup = 0;
    std::optional<Fingerprint> previous;
    std::int32_t contract_refusals = 0;
    for (std::int32_t round = 0; round < rounds; ++round) {
        auto taken = walk_and_draft(probe, work, arena, false, true);
        if (!taken.complete) {
            std::printf("  GATE  round %d did not complete\n", int(round));
            return;
        }
        const auto rates = rate_round(probe, work, taken);
        series.published.push_back(rates.published);
        series.contract.push_back(rates.contract);
        series.top1.push_back(rates.against_head_top1);
        series.head_prefix.push_back(rates.head_prefix);
        series.agree.push_back(rates.arms_agree);
        series.arm_prefix.push_back(rates.arm_prefix);
        series.q_drawn.push_back(rates.q_drawn);
        series.q_rank0.push_back(rates.q_rank0);
        contract_refusals += rates.contract_reason != nullptr;
        if (contract_refusals == 1) {
            std::printf("        the accept contract refused a round: %s\n",
                        rates.contract_reason);
        }
        // Round identity: the bytes of this round against the previous one, on identical inputs.
        std::vector<std::string> parted;
        if (taken.bytes.has_value() && previous.has_value()) {
            parted = taken.bytes->parted_against(*previous);
            parted_rounds += not parted.empty();
            if (not parted.empty() && round < 4) {
                std::printf("        round %d parts from round %d in %zu of %zu byte streams (first:"
                            " %s)\n",
                            int(round + 1), int(round), parted.size(),
                            taken.bytes->parts.size(), parted.front().c_str());
            }
        }
        previous = taken.bytes;
        same_as_warmup += taken.sampled == warm.sampled;
        if (round < 8 || round + 4 >= rounds) {
            std::printf("  round %3d: %2d/%d drafts published, %2d against the head's own argmax "
                        "(prefix %2d), %2d against the T=0 arm (prefix %2d), contract %s, "
                        "q(drawn) %.4f q(rank0) %.4f\n",
                        int(round), int(rates.published), std::int32_t(kBatch * slots),
                        int(rates.against_head_top1), int(rates.head_prefix),
                        int(rates.arms_agree), int(rates.arm_prefix),
                        rates.contract_reason == nullptr ? "ok" : "REFUSED",
                        double(rates.q_drawn), double(rates.q_rank0));
            std::fflush(stdout);
        }
        (void)parted;
    }
    const auto spread = [](const std::vector<std::int32_t>& values) {
        const auto [low, high] = std::minmax_element(values.begin(), values.end());
        const double mean      = std::accumulate(values.begin(), values.end(), 0.0)
                            / double(std::max<std::size_t>(1, values.size()));
        return std::make_tuple(mean, int(*low), int(*high));
    };
    const auto [published, published_low, published_high] = spread(series.published);
    const auto [top1, top1_low, top1_high]                 = spread(series.top1);
    const auto [head_prefix, head_prefix_low, head_prefix_high] = spread(series.head_prefix);
    const auto [agree, agree_low, agree_high]                   = spread(series.agree);
    const auto [arm_prefix, arm_prefix_low, arm_prefix_high]    = spread(series.arm_prefix);
    const auto q_mean = [](const std::vector<float>& values) {
        return std::accumulate(values.begin(), values.end(), 0.0) / double(std::max<std::size_t>(
                   1, values.size()));
    };
    std::printf("  multi-round over %d rounds, at the production policy (T=0.9, top_k 20, top_p 1.0,"
                " min_p 0.05):\n",
                rounds);
    std::printf("        drafts published per round: mean %.3f, range %d..%d  of %d slots\n",
                published, published_low, published_high, std::int32_t(kBatch * slots));
    std::printf("        SELF-REFERENTIAL (the tail's ids against the tower head that generated "
                "them):\n");
    std::printf("        agreement with the head's own argmax: mean %.3f, range %d..%d; as an "
                "accept-prefix mean %.3f, range %d..%d\n",
                top1, top1_low, top1_high, head_prefix, head_prefix_low, head_prefix_high);
    std::printf("        SELF-REFERENTIAL (the tail's T=0.9 arm against its own T=0 arm, same "
                "table):\n");
    std::printf("        arms agree: mean %.3f, range %d..%d; as an accept-prefix mean %.3f, "
                "range %d..%d\n",
                agree, agree_low, agree_high, arm_prefix, arm_prefix_low, arm_prefix_high);
    std::printf("        the q the tail put on the id it drew: mean %.5f; on its slot's rank-0 "
                "candidate: mean %.5f\n",
                q_mean(series.q_drawn), q_mean(series.q_rank0));
    std::printf("        the accept contract refused %d of %d rounds\n", contract_refusals,
                rounds);
    std::printf("  identity: rounds whose published ids equal the warmup round's: %d of %d\n",
                same_as_warmup, rounds);
    require(parted_rounds == 0,
            "two rounds of identical inputs did not publish identical bytes");
    require(published == double(kBatch * slots),
            "a round published fewer drafts than it has slots: the tail has no early abort, so "
            "every slot must publish");
    std::fflush(stdout);
}

} // namespace

int main(const int argc, char** argv) {
    (void)argc;
    (void)argv;
    // A probe that dies mid-stage must not lose the lines it already measured, and a device
    // fault must name itself: SIGSEGV is the fault the D-34 walk parked on, with the shell's 139
    // as the only evidence. The reporter prints the faulting signal and exits non-zero.
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
    const std::string stage = setting("NINFER_TEST_STAGE", "all");
    const auto wanted = [&](const std::string_view one) {
        return stage == "all" || stage == one;
    };
    try {
        Probe probe(artifact_path, device);
        work_device = probe.device;
        owner       = &probe;
        probe.bind();
        if (wanted("load")) {
            probe.materialize();
        }
        if (wanted("shapes")) {
            std::printf("the linear geometries the proposal path names:\n");
            report_shapes(probe);
        }
        if (wanted("forward")) {
            if (probe.backing == nullptr) { probe.materialize(); }
            std::size_t bytes = 0;
            try {
                bytes = walk_workspace_bytes(probe);
            } catch (const std::exception& error) {
                // A geometry the linear family has not compiled is the walk's own first gate;
                // report it the way the walk reports a gate and size the arena from the walk's
                // own floor, so the probe still says how far the walk gets.
                std::printf("  GATE  the walk's capacity plan : %s\n", error.what());
                ++counters.gates;
                bytes = 768u << 20;
            }
            auto arena_storage = DeviceBuffer(probe.device, bytes);
            MemoryArena arena(arena_storage.span());
            auto work = allocate(probe, bytes);
            auto codes_stream = probe.buffer(
                std::size_t(std::int64_t(dimension(probe.config.bridge.code_size)) *
                            dimension(probe.config.block_size) * kBatch) * 2,
                DType::BF16, {dimension(probe.config.bridge.code_size),
                              dimension(probe.config.block_size), kBatch, 1});
            auto snapshot = probe.buffer(
                std::size_t(std::int64_t(dimension(probe.target.hidden_size)) *
                                  dimension(probe.config.block_size) * kBatch) * 2,
                DType::BF16, {dimension(probe.target.hidden_size),
                              dimension(probe.config.block_size), kBatch, 1});
            bridge_snapshot = &snapshot;
            codes_snapshot  = &codes_stream;
            // The arms judge the walk's OWN gates: a capacity-plan gate above is a geometry
            // report, not a failure of the arms below.
            const auto gate_floor = counters.gates;
            const auto complete   = [&] { return counters.gates == gate_floor; };

            std::printf("the proposal path, on the real weights:\n");
            std::uint64_t free_before = 0, total = 0;
            cuda_ok(cudaMemGetInfo(&free_before, &total), "memory info");
            auto first = walk_and_draft(probe, work, arena, false);
            std::uint64_t free_after = 0;
            cuda_ok(cudaMemGetInfo(&free_after, &total), "memory info");
            if (complete()) {
                std::printf(
                    "  GREEN the walk allocated nothing of its own: device free %llu -> %llu MiB\n",
                    static_cast<unsigned long long>(free_before / 1024 / 1024),
                    static_cast<unsigned long long>(free_after / 1024 / 1024));
                require(free_after == free_before,
                        "a step of the walk took device memory of its own");

                // Determinism: the same inputs, walked again, must publish the same bytes.
                std::printf("determinism: the same inputs, walked again\n");
                auto second = walk_and_draft(probe, work, arena, false, true);
                require(second.complete, "the second walk did not complete");
                require(second.sampled == first.sampled,
                        "the sampled arm is not deterministic on the device");
                require(second.greedy == first.greedy,
                        "the T=0 arm is not deterministic on the device");

                // The byte-level form of the same row: two rounds of identical inputs, each behind
                // its own round reset, digest identically over every stream the round owns. The
                // ids alone are not the identity - a round can publish the same fifteen ids out of
                // a score table that moved.
                auto again = walk_and_draft(probe, work, arena, false, true);
                require(again.complete, "the third walk did not complete");
                require(again.sampled == second.sampled,
                        "the sampled arm is not deterministic between two publishing rounds");
                require(again.greedy == second.greedy,
                        "the T=0 arm is not deterministic between two publishing rounds");
                if (again.bytes.has_value() && second.bytes.has_value()) {
                    const auto parted = again.bytes->parted_against(*second.bytes);
                    std::printf("        two rounds of identical inputs: %zu of %zu byte streams "
                                "part\n",
                                parted.size(), again.bytes->parts.size());
                    for (std::size_t index = 0; index < std::min<std::size_t>(8, parted.size());
                         ++index) {
                        std::printf("          parts: %s\n", parted[index].c_str());
                    }
                    require(parted.empty(),
                            "two rounds of identical inputs did not produce identical bytes");
                    std::fflush(stdout);
                }

                if (const auto path = setting("NINFER_TEST_DUMP"); !path.empty()) {
                    std::printf("reference dump: %s\n", path.c_str());
                    Dumper dumper{std::fopen(path.c_str(), "wb")};
                    if (dumper.file == nullptr) {
                        throw std::runtime_error("the reference dump cannot be opened");
                    }
                    const auto h2 = dimension(probe.target.hidden_size);
                    const auto w2 = dimension(probe.config.block_size);
                    const auto columns2 = std::int32_t(w2 * kBatch);
                    dumper.put("residual_pre", *bridge_snapshot, "u2");
                    dumper.put("normalized", work.normalized, "u2");
                    dumper.put("rms", work.rms, "f4");
                    dumper.put("codes_pre", *codes_snapshot, "u2");
                    dumper.put("codes", work.codes, "u2");
                    dumper.put("packed", work.packed, "u2");
                    dumper.put("message", work.message, "u2");
                    dumper.put("delta", work.delta, "u2");
                    dumper.put("raw", work.stash, "u2");
                    dumper.put("normed", work.normed, "u2");
                    dumper.put("residual_post", work.residual, "u2");
                    // The norm vectors are weights, not probe buffers: copy them into probe
                    // storage so the readback goes through the same device boundary.
                    auto host_copy = [&](const Tensor& source) {
                        auto slot = probe.buffer(source.bytes(), source.dtype == DType::BF16
                                                      ? DType::BF16
                                                      : source.dtype,
                                                  {source.ne[0], source.ne[1], source.ne[2],
                                                   source.ne[3]});
                        cuda_ok(cudaMemcpy(slot.data, source.data, source.bytes(),
                                            cudaMemcpyDeviceToDevice),
                                "copy a weight vector into the probe's own storage");
                        return slot;
                    };
                    dumper.put("final_norm",
                               host_copy(probe.parameter(probe.weights.final_norm)), "u2");
                    dumper.put("gate_bias",
                               host_copy(probe.parameter(probe.weights.bridge.gate_bias)), "u2");
                    dumper.put("table", work.table, "f4");
                    dumper.put("tail_cand", work.tail_cand, "i4");
                    dumper.put("anchors", work.anchors, "i4");
                    dumper.put("idx", work.idx_scores, "i4");
                    dumper.put_i32("drafts_sampled", first.sampled);
                    dumper.put_i32("drafts_greedy", first.greedy);
                    dumper.put_i32("shape", {h2,
                                             w2,
                                             kBatch,
                                             columns2,
                                             dimension(probe.config.bridge.code_size),
                                             dimension(probe.config.bridge.rank),
                                             dimension(probe.config.target_layer_ids.size()),
                                             dimension(probe.config.head_dim),
                                             dimension(probe.config.num_attention_heads),
                                             dimension(probe.config.num_key_value_heads)});
                    dumper.put_value("bridge_eps",
                                     double(probe.config.bridge.eps));
                    dumper.put_value("rms_norm_eps",
                                     double(probe.config.rms_norm_eps));
                    std::fclose(dumper.file);
                }

                // Teeth: rotate the target-layer axis of the taps (a wrong tap index) and the
                // drafts must move. A walk that ignores the taps, or a comparison that cannot
                // see them, fails here.
                std::printf("teeth: the taps' target-layer axis rotated by one (a wrong tap)\n");
                auto mutated = walk_and_draft(probe, work, arena, true);
                require(mutated.complete, "the rotated-tap arm did not complete");
                require(mutated.sampled != first.sampled || mutated.greedy != first.greedy,
                        "a wrong tap index did not change the drafts: the walk cannot see the "
                        "taps");

                // The T=0 arm is the greedy degenerate of the sampled op: it must equal the
                // greedy pair (dspine_serial_scores' idx walked by dspine_serial_chain) over the
                // SAME round. The pair ran inside each round, over that round's own idx; the
                // streams the probe holds now are the last round's, so a pair launched out here
                // would walk the teeth round's idx against the first round's drafts (D-39's
                // "15 real ids vs the tail's ids"). The first pass of a process publishes nothing
                // (the module-load pass), so the verdict is taken on the passes that published.
                const auto slots = std::int32_t(work.candidates.ne[1]);
                std::printf("T=0 against the greedy pair, per round, on each round's own idx\n");
                const auto at = [&](const std::vector<std::int32_t>& v, std::int32_t batch,
                                     std::int32_t slot, const bool batch_first) {
                    return v[std::size_t(batch_first
                                             ? batch + std::int64_t(kBatch) * slot
                                             : slot + std::int64_t(slots) * batch)];
                };
                const auto judge = [&](const Round& round, const std::string& name) {
                    if (not round.complete) { return; }
                    std::printf("  %s\n", name.c_str());
                    require(round.misplaced_cand == 0,
                            "the tail's candidate stream is not the head's candidate set in the "
                            "tail's layout");
                    require(round.table_flips == 0,
                            "the argmax of the round's score table is not the idx serial_scores "
                            "published for it");
                    // serial_chain publishes [drafts, batch], serial_tail [batch, drafts]: the
                    // two declared layouts are transposed, so the identity is read on the
                    // (batch, slot) pairs they name, not on the flat bytes.
                    std::int32_t differ = 0, pair_zero = 0;
                    for (std::int32_t batch = 0; batch < kBatch; ++batch) {
                        for (std::int32_t slot = 0; slot < slots; ++slot) {
                            differ += at(round.chain, batch, slot, false)
                                   != at(round.greedy, batch, slot, true);
                            pair_zero += at(round.greedy, batch, slot, true) == 0;
                        }
                    }
                    if (pair_zero == std::int32_t(kBatch) * slots) {
                        std::printf("        the round published no drafts (the module-load "
                                    "pass): the identity is taken on a publishing pass\n");
                        std::fflush(stdout);
                        return;
                    }
                    require(differ == 0,
                            "the T=0 tail did not equal the greedy pair over the same idx");
                    // The sparse acceptor's own contract, on both arms: every published draft is
                    // one of that slot's 16 candidate ids (serial_tail_accept_q_error). A tail
                    // handed the wrong candidate table draws real ids of the wrong slot, which is
                    // exactly what this refuses.
                    const auto member_check = [&](const char* label,
                                                  const std::vector<std::int32_t>& drafts) {
                        std::int32_t foreign = 0;
                        for (std::int32_t batch = 0; batch < kBatch; ++batch) {
                            for (std::int32_t slot = 0; slot < slots; ++slot) {
                                const auto drawn  = at(drafts, batch, slot, true);
                                bool member       = false;
                                for (std::int32_t cand = 0; cand < kSlots; ++cand) {
                                    member = member || round.head[std::size_t(
                                        cand + std::int64_t(kSlots)
                                             * (slot + std::int64_t(slots) * batch))]
                                          == drawn;
                                }
                                foreign += not member;
                            }
                        }
                        std::printf("        %s: %d of %d drafts are not one of their slot's "
                                    "16 candidates\n",
                                    label, int(foreign), std::int32_t(kBatch) * slots);
                        std::fflush(stdout);
                        require(foreign == 0,
                                "a published draft is not one of its slot's 16 candidate ids");
                    };
                    member_check("the T=0 arm  ", round.greedy);
                    member_check("the T=0.9 arm", round.sampled);
                };
                judge(second, "the second pass (the first that publishes)");
                judge(mutated, "the rotated-tap pass (the teeth round)");

                // The multi-round rows: N rounds of identical inputs behind the round reset, the
                // per-round rates, and the statement of what the harness cannot measure.
                measure_rounds(probe, work, arena);
            }
        }
    } catch (const std::exception& error) {
        std::printf("FATAL: %s\n", error.what());
        return 1;
    }
    std::printf("dspine forward probe: %zu steps walked, %zu gates, %zu device uploads, "
                "%zu/%zu checks green\n",
                counters.steps, counters.gates, counters.uploads, counters.checks,
                counters.checks - counters.failures);
    return counters.failures == 0 ? 0 : 1;
}
