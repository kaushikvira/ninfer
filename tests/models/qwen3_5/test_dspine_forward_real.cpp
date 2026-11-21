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
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/sequence.h"
#include "ninfer/ops/softmax_attention.h"
#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "core/paged_kv_geometry.h"
#include "../../../tools/support/paged_kv_fixture.h"

#include <cuda_runtime.h>

#include <bit>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <execinfo.h>
#include <exception>
#include <memory>
#include <set>
#include <string>
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
        context_key, context_value, proposal;
    Tensor candidates, scores, table, tail_cand, drafts, proposal_q, index, idx_scores,
        positions_rng, configs;
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
    const auto policy = ops::LinearPolicy::A16Only;
    const auto at = [&](const WeightId id, const std::size_t rows, const std::size_t tokens) {
        const auto bound = probe.declared(id);
        return ops::linear_workspace_capacity_bytes(bound.qtype, std::int32_t(rows), bound.k,
                                                    policy, std::int32_t(tokens),
                                                    std::int32_t(tokens));
    };
    const auto packed_at = [&](const WeightId id, const std::size_t tokens) {
        const auto bound = probe.declared(id);
        const ops::ProjectionProfile profile{
            ops::LinearProfile{bound.qtype, std::int32_t(probe.packed_rows(id)), bound.k, policy,
                               QuantLayout::Contiguous},
            {}};
        return ops::linear_split_workspace_capacity_bytes(
            profile,
            std::vector<std::int32_t>{dimension(probe.config.query_width()),
                                      dimension(probe.config.key_width()),
                                      dimension(probe.config.key_width())},
            std::int32_t(tokens), std::int32_t(tokens));
    };
    std::size_t bytes = 0;
    const auto every = [&](const std::size_t one) { bytes = std::max(bytes, one); };
    every(at(probe.weights.feature_projection, std::size_t(h) * taps, kContext));
    every(at(probe.weights.layers.front().attention.output, h, kContext));
    every(at(probe.weights.layers.front().mlp.down, h, kContext));
    every(at(probe.weights.bridge.gate, r, kContext));
    every(at(probe.weights.bridge.read_weight, c, kContext));
    every(at(probe.weights.bridge.feature_write_weights, h, kContext));
    every(at(probe.weights.layers.front().attention.context_key,
              std::size_t(probe.config.key_width()), kContext));
    const auto head_bound = probe.declared(probe.head);
    every(ops::linear_topk_workspace_capacity_bytes(head_bound.qtype, head, h, kContext - 1,
                                                    kContext - 1));
    every(ops::linear_swiglu_workspace_capacity_bytes(head_bound.qtype, inter, h, policy,
                                                      kContext, kContext) * 0 +
          ops::linear_swiglu_workspace_capacity_bytes(
              probe.declared(probe.weights.layers.front().mlp.gate).qtype,
              std::int32_t(probe.packed_rows(probe.weights.layers.front().mlp.gate)), h, policy,
              kContext, kContext));
    const auto attention = ops::context_softmax_attention_workspace_capacity_bytes(
        ops::AttentionHeadGeometry{dimension(probe.config.head_dim),
                                   dimension(probe.config.num_attention_heads),
                                   dimension(probe.config.num_key_value_heads)},
        ops::ContextAttentionExecutionEnvelope{std::uint32_t(kContext), std::uint32_t(kContext)},
        kContext, kContext, kBatch);
    every(attention);
    (void)packed_at;
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
    out.context_value     = stream(std::int64_t(k) * ctx * b, {d, k / d, ctx, b});
    out.candidates        = integers(std::int64_t(kSlots) * slots * b, {kSlots, slots, b, 1});
    out.scores            = reals(std::int64_t(kSlots) * slots * b, {kSlots, slots, b, 1});
    out.idx_scores        = integers(std::int64_t(kSlots) * slots * b, {kSlots, slots, b, 1});
    out.table             = reals(std::int64_t(b) * kSlots * kSlots * slots,
                                  {b, kSlots, kSlots, slots});
    out.tail_cand         = integers(std::int64_t(b) * slots * kSlots, {b, slots, kSlots, 1});
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
    step("bridge read: rms of the layer stream" + suffix, [&] {
        ops::dspine_prepare_read(flat_residual, probe.config.bridge.eps, flat_normalized,
                                  flat_rms, cuda.stream());
    });
    step("bridge code projection" + suffix, [&] {
        const auto read =
            last ? probe.weight(bridge.read_weight)
                 : probe.stacked(bridge.early_read_weights, layer, std::uint64_t(c),
                                 std::uint64_t(h));
        ops::linear(flat_normalized, read, flat_codes, ops::LinearPolicy::A16Only, arena, cuda);
    });
    step("bridge normalize_codes over the real codebook" + suffix, [&] {
        ops::dspine_normalize_codes(flat_codes, work.anchors, probe.weight(bridge.codebook), w,
                                    probe.config.bridge.eps, flat_codes, cuda.stream());
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

// The head writes [candidate, slot*batch]; the tail reads [batch, slot, candidate]. The engine
// bridges the two at plan time; here the bridge is a host transpose of 240 words.
void seed_tail_inputs(Work& work) {
    const auto slots = work.candidates.ne[1];
    const auto from  = download_i32(work.candidates);
    std::vector<std::int32_t> to(std::size_t(std::int64_t(kBatch) * slots * kSlots));
    for (std::int32_t slot = 0; slot < slots; ++slot) {
        for (std::int32_t cand = 0; cand < kSlots; ++cand) {
            to[std::size_t(std::int64_t(cand) * kBatch * slots + std::int64_t(slot) * kBatch)] =
                from[std::size_t(cand + std::int64_t(slot) * kSlots)];
        }
    }
    upload(work.tail_cand.data, to.data(), to.size() * sizeof(std::int32_t), "tail candidates");
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

    std::vector<std::int32_t> anchors(std::size_t(b), 0), lengths(std::size_t(b), 0),
        valid(std::size_t(b), 0), context_lengths(std::size_t(b), 0),
        context_counts(std::size_t(b), 0), table_rows(std::size_t(b), 0),
        context_positions(std::size_t(std::int64_t(ctx) * b), 0),
        block_positions(std::size_t(std::int64_t(w) * b), 0);
    for (std::int32_t index = 0; index < b; ++index) {
        anchors[index]         = 40901 + index * 7;
        lengths[index]         = ctx;  // the block starts where the context ends
        valid[index]           = columns;
        context_lengths[index] = ctx;
        context_counts[index]  = ctx;
        table_rows[index]      = 0;
    }
    for (std::int32_t index = 0; index < ctx * b; ++index) {
        context_positions[std::size_t(index)] = index;
    }
    for (std::int32_t index = 0; index < w * b; ++index) {
        block_positions[std::size_t(index)] = ctx + index;
    }
    upload_i32(work.anchors, anchors, "anchors");
    upload_i32(work.lengths, lengths, "block lengths");
    upload_i32(work.valid, valid, "valid columns");
    upload_i32(work.context_lengths, context_lengths, "context lengths");
    upload_i32(work.context_counts, context_counts, "context counts");
    upload_i32(work.table_rows, table_rows, "page table rows");
    upload_i32(work.context_positions, context_positions, "context positions");
    upload_i32(work.positions, block_positions, "block positions");
    seed(work.taps);

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
                 auto context  = work.context.view({h, flat, 1, 1});
                 auto key_out  = work.context_key.view({k, flat, 1, 1});
                 auto value_out = work.context_value.view({k, flat, 1, 1});
                 ops::linear(context, probe.weight(block.attention.context_key), key_out,
                             ops::LinearPolicy::A16Only, arena, cuda);
                 ops::linear(context, probe.weight(block.attention.context_value), value_out,
                             ops::LinearPolicy::A16Only, arena, cuda);
                 ops::rmsnorm(work.context_key, probe.parameter(block.attention.key_norm),
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
            const std::array outputs{work.query.view({q, columns, 1, 1}),
                                     work.key.view({k, columns, 1, 1}),
                                     work.value.view({k, columns, 1, 1})};
            ops::linear_split(work.hidden.view({h, columns, 1, 1}),
                              probe.packed_parent(block.attention.query), outputs, cuda);
        });
        step("query and key head norms, then RoPE" + suffix, [&] {
            ops::rmsnorm(work.query, probe.parameter(block.attention.query_norm),
                         probe.config.rms_norm_eps, false, work.query, cuda);
            ops::rmsnorm(work.key, probe.parameter(block.attention.key_norm),
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
            ops::linear_add(attended, probe.weight(block.attention.output), residual,
                            ops::LinearPolicy::A16Only, arena, cuda);
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
        cuda_ok(cudaMemcpy2DAsync(work.proposal.data, std::size_t(std::int64_t(h) * slots) * 2,
                                  static_cast<std::byte*>(work.normed.data) + std::size_t(h) * 2,
                                  std::size_t(std::int64_t(h) * columns) * 2,
                                  std::size_t(std::int64_t(h) * slots) * 2, std::int64_t(slots * b),
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
         [&] { seed_tail_inputs(work); });
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

// One full pass: the block walk, the head, the score table, then the tail at both policies.
struct Round {
    std::vector<std::int32_t> sampled, greedy;
    bool complete = false;
};

[[nodiscard]] Round walk_and_draft(Probe& probe, Work& work, MemoryArena& arena,
                                   const bool rotate_taps) {
    if (rotate_taps) { seed_taps_rotated(work.taps); } else { seed(work.taps); }
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
    return out;
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
            const auto bytes = walk_workspace_bytes(probe);
            auto arena_storage = DeviceBuffer(probe.device, bytes);
            MemoryArena arena(arena_storage.span());
            auto work = allocate(probe, bytes);
            const auto complete = [&] { return counters.gates == 0; };

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
                auto second = walk_and_draft(probe, work, arena, false);
                require(second.complete && second.sampled == first.sampled &&
                            second.greedy == first.greedy,
                        "the walk is not deterministic on the device");

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
                // greedy pair (dspine_serial_scores' idx walked by dspine_serial_chain).
                // The greedy pair's own two ops over the walk's head rows and the score table's
                // argmax twin (serial_scores' idx): the chain this T=0 arm must reproduce exactly.
                std::printf("T=0 against the greedy pair over this round's own idx\n");
                const auto slots = std::int32_t(work.candidates.ne[1]);
                auto greedy = probe.buffer(std::size_t(4) * slots * kBatch, DType::I32,
                                           {slots, kBatch, 1, 1});
                step("the greedy pair: dspine_serial_chain over the head's rows and the "
                     "serial_scores idx",
                     [&] {
                         auto chain_cand = work.candidates.view({kSlots, slots, kBatch, 1});
                         auto chain_idx  = work.idx_scores.view({kSlots, slots, kBatch, 1});
                         ops::dspine_serial_chain(chain_cand, chain_idx, greedy,
                                                  probe.cuda().stream());
                         cuda_ok(cudaGetLastError(), "serial chain launch");
                     });
                if (counters.gates == 0) {
                    require(download_i32(greedy) == first.greedy,
                            "the T=0 tail did not equal the greedy pair over the same idx");
                }
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
