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
// NINFER_TEST_STAGE      load | shapes | all (default all)
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
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_split.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/linear_topk.h"
#include "ninfer/ops/projection_weights.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    std::size_t steps = 0;
    std::size_t gates = 0;
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
    (void)what;
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

} // namespace

int main(const int argc, char** argv) {
    (void)argc;
    (void)argv;
    // A probe that dies mid-stage must not lose the lines it already measured.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
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
    } catch (const std::exception& error) {
        std::printf("FATAL: %s\n", error.what());
        return 1;
    }
    std::printf("dspine forward probe: %zu steps walked, %zu gated by the engine\n",
                counters.steps, counters.gates);
    return 0;
}
