// P2 stage 3b: the DSpine draft forward's linear gate, read out of the compiled binary against the
// real V3 drafter artifact.
//
// This is the shape half of the forward probe (wt-dspine-fwd @ 59830140,
// tests/models/qwen3_5/test_dspine_forward_real.cpp) - copied, not edited there - with the load
// half dropped: binding the drafter is a manifest read, and every linear family resolves
// (representation, rows, columns, token interval) against a registered shape table on the host,
// before any launch. So the gate is measured at the geometries the artifact really declares, at T
// of one token (a context append) and of a whole draft block, and nothing is allocated.
//
// A refused geometry is a proposal step that cannot be issued. This probe exits 1 when any of the
// nine is refused; the three the proposal path was blocked on are the feature projection, the
// packed attention query/key/value parent and the attention output.
//
// The artifact is the one D-14/D-20 sized: …_dspine-nvfp4tables.ninfer.
//
// NINFER_TEST_ARTIFACT   the artifact (skip, exit 77, when unset)
// NINFER_TEST_DEVICE     the visible CUDA device ordinal to run on. Required, with no default: a
//                        device probe never guesses a card (D-29), and the card it landed on is
//                        printed (ordinal, name, PCI address) as part of the evidence.
//
// Run it pinned to the free card: CUDA_VISIBLE_DEVICES=1 NINFER_TEST_DEVICE=0 …

#include "artifact/binder.h"
#include "artifact/formats.h"
#include "artifact/reader.h"
#include "artifact/schema.h"
#include "core/device.h"
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
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::models;
namespace dspine = ninfer::models::dspine;

constexpr std::int32_t kSlots = 16;

std::string setting(std::string_view name, std::string_view fallback = {}) {
    const char* value = std::getenv(std::string(name).c_str());
    return value && *value ? std::string(value) : std::string(fallback);
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
        cudaSetDevice(ordinal);
        cudaDeviceProp properties{};
        cudaGetDeviceProperties(&properties, ordinal);
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

    // The whole packed object a split projection is a row range of: what the fused linear family
    // would have to compile, as one geometry.
    [[nodiscard]] std::int32_t packed_rows(const WeightId id) const {
        const auto& parts = bindings->at(id).reference.binding.parts;
        return std::int32_t(reader.directory().tensor(parts.front().object).shape[0]);
    }

    artifact::Reader reader;
    Device device;
    DeviceContext context;
    dspine::TargetGeometry target{};
    dspine::Config config;
    std::unique_ptr<artifact::Binder> binder;
    std::unique_ptr<Bindings> bindings;
    dspine::Weights weights;
};

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
    };
    for (const std::int32_t tokens : {1, kSlots}) {
        for (const auto& request : requests) {
            const auto bound   = probe.declared(request.parameter);
            const auto rows    = request.rows != 0 ? request.rows : bound.n;
            const auto label   = request.label + " [" + std::to_string(rows) + ", " +
                                 std::to_string(bound.k) + "] " +
                                 std::string(artifact::format_name(bound.qtype));
            const auto policy  = ops::LinearPolicy::A16Only;
            const auto attempt = [&]() -> std::size_t {
                if (request.family == "linear") {
                    return ops::linear_workspace_capacity_bytes(bound.qtype, rows, bound.k, policy,
                                                                tokens, tokens);
                }
                if (request.family == "linear_add") {
                    return ops::linear_add_workspace_capacity_bytes(bound.qtype, rows, bound.k,
                                                                    policy, tokens, tokens);
                }
                if (request.family == "linear_swiglu") {
                    return ops::linear_swiglu_workspace_capacity_bytes(bound.qtype, rows, bound.k,
                                                                       policy, tokens, tokens);
                }
                const ops::ProjectionProfile profile{
                    ops::LinearProfile{
                        bound.qtype, rows, bound.k, policy, QuantLayout::Contiguous},
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
    try {
        Probe probe(artifact_path, device);
        probe.bind();
        std::printf("the linear geometries the proposal path names:\n");
        report_shapes(probe);
    } catch (const std::exception& error) {
        std::printf("FATAL: %s\n", error.what());
        return 1;
    }
    std::printf("dspine shape probe: %zu lookups, %zu refused by the engine\n", counters.steps,
                counters.gates);
    return counters.gates == 0 ? 0 : 1;
}
