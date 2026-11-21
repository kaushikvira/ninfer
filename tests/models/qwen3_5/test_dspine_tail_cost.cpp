// DSpine proposal-tail COST curve (D-43's perf question: criterion ⑤ is accepted tokens per unit
// TIME, and D-40 put the bar at DFlash2's measured 2.998 accepted drafts/round).
//
// THE QUESTION. DSpine drafts K=15 and DFlash2 K=7. The plan reads the tail's score table as a
// K-by-K grid of head projections over the vocabulary - "K^2 x 248320 x 5120 matvec-batches per
// round", 49 batches at K=7 against 225 at K=15, i.e. 4.6x the wall clock for at most 2x the
// accepted tokens. If that scaling is real the build cannot win on tok/s at any acceptance. This
// probe measures the cost per round of the real proposal-tail op sequence at K = 1, 3, 7, 15, in
// CUDA-event milliseconds, on the REAL drafter head and the REAL tg/tsv tables of the V3 artifact,
// and reports the measured scaling exponent against the K^2 theory.
//
// THE SKELETON IS NOT INVENTED HERE. Every line of device bring-up - the `Probe` struct, its
// `Device`/`DeviceContext`/`CudaDeviceScope` use, the artifact `Reader` open, the materialization
// set, the `buffer` allocation and the `upload`/`download` boundary - is copied from the probe that
// RUNS this artifact (tests/models/qwen3_5/test_dspine_forward_real.cpp, #71: manifest 0.29 s, 55
// device objects, 3.83186 GiB, then 3514 walk steps). D-45 recorded a sibling probe (#70) that
// stops before its first printed line, and the LOCKED instruction for this measurement is that a
// verbatim copy of the working path is the only accepted setup. What is added on top of it is the
// tail's streams at a given K and the events around the tail's launches - nothing else.
//
// WHAT IS MEASURED (every number is a launch of engine code; nothing is modelled)
//   topk    ops::linear_topk - the tower head's top-16 per proposal slot over 248077 valid rows.
//           One launch at K columns: the `slots x K` head-projection grid startup.cpp:399-402
//           sizes, i.e. the K term of D-43's reading.
//   scores  ops::dspine_serial_scores - the K-by-K score table: one CTA per (batch, proposal row),
//           so `drafts` CTAs, each carrying the 16 head-row gathers over H and the 16x16 terminal
//           dots over the bridge rank. THE K^2 GRID LIVES INSIDE THIS ONE LAUNCH; the head-row
//           gathers and the terminal dots are the same kernel, so they cannot be timed apart
//           without editing the kernel - and the locked instruction says measure, not optimise.
//   tail    ops::dspine_serial_tail - the policy filter and the drawn chain at the production
//           policy (T=0.9, top_k 20, top_p 1.0, min_p 0.05), one CTA, batch-wide.
//   chain   the three above as the round runs them, in order, on one stream.
//   bridge  execution/dspine_propose_tail step 4: the head-to-tail candidate layout crossing, which
//           is two stream-settled host crossings and a relabel of 16*K words (execution/dspine.cpp
//           :196-203). Timed on the wall clock, because it settles the stream.
//   df2     the SAME op (linear_topk) at the DFlash2 proposal geometry, the (K+1)-wide masked block
//           (2, 4, 8, 16 columns): the K term of the other arm's head cost at the same head. It is
//           NOT DFlash2's whole round - its draft layers and selector need a DFlash2 drafter and a
//           lane - and the line says so.
//
// WHAT IT IS NOT. No acceptance number (D-42: that needs the target's logits, i.e. the a5000-2 lane
// arm), no numerics claim (the activation streams are synthetic carriers, the weights are the
// artifact's), no optimisation: no kernel, tolerance, weight copy, shape gate or allocation was
// touched. The candidate ids are spread over the token domain, so the head/tg/tsv row gathers are
// cold rows, which is the shape of a real round and not its content.
//
// NINFER_TEST_ARTIFACT  the V3-tailored DSpine artifact (skip, exit 77, when unset)
// NINFER_TEST_DEVICE    the visible CUDA ordinal to run on; required, no default (D-29)
// NINFER_BENCH_DRAFTS   comma list of K to cost (default "1,3,7,15")
// NINFER_BENCH_ITERS    timed iterations per arm per K (default 24, floor 20)
// NINFER_BENCH_WARMUP   un-timed iterations that pay the module load (default 6)
// NINFER_BENCH_DF2      "0" drops the DFlash2-geometry linear_topk arm
// NINFER_BENCH_BAR      the bar, accepted drafts/round (default 2.998, D-40)
// NINFER_BENCH_ROUND_MS prod's measured ms/round (default 19.137 = the batch-1
//                       ninfer_decode_step_seconds_sum/count read off prod's /metrics)
//
// Run it pinned to the free card: CUDA_VISIBLE_DEVICES=1 NINFER_TEST_DEVICE=0 ...

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/placement.h"
#include "artifact/reader.h"
#include "artifact/views.h"
#include "core/arena.h"
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
#include "ninfer/ops/linear_topk.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::models;
namespace dspine = ninfer::models::dspine;

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
constexpr std::int32_t kSlots = 16;
// One production proposal round: batch 1, and the head's valid row domain (execution/dspine.cpp:32).
constexpr std::int32_t kBatch = 1;
constexpr std::int32_t kTokenDomain = 248077;

// A harness that dies mid-stage must not lose the lines it already measured, and a device harness
// that dies in a driver call must name the call. Copied from the working probe, which installed it
// for the same reason (D-34's fault, D-45's stop).
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
        std::fflush(stdout);
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
    // representation the artifact declares for it.
    struct Declared {
        std::int32_t n, k;
        QType qtype;
    };

    [[nodiscard]] Declared declared(const WeightId id) const {
        const auto& shape = bindings->at(id).reference.shape;
        return Declared{std::int32_t(shape[shape.size() - 2]),
                        std::int32_t(shape[shape.size() - 1]),
                        binder->declared_format(bindings->at(id).reference.name)};
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
// host and therefore on the device. The tail's activations are synthetic carriers by design; its
// weights are the artifact's, which is what makes the cost a measurement.
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

// ---------------------------------------------------------------- the round at one K -----------------

// The proposal tail's streams at K drafts, at the shapes `dspine_propose_tail` names them by
// (src/models/qwen3_5/execution/dspine.cpp:164-205) and the ones the ops validate
// (src/ops/dspine_bridge/dspine_bridge.cpp:127-206). width = drafts + 1: column 0 is the anchor.
struct Tail {
    std::int32_t drafts = 0;
    // The last bridge's stash: the pre-refinement residual, its rms, and the message as the Q.
    Tensor stash, rms, message, anchors;
    // 1-3. the gathered proposal columns, the head's top-16 grid, serial_scores' chain and table.
    Tensor proposal, candidates, scores, index, table;
    // 4. the tail's own layouts, and its outputs.
    Tensor tail_cand, tail_drafts, tail_q, tail_index, positions, configs;
};

[[nodiscard]] Tail allocate(Probe& probe, const std::int32_t drafts) {
    const auto h   = dimension(probe.target.hidden_size);
    const auto r   = dimension(probe.config.bridge.rank);
    const auto w   = std::int32_t(drafts + 1);
    const auto b   = kBatch;
    Tail out;
    out.drafts = drafts;
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
    out.stash        = stream(std::int64_t(h) * w * b, {h, w, b, 1});
    out.rms          = reals(std::int64_t(w) * b, {w, b, 1, 1});
    out.message      = stream(std::int64_t(r) * w * b, {r, w, b, 1});
    out.anchors      = integers(b, {b, 1, 1, 1});
    out.proposal     = stream(std::int64_t(h) * drafts * b, {h, drafts, b, 1});
    out.candidates   = integers(std::int64_t(kSlots) * drafts * b, {kSlots, drafts, b, 1});
    out.scores       = reals(std::int64_t(kSlots) * drafts * b, {kSlots, drafts, b, 1});
    out.index        = integers(std::int64_t(kSlots) * drafts * b, {kSlots, drafts, b, 1});
    out.table        = reals(std::int64_t(b) * kSlots * kSlots * drafts,
                              {b, kSlots, kSlots, drafts});
    out.tail_cand    = integers(std::int64_t(b) * drafts * kSlots, {b, drafts, kSlots, 1});
    out.tail_drafts  = integers(std::int64_t(b) * drafts, {b, drafts, 1, 1});
    out.tail_q       = reals(std::int64_t(b) * drafts * kSlots, {b, drafts, kSlots, 1});
    out.tail_index   = integers(std::int64_t(b) * drafts * kSlots, {b, drafts, kSlots, 1});
    out.positions    = integers(b, {b, 1, 1, 1});
    out.configs      = probe.buffer(sizeof(ops::SamplingConfig) * std::size_t(b), DType::FP32,
                                     {b, 1, 1, 1});
    return out;
}

// A deterministic id spread over the token domain: the head-, tg- and tsv-row gathers land on
// cold rows of a 248320-row table, the shape of a real round's candidate set.
[[nodiscard]] std::int32_t candidate_id(const std::int32_t slot, const std::int32_t cand) {
    const auto mixed = std::int64_t(slot) * 16 + cand;
    return std::int32_t((mixed * 1013 + 7) % kTokenDomain);
}

void seed_tail(Probe& probe, Tail& tail, const Weight& head, const Weight& tg,
               const Weight& tsv) {
    seed(tail.stash);
    seed(tail.message);
    seed(tail.proposal);
    {
        std::vector<float> values(std::size_t(tail.rms.numel()));
        for (std::size_t index = 0; index < values.size(); ++index) {
            values[index] = 1.0f + 0.25f * std::sin(0.31f * float(index) + 0.7f);
        }
        upload(tail.rms.data, values.data(), values.size() * sizeof(float), "rms");
    }
    upload_i32(tail.anchors, std::vector<std::int32_t>(std::size_t(kBatch), 1024), "anchors");
    {
        const auto drafts = tail.drafts;
        // The head's layout (candidate, slot, batch) and the tail's (batch, slot, candidate) are
        // the same grid with its axes exchanged (execution/dspine.cpp:69-88, D-40's RED 2).
        const auto at_head = [&](const std::int32_t cand, const std::int32_t slot,
                                  const std::int32_t batch) {
            return std::size_t(std::int64_t(cand)
                               + std::int64_t(kSlots) * (slot + std::int64_t(drafts) * batch));
        };
        const auto at_tail = [&](const std::int32_t batch, const std::int32_t slot,
                                  const std::int32_t cand) {
            return std::size_t(std::int64_t(batch)
                               + std::int64_t(kBatch) * (slot + std::int64_t(drafts) * cand));
        };
        std::vector<std::int32_t> ids(std::size_t(std::int64_t(kSlots) * drafts * kBatch));
        for (std::int32_t batch = 0; batch < kBatch; ++batch) {
            for (std::int32_t slot = 0; slot < drafts; ++slot) {
                for (std::int32_t cand = 0; cand < kSlots; ++cand) {
                    ids[at_head(cand, slot, batch)] = candidate_id(slot, cand);
                }
            }
        }
        upload_i32(tail.candidates, ids, "candidates");
        // The tail's candidate layout is the head's grid with its axes exchanged; the engine's
        // step-4 bridge does this relabel every round. Doing it once here, outside the timed
        // region, is what lets the tail arm be timed alone - the bridge itself is its own arm.
        std::vector<std::int32_t> crossed(std::size_t(std::int64_t(kBatch) * drafts * kSlots));
        for (std::int32_t batch = 0; batch < kBatch; ++batch) {
            for (std::int32_t slot = 0; slot < drafts; ++slot) {
                for (std::int32_t cand = 0; cand < kSlots; ++cand) {
                    crossed[at_tail(batch, slot, cand)] = ids[at_head(cand, slot, batch)];
                }
            }
        }
        upload_i32(tail.tail_cand, crossed, "tail candidates");
    }
    // The production policy: the server's --temperature 0.9 with the tail's own top_k/top_p/min_p.
    std::vector<ops::SamplingConfig> configs(std::size_t(kBatch), ops::SamplingConfig{});
    for (auto& config : configs) {
        config.temperature = 0.9f;
        config.top_k       = 20;
        config.top_p       = 1.0f;
        config.min_p       = 0.05f;
        config.seed        = 0x50371153ULL;
    }
    upload(tail.configs.data, configs.data(), configs.size() * sizeof(ops::SamplingConfig),
           "tail configs");
    std::vector<std::int32_t> positions;
    positions.resize(std::size_t(kBatch));
    for (std::int32_t index = 0; index < kBatch; ++index) {
        positions[std::size_t(index)] = 1024 + index;
    }
    upload_i32(tail.positions, positions, "tail draw positions");
    (void)head;
    (void)tg;
    (void)tsv;
}

// ---------------------------------------------------------------- the timed arms ---------------------

// The proposal columns out of the (K+1)-wide block: the same 2-D strided gather the engine issues
// (execution/dspine.cpp:152-163).
void gather_proposals(Probe& probe, Tail& tail) {
    const auto h    = dimension(probe.target.hidden_size);
    const auto w    = std::int32_t(tail.drafts + 1);
    const auto cuda = probe.cuda();
    const auto row    = std::size_t(std::int64_t(h) * tail.drafts) * 2;
    const auto spitch = std::size_t(std::int64_t(h) * w) * 2;
    const auto source = static_cast<std::byte*>(tail.stash.data) + std::size_t(h) * 2;
    cuda_ok(cudaMemcpy2DAsync(tail.proposal.data, row, source, spitch, row, std::size_t(kBatch),
                              cudaMemcpyDeviceToDevice, cuda.stream()),
            "gather the proposal columns");
}

void run_topk(Probe& probe, Tail& tail, const Weight& head, MemoryArena& arena) {
    const auto cuda = probe.cuda();
    ops::linear_topk(tail.proposal, head, kTokenDomain, tail.candidates, tail.scores, arena, cuda);
    cuda_ok(cudaGetLastError(), "linear_topk launch");
}

void run_scores(Probe& probe, Tail& tail, const Weight& head, const Weight& tg,
                const Weight& tsv) {
    const auto cuda = probe.cuda();
    ops::dspine_serial_scores(tail.candidates, tail.stash, tail.rms, tail.message, tail.anchors,
                               probe.parameter(probe.weights.final_norm),
                               probe.parameter(probe.weights.bridge.gate_bias), head, tg, tsv,
                               std::int32_t(tail.drafts + 1), tail.index, &tail.table,
                               cuda.stream());
    cuda_ok(cudaGetLastError(), "serial scores launch");
}

void run_tail(Probe& probe, Tail& tail) {
    const auto cuda = probe.cuda();
    ops::dspine_serial_tail(tail.table, tail.tail_cand,
                             static_cast<const ops::SamplingConfig*>(tail.configs.data),
                             tail.positions, tail.tail_drafts, tail.tail_q, tail.tail_index,
                             cuda.stream());
    cuda_ok(cudaGetLastError(), "serial tail launch");
}

// execution/dspine_propose_tail step 4: the head's [candidate, slot, batch] grid to the tail's
// [batch, slot, candidate], over two stream-settled host crossings. Measured, not hidden: the seam
// is eager (D-45: graphs.cpp names no DSpine profile), so these settles are in the round.
double bridge_round_trip(Probe& probe, Tail& tail, std::vector<std::int32_t>& from,
                         std::vector<std::int32_t>& to) {
    const auto cuda = probe.cuda();
    const auto began = std::chrono::steady_clock::now();
    from.resize(std::size_t(std::int64_t(kBatch) * tail.drafts * kSlots));
    to.resize(from.size());
    cuda_ok(cudaMemcpyAsync(from.data(), tail.candidates.data, from.size() * sizeof(std::int32_t),
                            cudaMemcpyDeviceToHost, cuda.stream()),
            "bridge: the head's grid to the host");
    cuda_ok(cudaStreamSynchronize(cuda.stream()), "bridge: the head's grid settled");
    const auto at_head = [&](const std::int32_t cand, const std::int32_t slot,
                             const std::int32_t batch) {
        return std::size_t(std::int64_t(cand)
                            + std::int64_t(kSlots) * (slot + std::int64_t(tail.drafts) * batch));
    };
    const auto at_tail = [&](const std::int32_t batch, const std::int32_t slot,
                             const std::int32_t cand) {
        return std::size_t(std::int64_t(batch)
                            + std::int64_t(kBatch) * (slot + std::int64_t(tail.drafts) * cand));
    };
    for (std::int32_t batch = 0; batch < kBatch; ++batch) {
        for (std::int32_t slot = 0; slot < tail.drafts; ++slot) {
            for (std::int32_t cand = 0; cand < kSlots; ++cand) {
                to[at_tail(batch, slot, cand)] = from[at_head(cand, slot, batch)];
            }
        }
    }
    cuda_ok(cudaMemcpyAsync(tail.tail_cand.data, to.data(), to.size() * sizeof(std::int32_t),
                            cudaMemcpyHostToDevice, cuda.stream()),
            "bridge: the tail's grid to the device");
    cuda_ok(cudaStreamSynchronize(cuda.stream()), "bridge: the tail's grid settled");
    const auto elapsed = std::chrono::duration_cast<
        std::chrono::duration<double, std::ratio<1, 1000>>>(std::chrono::steady_clock::now()
                                                            - began);
    return elapsed.count();
}

// The round's own proof that the launches did work: the tail published a draft id in every slot,
// and the head named a candidate row for every (candidate, slot). A cost curve of ops that refused
// to run would be the fastest curve on the machine, so this rides with every K.
void publish_check(Probe& probe, Tail& tail) {
    const CudaDeviceScope settling(probe.device);
    cuda_ok(cudaDeviceSynchronize(), "the round's outputs before the publish check");
    std::vector<std::int32_t> published(std::size_t(tail.tail_drafts.numel()));
    download(published.data(), tail.tail_drafts.data,
              published.size() * sizeof(std::int32_t), "published drafts");
    std::int32_t filled = 0;
    for (const auto id : published) { filled += id != 0; }
    std::vector<std::int32_t> named(std::size_t(tail.candidates.numel()));
    download(named.data(), tail.candidates.data, named.size() * sizeof(std::int32_t),
              "head candidate rows");
    std::int32_t rows = 0;
    for (const auto id : named) { rows += id != 0; }
    std::printf("        the round published %d of %d drafts and the head named %d of %d "
                "candidate rows\n",
                int(filled), int(kBatch * tail.drafts), int(rows),
                int(kSlots * tail.drafts * kBatch));
    std::fflush(stdout);
}

// ---------------------------------------------------------------- the statistics -------------------

struct Cost {
    std::int32_t samples = 0;
    double mean = 0.0, p50 = 0.0, low = 0.0, high = 0.0;
};

[[nodiscard]] Cost describe(std::vector<double> values) {
    Cost out;
    if (values.empty()) { return out; }
    std::sort(values.begin(), values.end());
    out.samples = std::int32_t(values.size());
    double total = 0.0;
    for (const auto value : values) { total += value; }
    out.mean = total / double(values.size());
    out.p50  = values[values.size() / 2];
    out.low  = values.front();
    out.high = values.back();
    return out;
}

void report(const char* label, const std::int32_t drafts, const Cost& cost) {
    std::printf("COST %-14s K=%-2d ms/round mean %9.3f  p50 %9.3f  min %9.3f  max %9.3f  N=%d\n",
                label, drafts, cost.mean, cost.p50, cost.low, cost.high, cost.samples);
    std::fflush(stdout);
}

// log-log least squares: the measured scaling exponent of cost against K.
[[nodiscard]] double exponent(const std::vector<std::int32_t>& ks,
                              const std::vector<double>& costs) {
    double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
    const auto n = double(ks.size());
    for (std::size_t index = 0; index < ks.size(); ++index) {
        const auto x = std::log(double(ks[index]));
        const auto y = std::log(std::max(1e-9, costs[index]));
        sx += x; sy += y; sxx += x * x; sxy += x * y;
    }
    const auto den = n * sxx - sx * sx;
    return den == 0.0 ? 0.0 : (n * sxy - sx * sy) / den;
}

// CUDA-event timing over one stream. The events bracket the launches; the wall clock rides along,
// because a bridge settles its stream and then the two intervals are the same interval - a
// divergence between them is a launch gap, and that is a fact about the route, not noise.
struct Clock {
    cudaEvent_t start{}, stop{};
    std::vector<double> events;

    Clock() {
        cuda_ok(cudaEventCreate(&start, cudaEventDefault), "event create");
        cuda_ok(cudaEventCreate(&stop, cudaEventDefault), "event create");
    }
    Clock(const Clock&)            = delete;
    Clock& operator=(const Clock&) = delete;
    ~Clock() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    template <class Run>
    void operator()(cudaStream_t stream, Run&& run) {
        cuda_ok(cudaEventRecord(start, stream), "event record start");
        run();
        cuda_ok(cudaEventRecord(stop, stream), "event record stop");
        cuda_ok(cudaEventSynchronize(stop), "event synchronize");
        float milliseconds = 0.0f;
        cuda_ok(cudaEventElapsedTime(&milliseconds, start, stop), "event elapsed");
        events.push_back(milliseconds);
    }
};

} // namespace

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
    std::vector<std::int32_t> ks;
    {
        std::string list = setting("NINFER_BENCH_DRAFTS", "1,3,7,15");
        std::size_t at   = 0;
        while (at < list.size()) {
            const auto next = list.find(',', at);
            const auto part = list.substr(at, next == std::string::npos ? std::string::npos
                                                                       : next - at);
            const auto value = std::stoi(part);
            if (value >= 1 && value <= 15) { ks.push_back(std::int32_t(value)); }
            if (next == std::string::npos) { break; }
            at = next + 1;
        }
    }
    const auto asked   = std::stoi(setting("NINFER_BENCH_ITERS", "24"));
    const auto iters   = std::max(20, asked > 0 ? asked : 24);
    const auto warmup  = std::max(1, std::stoi(setting("NINFER_BENCH_WARMUP", "6")));
    const auto with_df2 = setting("NINFER_BENCH_DF2", "1") != "0";
    const auto bar     = std::stod(setting("NINFER_BENCH_BAR", "2.998"));
    const auto round_ms = std::stod(setting("NINFER_BENCH_ROUND_MS", "19.137"));

    try {
        Probe probe(artifact_path, device);
        work_device = probe.device;
        owner       = &probe;
        probe.bind();
        probe.materialize();

        const auto head = probe.weight(probe.head);
        const auto tg   = probe.weight(probe.weights.bridge.tg);
        const auto tsv  = probe.weight(probe.weights.bridge.tsv);
        const auto bound = probe.declared(probe.head);
        std::printf("head: %d x %d, tg/tsv rows %d x %d / %d x %d, the head's top-%d over %d valid "
                    "rows\n",
                    int(bound.n), int(bound.k), int(probe.target.vocabulary),
                    int(probe.config.bridge.rank), int(probe.target.vocabulary),
                    int(2 * probe.config.bridge.rank), int(kSlots), int(kTokenDomain));

        // One arena, sized from the op's own capacity query at the widest arm, as the working
        // probe sizes its walk's.
        std::size_t bytes = 0;
        for (const auto drafts : ks) {
            bytes = std::max(bytes, ops::linear_topk_workspace_capacity_bytes(
                                        bound.qtype, bound.n, bound.k, drafts, drafts));
        }
        bytes += 16u << 20;
        auto arena_storage = DeviceBuffer(probe.device, bytes);
        MemoryArena arena(arena_storage.span());
        std::printf("cost: %d timed iterations per arm behind %d warmup passes; one %zu B arena, "
                    "sized from linear_topk's own capacity query\n",
                    iters, warmup, bytes);
        std::fflush(stdout);

        std::vector<double> chain, scores, topk, tail_cost;
        for (const auto drafts : ks) {
            auto tail = allocate(probe, drafts);
            seed_tail(probe, tail, head, tg, tsv);
            const auto cuda = probe.cuda();

            // The module-load pass: the first launch of a kernel of a process is not a
            // measurement (D-39's red 1, the working probe's warmup walk).
            std::vector<std::int32_t> from, to;
            for (std::int32_t pass = 0; pass < warmup; ++pass) {
                gather_proposals(probe, tail);
                run_topk(probe, tail, head, arena);
                run_scores(probe, tail, head, tg, tsv);
                (void)bridge_round_trip(probe, tail, from, to);
                run_tail(probe, tail);
            }
            cuda_ok(cudaStreamSynchronize(cuda.stream()), "the warmup pass");

            Clock topk_clock, scores_clock, tail_clock, chain_clock;
            std::vector<double> bridge;
            for (std::int32_t pass = 0; pass < iters; ++pass) {
                topk_clock(cuda.stream(), [&] {
                    gather_proposals(probe, tail);
                    run_topk(probe, tail, head, arena);
                });
                scores_clock(cuda.stream(),
                             [&] { run_scores(probe, tail, head, tg, tsv); });
                bridge.push_back(bridge_round_trip(probe, tail, from, to));
                tail_clock(cuda.stream(), [&] { run_tail(probe, tail); });
                chain_clock(cuda.stream(), [&] {
                    gather_proposals(probe, tail);
                    run_topk(probe, tail, head, arena);
                    run_scores(probe, tail, head, tg, tsv);
                    (void)bridge_round_trip(probe, tail, from, to);
                    run_tail(probe, tail);
                });
            }

            const auto topk_cost   = describe(topk_clock.events);
            const auto scores_cost = describe(scores_clock.events);
            const auto tail_costs  = describe(tail_clock.events);
            const auto chain_cost  = describe(chain_clock.events);
            const auto bridge_cost = describe(bridge);
            topk_clock.events.clear();
            scores_clock.events.clear();
            tail_clock.events.clear();
            chain_clock.events.clear();
            std::printf("round at K=%d: the score table is %d CTAs (one per proposal row) x (16 "
                        "head-row gathers over %d = %d dequantised elements + a 16x16 terminal grid "
                        "over the rank %d = %d FMA); the K-by-K grid of the op is over its 16 "
                        "CANDIDATE axes (the artifact's head_topk), and the draft count is its CTA "
                        "count\n",
                        drafts, drafts, int(probe.target.hidden_size),
                        int(kSlots * probe.target.hidden_size), int(probe.config.bridge.rank),
                        int(kSlots * kSlots * probe.config.bridge.rank));
            publish_check(probe, tail);
            report("head_topk", drafts, topk_cost);
            report("score_table", drafts, scores_cost);
            report("serial_tail", drafts, tail_costs);
            report("layout_bridge", drafts, bridge_cost);
            report("whole_round", drafts, chain_cost);
            topk.push_back(topk_cost.mean);
            scores.push_back(scores_cost.mean);
            tail_cost.push_back(tail_costs.mean);
            chain.push_back(chain_cost.mean);

            if (with_df2) {
                // The same op at the DFlash2 proposal geometry: the (K+1)-wide masked block. The
                // block's columns are what the head projects, so the K term of both arms' costs is
                // this launch. NOT DFlash2's whole round: its draft layers and selector need a
                // DFlash2 drafter and a lane.
                const auto columns = drafts + 1;
                auto df2 = allocate(probe, columns);
                seed_tail(probe, df2, head, tg, tsv);
                for (std::int32_t pass = 0; pass < warmup; ++pass) {
                    gather_proposals(probe, df2);
                    run_topk(probe, df2, head, arena);
                }
                cuda_ok(cudaStreamSynchronize(cuda.stream()), "the df2 warmup pass");
                Clock df2_clock;
                for (std::int32_t pass = 0; pass < iters; ++pass) {
                    df2_clock(cuda.stream(), [&] {
                        gather_proposals(probe, df2);
                        run_topk(probe, df2, head, arena);
                    });
                }
                std::printf("df2 geometry: the same head top-%d over %d valid rows at the "
                            "(K+1)=%d-wide masked block (the K term of the other arm's head cost, "
                            "not its whole round)\n",
                            int(kSlots), int(kTokenDomain), columns);
                report("df2_topk", columns, describe(df2_clock.events));
            }
        }

        std::printf("scaling: the measured exponent of ms/round against K (K^1 = one term per "
                    "draft, K^2 = D-43's grid reading)\n");
        std::printf("SCALING whole_round  measured exponent %.3f (a K^1 term reads 1.000, D-43's"
                    " K^2 grid reading reads 2.000); cost(K=%d)/cost(K=%d) = %.3f, where a K^1"
                    " term predicts %.3f and a K^2 term %.3f\n",
                    exponent(ks, chain), ks.back(), ks.front(),
                    chain.back() / std::max(1e-9, chain.front()),
                    double(ks.back()) / double(ks.front()),
                    std::pow(double(ks.back()) / double(ks.front()), 2.0));
        std::printf("SCALING head_topk    measured %.3f  cost ratio %.3f\n",
                    exponent(ks, topk), topk.back() / std::max(1e-9, topk.front()));
        std::printf("SCALING score_table  measured %.3f  cost ratio %.3f\n",
                    exponent(ks, scores), scores.back() / std::max(1e-9, scores.front()));
        std::printf("SCALING serial_tail  measured %.3f  cost ratio %.3f\n",
                    exponent(ks, tail_cost), tail_cost.back() / std::max(1e-9, tail_cost.front()));

        // The break-even, on measured cost: criterion ⑤ is accepted tokens per unit TIME, so a
        // round that costs C ms buys acceptance proportionally. prod's own round time is the
        // denominator: the bar 2.998 arrives every round of that length.
        const auto widest = ks.back();
        const auto widest_cost = chain[chain.size() - 1];
        const auto k7 = [&] {
            for (std::size_t index = 0; index < ks.size(); ++index) {
                if (ks[index] == 7) { return chain[index]; }
            }
            return -1.0;
        }();
        std::printf("break-even at K=%d: the round costs %.3f ms against prod's measured %.3f "
                    "ms/round (the batch-1 decode step), the bar %.3f accepted drafts/round\n",
                    widest, widest_cost, round_ms, bar);
        const auto needed = [&](const double extra_ms) {
            return bar * (round_ms + extra_ms) / round_ms;
        };
        std::printf("BREAK-EVEN additive bound (all %.3f ms is new work on top of a prod round): "
                    "accepted/round must be >= %.3f, i.e. +%.3f over the bar\n",
                    widest_cost, needed(widest_cost), needed(widest_cost) - bar);
        if (k7 > 0.0) {
            const auto delta = widest_cost - k7;
            std::printf("BREAK-EVEN window delta (K=%d costs %.3f ms, K=7 costs %.3f ms, delta "
                        "%.3f ms): accepted/round must be >= %.3f, i.e. +%.3f over the bar\n",
                        widest, widest_cost, k7, delta, needed(delta), needed(delta) - bar);
        }
        std::printf("CEILING: K=%d can publish at most %d accepted drafts/round; the bar is %.3f, "
                    "so the headroom the extra window must beat is +%.3f\n",
                    widest, widest, bar, double(widest) - bar);
        const auto tokens_per_round = bar + 1.0;
        const auto tok_s            = tokens_per_round / (round_ms / 1000.0);
        const auto carried          = round_ms + widest_cost;
        std::printf("PROD-SCALE: the bar's %.3f accepted + the verified target token over prod's "
                    "%.3f ms/round = %.1f tok/s; a round that carries the whole %.3f ms DSpine tail "
                    "on top runs %.1f tok/s at the SAME acceptance, %.2f %% of it, so the acceptance "
                    "the wide window must add to break even is +%.2f %%\n",
                    bar, round_ms, tok_s, widest_cost, tokens_per_round / (carried / 1000.0),
                    100.0 * tokens_per_round / (carried / 1000.0) / tok_s,
                    100.0 * (needed(widest_cost) - bar) / bar);
        std::fflush(stdout);
        return 0;
    } catch (const std::exception& error) {
        std::printf("STOP: %s\n", error.what());
        std::fflush(stdout);
        return 1;
    }
}
