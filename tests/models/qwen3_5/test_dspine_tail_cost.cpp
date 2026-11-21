// The DSpine proposal-tail COST curve, measured on the real drafter artifact.
//
// THE QUESTION THIS EXISTS TO ANSWER (D-43, unmeasured there; criterion ⑤ is accepted tokens per
// unit TIME, and D-40 put the bar at DFlash2's measured 2.998 accepted drafts/round). DSpine
// drafts K=15 and DFlash2 K=7, and the tail's serial half is read in the plan as a K-by-K grid of
// head projections over the vocabulary - "K^2 x 248320 x 5120 matvecs per round", 49 batches at
// K=7 against 225 at K=15. If that scaling is real, DSpine's extra window costs 4.6x the wall
// clock for at most 2x the accepted tokens and the build cannot win on tok/s at any acceptance.
// This bench measures the cost/round of the proposal-tail path at K = 1, 3, 7, 15 on the REAL
// head/tg/tsv of the V3 drafter, in CUDA-event milliseconds, with the per-op breakdown, and the
// same op (linear_topk over a [248320, 5120] head) at DFlash2's K+1-wide proposal geometry.
//
// WHAT IS MEASURED (every number is a launch of engine code, nothing is modelled)
//   whole   execution::dspine_propose_tail - the engine's proposal tail, the half a decode round
//           runs (src/models/qwen3_5/execution/dspine.cpp): the K-way head top-k, the score table,
//           the sampled tail, and the layout bridges the accept op's contract requires.
//   steps   the same sequence op by op: the proposal gather, ops::linear_topk (the tower head's
//           top-16 over 248077 valid rows, one K-wide launch), ops::dspine_serial_scores (the
//           score table: the head-row gathers + the K-by-K terminal dots), ops::dspine_serial_tail,
//           and the host-side layout bridges with their stream settles.
//   isolate three arms of ONE op (dspine_serial_scores) at the SAME launch geometry, separating
//           the vocabulary-random head gather from the rank-scaled dots: the real tables; a
//           rank=1 table set (the dots collapse, the head gather does not); and a 16-row head with
//           16-row tables (the random 248320-row gather disappears, the dots do not).
//   dflash2 ops::linear_topk over a [248320, 5120] head at the DFlash2 proposal geometry, i.e. the
//           (K+1)-wide masked block (2, 4, 8, 16 columns) - the same op, the same head, the K term
//           of both arm's costs. It is NOT DFlash2's whole round: its draft layers and selector
//           need a DFlash2 drafter and a lane, and this bench says so rather than guess them.
//
// WHAT IT IS NOT. No acceptance number (D-42: that needs the target's logits, i.e. the a5000-2
// lane), no numerics claim (the synthetic streams are cost carriers; the rank=1 and 16-row arms
// are shape isolations and are labelled), and no optimisation: nothing here changes a kernel, a
// tolerance, a weight copy, or a shape gate. The K=15 arm is the production draft count, and the
// locked instruction is to report what exists.
//
// NINFER_TEST_ARTIFACT  the V3-tailored DSpine artifact (skip, exit 77, when unset)
// NINFER_TEST_DEVICE   the visible CUDA ordinal to run on; required, no default (D-29: a device
//                      probe never guesses a card; the card it landed on is printed)
// NINFER_BENCH_DRAFTS  comma list of K to cost (default "1,3,7,15")
// NINFER_BENCH_ITERS   timed iterations per K (default 24, floor 20 as the task requires)
// NINFER_BENCH_WARMUP  un-timed iterations, which pay the module load (default 4)
// NINFER_BENCH_DF2     "0" drops the DFlash2-geometry linear_topk arm
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
#include "models/common/dspine/weights.h"
#ifdef NINFER_TAIL_COST_WITH_EXECUTION
#include "models/qwen3_5/execution/dspine.h"
#endif
#include "ninfer/ops/dspine_bridge.h"
#include "ninfer/ops/linear_topk.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace ninfer;
using namespace ninfer::models;
#ifdef NINFER_TAIL_COST_WITH_EXECUTION
using namespace ninfer::models::qwen3_5;
#endif
namespace dspine = ninfer::models::dspine;

constexpr double kGiB          = 1024.0 * 1024.0 * 1024.0;
constexpr std::int32_t kBatch  = 1;
constexpr std::int32_t kTokenDomain = 248077;
constexpr std::int32_t kSlots       = 16;

std::int32_t dimension(const std::uint64_t value) {
    if (value == 0 || value > std::uint64_t(std::numeric_limits<std::int32_t>::max()))
        throw std::overflow_error("a dimension leaves the Tensor integer domain");
    return static_cast<std::int32_t>(value);
}

void cuda_ok(const cudaError_t error, const char* what) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(error));
    cuda_ok(cudaGetLastError(), what);
}

std::string setting(const std::string_view name, const std::string_view fallback = {}) {
    const char* value = std::getenv(std::string(name).c_str());
    return value && *value ? std::string(value) : std::string(fallback);
}

std::uint32_t field(const artifact::Json& config, const char* name) {
    return static_cast<std::uint32_t>(artifact::require_u64(config.at(name), name));
}

std::uint16_t to_bf16(const float value) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    return static_cast<std::uint16_t>((bits >> 16) + ((bits >> 16) & 1U));
}

// ---------------------------------------------------------------- the statistics -------------

struct Cost {
    std::int32_t samples = 0;
    double mean = 0.0, p50 = 0.0, low = 0.0, high = 0.0;
};

Cost describe(std::vector<double> values) {
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

void report(const char* label, const std::int32_t drafts, const char* unit, const Cost& cost) {
    std::printf("COST %-22s K=%-2d %s mean %8.3f  p50 %8.3f  range %8.3f .. %8.3f  N=%d\n", label,
                drafts, unit, cost.mean, cost.p50, cost.low, cost.high, cost.samples);
    std::fflush(stdout);
}

// CUDA-event timing over one stream. The events bracket the launches; the wall clock is carried
// alongside because the tail's layout bridges settle the stream, so a round's host time and its
// device span are the same interval and a divergence between the two is a launch gap.
struct Clock {
    cudaEvent_t start{}, stop{};
    std::vector<double> events, wall;

    explicit Clock(const char* name) {
        cuda_ok(cudaEventCreate(&start, cudaEventDefault), "event create");
        cuda_ok(cudaEventCreate(&stop, cudaEventDefault), "event create");
        (void)name;
    }
    Clock(const Clock&)            = delete;
    Clock& operator=(const Clock&) = delete;
    ~Clock() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    template <class Run>
    void operator()(cudaStream_t stream, Run&& run) {
        const auto began = std::chrono::steady_clock::now();
        cuda_ok(cudaEventRecord(start, stream), "event record start");
        run();
        cuda_ok(cudaEventRecord(stop, stream), "event record stop");
        cuda_ok(cudaEventSynchronize(stop), "event synchronize");
        float milliseconds = 0.0f;
        cuda_ok(cudaEventElapsedTime(&milliseconds, start, stop), "event elapsed");
        events.push_back(milliseconds);
        const auto elapsed = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(
            std::chrono::steady_clock::now() - began);
        wall.push_back(elapsed.count());
    }
};

// ---------------------------------------------------------------- the harness ---------------

// The round's world: the streams a proposal tail reads and writes, plus the load set it names.
// The 24 GiB target payload is never requested; the head, the drafter's final norm, the gate bias
// and the bridge's two serial tables are, which is what makes a proposal tail costable at all
// (the D-20 load discipline, taken from the seam probe).
struct Harness {
    Harness(const std::string& path, const int ordinal)
        : reader(make_reader(path)), device(Device::cuda(ordinal)), context(ordinal) {
        cuda_ok(cudaSetDevice(ordinal), "set device");
        cudaDeviceProp properties{};
        cuda_ok(cudaGetDeviceProperties(&properties, ordinal), "device properties");
        std::printf("device: visible ordinal %d, name \"%s\", pci %02x:%02x, %llu MiB total\n",
                    ordinal, properties.name, properties.pciBusID, properties.pciDeviceID,
                    static_cast<unsigned long long>(properties.totalGlobalMem / 1024 / 1024));
        std::printf("        CUDA_VISIBLE_DEVICES=\"%s\"\n",
                    setting("CUDA_VISIBLE_DEVICES", "<unset>").c_str());
    }

    // The D-45 stop is inside this construction and the three members initialise on one line, so
    // each of them names itself: a curve that stops knows whether it stopped in the manifest
    // parse, in the device, or in the context.
    // The setup names each of its three steps. This probe is a measurement, and the failure it is
    // most likely to have is that it never reaches a measurement: D-45 recorded a target of this
    // family that stops before its first line, and a curve that stops is only useful if it says
    // which step it stopped in.
    static artifact::Reader make_reader(const std::string& path) {
        std::printf("        phase: the manifest parse begins\n");
        std::fflush(stdout);
        auto reader = artifact::Reader(path);
        std::printf("        phase: the manifest parse ends\n");
        std::fflush(stdout);
        return reader;
    }

    void bind() {
        std::printf("        phase: the bind begins\n"); std::fflush(stdout);
        // A full artifact names the tower and its output head; a drafter-only manifest names the
        // drafter's own geometry and carries no head, so the head's geometry (248320 x 5120,
        // q8_g32_fp16, row_split_k128, and 5120 = 40 x 128 so the padded row stride is the logical
        // one) is rebuilt from the numbers and the head is made resident at it. The line that
        // says so is printed with the curve, and the cost of every arm here is a function of that
        // geometry and not of the head's bits.
        const auto has_text = reader.directory().components.contains("text");
        if (has_text) {
            const auto& text = reader.directory().component("text").config;
            target           = dspine::TargetGeometry{field(text, "hidden_size"),
                                                      field(text, "vocab_size"),
                                                      field(text, "num_hidden_layers")};
        } else {
            const auto& draft = reader.directory().component("dspine").config;
            target            = dspine::TargetGeometry{field(draft, "hidden_size"),
                                                       field(draft, "vocab_size"),
                                                       field(draft, "num_target_layers")};
        }
        config = dspine::parse_config(reader.directory().component("dspine"), target);
        binder   = std::make_unique<artifact::Binder>(reader);
        bindings = std::make_unique<Bindings>(*binder);
        // The proposal tail names five weights and no draft layer: the tower's head, the
        // drafter's final norm, the bridge's gate bias, and its tg/tsv serial tables. They are
        // bound by name, in the representation each one declares, which is the same binding
        // dspine::bind_weights performs for them - and it is the whole load set of a tail, so a
        // drafter-only manifest carries all of it but the head.
        const auto rank = config.bridge.rank;
        final_norm_id   = bindings->direct_as_declared("dspine/final_norm",
                                                       artifact::Shape{target.hidden_size},
                                                       std::vector<QType>{QType::BF16});
        gate_bias_id    = bindings->direct_as_declared("dspine/bridge/gate_bias",
                                                        artifact::Shape{rank},
                                                        std::vector<QType>{QType::BF16});
        tg_id           = bindings->direct_as_declared(
            "dspine/bridge/tg", artifact::Shape{target.vocabulary, rank},
            std::vector<QType>{QType::NVFP4, QType::BF16});
        tsv_id          = bindings->direct_as_declared(
            "dspine/bridge/tsv", artifact::Shape{target.vocabulary, 2 * rank},
            std::vector<QType>{QType::NVFP4, QType::BF16});
        if (has_text) {
            head = bindings->direct_as_declared(
                "text/output_head", artifact::Shape{target.vocabulary, target.hidden_size},
                std::vector<QType>{QType::Q8_G32_FP16, QType::FP8_E4M3FN_ROW_BF16, QType::BF16});
        }
        synthetic_head = !has_text;
        std::printf("        phase: the drafter is bound\n"); std::fflush(stdout);
        std::printf("dspine: hidden=%u block=%u code=%u rank=%u head_topk=%u mask=%u vocab=%u\n",
                    config.hidden_size, config.block_size, config.bridge.code_size,
                    config.bridge.rank, config.head_topk, config.mask_token_id,
                    target.vocabulary);
    }

    std::vector<WeightId> requested() const {
        if (synthetic_head) { return {final_norm_id, gate_bias_id, tg_id, tsv_id}; }
        return {final_norm_id, gate_bias_id, tg_id, tsv_id, head};
    }

    void materialize() {
        std::set<std::size_t> objects;
        std::vector<artifact::PlacementRequest> requests;
        for (const auto id : requested())
            for (const auto& part : bindings->at(id).reference.binding.parts)
                if (objects.insert(part.object.index).second)
                    requests.push_back({part.object, device, artifact::HostAccess::CpuOnly});
        std::uint64_t declared = 0;
        for (const auto index : objects)
            declared += artifact::object_bytes(
                reader.directory().object(artifact::ObjectHandle{std::size_t(index)}));
        std::uint64_t free_before = 0, total = 0;
        cuda_ok(cudaMemGetInfo(&free_before, &total), "memory info");
        backing = std::make_unique<artifact::MaterializedArtifact>(artifact::materialize(
            reader, artifact::plan_materialization(reader, requests), context.transfer_view()));
        context.synchronize();
        std::uint64_t free_after = 0;
        cuda_ok(cudaMemGetInfo(&free_after, &total), "memory info");
        std::printf("load: %zu objects, %.5f GiB declared; device free %llu -> %llu MiB of %llu\n",
                    objects.size(), declared / kGiB,
                    static_cast<unsigned long long>(free_before / 1024 / 1024),
                    static_cast<unsigned long long>(free_after / 1024 / 1024),
                    static_cast<unsigned long long>(total / 1024 / 1024));
        std::fflush(stdout);
    }

    [[nodiscard]] WeightView view(const WeightId id) const {
        return artifact::bind_view(bindings->at(id).reference, *backing, device);
    }
    [[nodiscard]] Weight weight(const WeightId id) {
        if (id == head && head_is_synthetic()) { return synthetic_head_weight(); }
        return native_weight(view(id), 1.0F);
    }

    [[nodiscard]] bool head_is_synthetic() const { return synthetic_head; }

    // A [vocabulary, hidden] Q8_G32_FP16 RowSplit head, at the geometry the attach record names
    // (row_split_k128, group 32, FP16 group scales), resident on the card. Every arm's work is a
    // function of the geometry, and the dequantisation path it exercises is the geometry's, not
    // the bytes'.
    [[nodiscard]] Weight synthetic_head_weight() {
        if (not synthetic_head_storage.empty()) { return synthetic_head_storage.front(); }
        const auto rows     = std::int32_t(target.vocabulary);
        const auto columns  = std::int32_t(target.hidden_size);
        const auto groups   = columns / 32;
        auto codes          = std::make_unique<DeviceBuffer>(device,
                                                            std::size_t(rows) * columns);
        auto scales         = std::make_unique<DeviceBuffer>(device,
                                            std::size_t(rows) * groups * 2);
        const CudaDeviceScope settling(device);
        cuda_ok(cudaMemset(codes->data(), 0x3C, codes->size()), "clear the head's codes");
        cuda_ok(cudaMemset(scales->data(), 0x2D, scales->size()), "clear the head's scales");
        Weight out{};
        out.device          = device;
        out.qtype           = QType::Q8_G32_FP16;
        out.layout          = QuantLayout::RowSplit;
        out.group           = 32;
        out.group_size      = 32;
        out.qdata           = codes->data();
        out.scales          = scales->data();
        out.n               = rows;
        out.k               = columns;
        out.shape[0]        = rows;
        out.shape[1]        = columns;
        out.padded_shape[0] = rows;
        out.padded_shape[1] = columns;
        out.ndim            = 2;
        out.scale_dtype     = DType::FP16;
        synthetic_head_storage.push_back(out);
        head_storage.push_back(std::move(codes));
        head_storage.push_back(std::move(scales));
        std::printf("head: SYNTHETIC at the artifact's geometry, %d x %d q8_g32_fp16 row_split "
                    "(the cost of these arms is a function of the geometry, not of the head's "
                    "bits)\n", rows, columns);
        std::fflush(stdout);
        return out;
    }
    [[nodiscard]] Tensor parameter(const WeightId id) const {
        const auto& shape = bindings->at(id).reference.shape;
        return weight_tensor(view(id), {dimension(shape[0]), 1, 1, 1});
    }

    [[nodiscard]] const std::vector<Weight>& synthetic_head_weights() {
        if (synthetic_head_storage.empty()) { synthetic_head_weight(); }
        return synthetic_head_storage;
    }

    [[nodiscard]] Tensor buffer(const std::size_t bytes, const DType dtype,
                                 const std::initializer_list<std::int32_t> shape) {
        storage.push_back(std::make_unique<DeviceBuffer>(device, bytes));
        auto& slot = *storage.back();
        const CudaDeviceScope clearing(device);
        cuda_ok(cudaMemset(slot.data(), 0, slot.size()), "clear buffer");
        return Tensor(slot.span(), dtype, shape);
    }

    void upload(const Tensor& stream, const void* source, const std::size_t bytes) {
        const auto target = static_cast<std::byte*>(stream.data);
        for (const auto& slot : storage) {
            const auto base = static_cast<std::byte*>(const_cast<void*>(
                static_cast<const void*>(slot->data())));
            if (target >= base && target < base + slot->size()) {
                slot->copy_from_host(source, bytes, std::size_t(target - base));
                return;
            }
        }
        throw std::runtime_error("a copy named a buffer the harness does not own");
    }

    void download(void* destination, const Tensor& stream, const std::size_t bytes) {
        const auto target = static_cast<const std::byte*>(stream.data);
        for (const auto& slot : storage) {
            const auto base = static_cast<std::byte*>(slot->data());
            if (target >= base && target < base + slot->size()) {
                slot->copy_to_host(destination, bytes, std::size_t(target - base));
                return;
            }
        }
        throw std::runtime_error("a copy named a buffer the harness does not own");
    }

    CudaExecutionView cuda() const { return context.execution_view(); }

    // The members and their initialisation are in this order because the two probes that already
    // read this artifact have them in this order: manifest, device, context. A probe that costs a
    // proposal tail has no business inventing a device bring-up sequence of its own.
    artifact::Reader reader;
    Device device;
    DeviceContext context;
    dspine::TargetGeometry target{};
    dspine::Config config;
    std::unique_ptr<artifact::Binder> binder;
    std::unique_ptr<Bindings> bindings;
    WeightId head{}, final_norm_id{}, gate_bias_id{}, tg_id{}, tsv_id{};
    bool synthetic_head = false;
    std::unique_ptr<artifact::MaterializedArtifact> backing;
    std::vector<std::unique_ptr<DeviceBuffer>> storage;
    std::vector<std::unique_ptr<DeviceBuffer>> head_storage;
    std::vector<Weight> synthetic_head_storage;
};

Harness* world = nullptr;

std::vector<std::uint16_t> raw_for(Harness& probe, std::int32_t columns);

// ---------------------------------------------------------------- one round ------------------

// The streams of one proposal round at K drafts and batch one, plus the pieces the step-wise
// breakdown needs (the same intermediates the engine's tail allocates on its arena).
struct Round {
    std::int32_t drafts = 0;
    std::int32_t width  = 0;
    Tensor bridge_raw, bridge_rms, bridge_message, proposal_hidden, anchors, draw_positions;
    Tensor configs;
    Tensor out_drafts, out_candidates, out_proposal_q;
    // the breakdown's streams
    Tensor proposals, scores, index, table, tail_candidates, tail_drafts, tail_q, tail_index;
    Tensor ids_small;
    // the isolation arms' synthetic shapes
    Tensor tiny_codes, tiny_scales, tiny_raw, tiny_message, tiny_gate_bias;
    Tensor rank1_raw, rank1_message, rank1_gate_bias, rank1_tg, rank1_tsv;
};

Round allocate_round(Harness& probe, const std::int32_t drafts) {
    const auto h       = dimension(probe.target.hidden_size);
    const auto r       = dimension(probe.config.bridge.rank);
    const auto b       = kBatch;
    const auto width   = drafts + 1;
    const auto vocab   = dimension(probe.target.vocabulary);
    Round out;
    out.drafts = drafts;
    out.width  = width;
    out.bridge_raw = probe.buffer(std::size_t(h) * width * b * 2, DType::BF16, {h, width, b, 1});
    out.bridge_rms = probe.buffer(std::size_t(width * b) * 4, DType::FP32, {width, b, 1, 1});
    out.bridge_message =
        probe.buffer(std::size_t(r) * width * b * 2, DType::BF16, {r, width, b, 1});
    out.proposal_hidden =
        probe.buffer(std::size_t(h) * width * b * 2, DType::BF16, {h, width, b, 1});
    out.anchors        = probe.buffer(std::size_t(b) * 4, DType::I32, {b, 1, 1, 1});
    out.draw_positions = probe.buffer(std::size_t(b) * 4, DType::I32, {b, 1, 1, 1});
    out.configs =
        probe.buffer(sizeof(ops::SamplingConfig) * std::size_t(b), DType::FP32, {b, 1, 1, 1});
    out.out_drafts     = probe.buffer(std::size_t(drafts * b) * 4, DType::I32, {drafts, b, 1, 1});
    out.out_candidates = probe.buffer(std::size_t(kSlots) * drafts * b * 4, DType::I32,
                                      {kSlots, drafts, b, 1});
    out.out_proposal_q = probe.buffer(std::size_t(kSlots) * drafts * b * 4, DType::FP32,
                                       {kSlots, drafts, b, 1});
    out.proposals = probe.buffer(std::size_t(h) * drafts * b * 2, DType::BF16, {h, drafts, b, 1});
    out.scores = probe.buffer(std::size_t(kSlots) * drafts * b * 4, DType::FP32,
                              {kSlots, drafts, b, 1});
    out.index = probe.buffer(std::size_t(kSlots) * drafts * b * 4, DType::I32,
                             {kSlots, drafts, b, 1});
    out.table = probe.buffer(std::size_t(b) * kSlots * kSlots * drafts * 4, DType::FP32,
                             {b, kSlots, kSlots, drafts});
    out.tail_candidates = probe.buffer(std::size_t(b) * drafts * kSlots * 4, DType::I32,
                                       {b, drafts, kSlots, 1});
    out.tail_drafts = probe.buffer(std::size_t(b) * drafts * 4, DType::I32, {b, drafts, 1, 1});
    out.tail_q      = probe.buffer(std::size_t(b) * drafts * kSlots * 4, DType::FP32,
                                   {b, drafts, kSlots, 1});
    out.tail_index  = probe.buffer(std::size_t(b) * drafts * kSlots * 4, DType::I32,
                                   {b, drafts, kSlots, 1});
    out.ids_small   = probe.buffer(std::size_t(kSlots) * drafts * b * 4, DType::I32,
                                   {kSlots, drafts, b, 1});

    // A 16-row head in the tower's own representation (Q8_G32_FP16, RowSplit, group 32, FP16
    // scales, the padded row stride of the real head) with 16-row tg/tsv. Same kernel, same grid,
    // same dots: the vocabulary-random gather out of the 1.27 GiB head is the only term it removes.
    const auto head         = probe.weight(probe.head);
    const auto row_stride   = head.padded_shape[1] > 0 ? head.padded_shape[1] : head.k;
    out.tiny_codes  = probe.buffer(std::size_t(kSlots) * row_stride, DType::U8, {kSlots, 1, 1, 1});
    out.tiny_scales = probe.buffer(std::size_t(kSlots) * (row_stride / 32) * 2, DType::FP16,
                                   {kSlots, 1, 1, 1});
    out.tiny_raw    = probe.buffer(std::size_t(kSlots) * r * 2, DType::BF16, {kSlots, 1, 1, 1});
    out.tiny_message = probe.buffer(std::size_t(kSlots) * 2 * r * 2, DType::BF16, {kSlots, 1, 1, 1});
    out.tiny_gate_bias = probe.buffer(std::size_t(r) * 2, DType::BF16, {r, 1, 1, 1});

    // The same op at rank 1: the K-by-K terminal dots and the m_s build collapse to one multiply,
    // the head-row gathers and the grid geometry do not. The tables are BF16, the format
    // require_table accepts beside NVFP4, over the head's own row count.
    out.rank1_raw        = out.bridge_raw;
    out.rank1_message    = probe.buffer(std::size_t(1) * width * b * 2, DType::BF16,
                                         {1, width, b, 1});
    out.rank1_gate_bias  = probe.buffer(2, DType::BF16, {1, 1, 1, 1});
    out.rank1_tg         = probe.buffer(std::size_t(vocab) * 1 * 2, DType::BF16, {vocab, 1, 1, 1});
    out.rank1_tsv        = probe.buffer(std::size_t(vocab) * 2 * 2, DType::BF16, {vocab, 2, 1, 1});
    (void)vocab;
    return out;
}

void upload_i32(const Tensor& stream, const std::vector<std::int32_t>& values) {
    world->upload(stream, values.data(), values.size() * sizeof(std::int32_t));
}

// The round's inputs, seeded from the iteration so a timed stream is never a constant the driver
// can fold. The values are cost carriers: the ops' work per round is a function of the shapes and
// the geometry, not of these bytes, and the two isolation arms say which shape they changed.
void seed_round(Harness& probe, Round& round, const std::uint64_t seed) {
    const CudaDeviceScope settling(probe.device);
    cuda_ok(cudaDeviceSynchronize(), "settle before a reset");
    for (const auto* stream : {&round.bridge_raw, &round.bridge_rms, &round.bridge_message,
                               &round.proposal_hidden, &round.out_drafts, &round.out_candidates,
                               &round.out_proposal_q, &round.table})
        cuda_ok(cudaMemset(stream->data, 0x3F, stream->bytes()), "reset a round stream");

    const std::int32_t anchor = std::int32_t(1000 + std::int32_t(seed % 90000));
    upload_i32(round.anchors, {anchor});
    upload_i32(round.draw_positions, {std::int32_t(2048 + std::int32_t(seed % 4096))});

    const auto h        = std::int64_t(probe.target.hidden_size);
    const auto r        = std::int64_t(probe.config.bridge.rank);
    const auto columns  = std::int64_t(round.width) * kBatch;
    std::uint64_t state = seed * 0x9E3779B97C460C17ULL + 0x2545F4914F6CDD1DULL;
    const auto unit     = [&] {
        state ^= state >> 12; state ^= state << 25; state ^= state >> 27;
        return float(((state * 0x2545F4914F6CDD1DULL) >> 40) / float(1ULL << 24));
    };
    std::vector<std::uint16_t> raw{std::size_t(h * columns)};
    std::vector<std::uint16_t> message{std::size_t(r * columns)};
    for (auto& word : raw) { word = to_bf16(unit() * 3.0F - 1.5F); }
    for (auto& word : message) { word = to_bf16(unit() * 1.6F - 0.8F); }
    world->upload(round.bridge_raw, raw.data(), raw.size() * 2);
    world->upload(round.bridge_message, message.data(), message.size() * 2);
    world->upload(round.proposal_hidden, raw.data(), raw.size() * 2);

    auto rms = std::vector<float>(std::size_t(columns), 1.0F);
    for (std::size_t index = 1; index < rms.size(); ++index)
        rms[index] = 1.0F + 0.05F * float(index % 7);
    world->upload(round.bridge_rms, rms.data(), rms.size() * 4);

    // The production policy (D-24's arm: T=0.9, top_k 20, top_p 0.95) - the one the tail is
    // sampled under in a live round, and the non-degenerate one D-42 says the probe lacks.
    std::vector<ops::SamplingConfig> configs{std::size_t(kBatch), ops::SamplingConfig{}};
    for (auto& config : configs) {
        config.temperature = 0.9F;
        config.top_k       = 20;
        config.top_p       = 0.95F;
        config.min_p       = 0.0F;
        config.seed        = seed;
    }
    world->upload(round.configs, configs.data(), configs.size() * sizeof(ops::SamplingConfig));

    // The isolation arms' carriers.
    std::vector<std::int32_t> small(std::size_t(kSlots) * round.drafts * kBatch);
    for (std::size_t index = 0; index < small.size(); ++index)
        small[index] = std::int32_t((index + seed) % kSlots);
    upload_i32(round.ids_small, small);
    std::vector<std::uint16_t> small_field(std::size_t(kSlots) * 2 * r);
    for (auto& word : small_field) { word = to_bf16(unit() * 1.6F - 0.8F); }
    world->upload(round.tiny_raw, small_field.data(), std::size_t(kSlots) * r * 2);
    world->upload(round.tiny_message, small_field.data(), small_field.size() * 2);
    world->upload(round.tiny_gate_bias, small_field.data(), std::size_t(r) * 2);
    const auto rank1_words = std::size_t(columns);
    std::vector<std::uint16_t> rank1(rank1_words, to_bf16(0.0F));
    for (auto& word : rank1) { word = to_bf16(unit() * 1.6F - 0.8F); }
    world->upload(round.rank1_message, rank1.data(), rank1.size() * 2);
    world->upload(round.rank1_gate_bias, rank1.data(), 2);
}

// ---------------------------------------------------------------- the arms --------------------

// The tail's view over the real weights: the tower's own head (never copied), the drafter's
// final norm and gate bias, and the bridge's tg/tsv tables.
#ifdef NINFER_TAIL_COST_WITH_EXECUTION
execution::DspineTailView tail_view(Harness& probe) {
    return execution::DspineTailView{probe.weight(probe.head),
                                     probe.parameter(probe.final_norm_id),
                                     probe.parameter(probe.gate_bias_id),
                                     probe.weight(probe.tg_id),
                                     probe.weight(probe.tsv_id),
                                     dimension(probe.config.bridge.rank)};
}
#endif

// The head in a 16-row shape: the same Q8_G32_FP16 RowSplit geometry, the same padded row stride,
// the same kernel arm (HeadQ8=true), over 16 rows so every gather hits one L2-resident page.
Weight tiny_head(Harness& probe, const Round& round) {
    const auto real       = probe.weight(probe.head);
    const auto row_stride = real.padded_shape[1] > 0 ? real.padded_shape[1] : real.k;
    Weight out            = real;
    out.n            = kSlots;
    out.shape[0]     = kSlots;
    out.padded_shape[0] = kSlots;
    out.padded_shape[1] = row_stride;
    out.k            = real.k;
    out.qdata        = round.tiny_codes.data;
    out.scales       = round.tiny_scales.data;
    return out;
}

// A [rows, columns] BF16 table over the head's row count, for the rank=1 arm.
Weight bf16_table(const Tensor& stream, const std::int32_t rows, const std::int32_t columns) {
    Weight out{};
    out.device       = stream.device;
    out.qtype        = QType::BF16;
    out.layout       = QuantLayout::Contiguous;
    out.qdata        = stream.data;
    out.n            = rows;
    out.k            = columns;
    out.shape[0]     = rows;
    out.shape[1]     = columns;
    out.padded_shape[0] = rows;
    out.padded_shape[1] = columns;
    out.ndim         = 2;
    out.scale_dtype  = DType::BF16;
    return out;
}

// One score-table launch at a chosen (head, rank, tables, candidate stream). The grid is
// drafts * batch CTAs in every arm, so the geometry the cost question is about is held.
void score_table(Harness& probe, const Round& round, const Weight& head, const Tensor& raw,
                 const Tensor& rms, const Tensor& message, const Tensor& gate_bias,
                 const Tensor& norm_w, const Weight& tg, const Weight& tsv, Tensor& candidates,
                 Tensor& index, Tensor& table) {
    const auto width = round.width;
    ops::dspine_serial_scores(candidates, raw, rms, message, round.anchors, norm_w, gate_bias,
                              head, tg, tsv, width, index, &table, probe.cuda().stream());
    cuda_ok(cudaGetLastError(), "serial scores launch");
}

void cost_k(Harness& probe, MemoryArena& arena, const std::int32_t drafts,
            const std::int32_t iterations, const std::int32_t warmup) {
    const auto cuda   = probe.cuda();
    const auto stream = cuda.stream();
    auto round        = allocate_round(probe, drafts);
    const auto* policies = static_cast<const ops::SamplingConfig*>(round.configs.data);

    std::printf("\n===== K=%d (a %d-wide block, batch %d) =====\n", drafts, drafts + 1, kBatch);
    std::printf("grid: dspine_serial_scores %d CTA (256 threads), head rows/round %d, "
                "terminal dots/round %d, score table %.1f KiB\n",
                drafts * kBatch, drafts * kBatch * kSlots, drafts * kBatch * kSlots * kSlots,
                double(kBatch * kSlots * kSlots * drafts * 4) / 1024.0);
    std::fflush(stdout);

    // ---- the engine's proposal tail, whole --------------------------------------------------
    // The landmine this bench reproduces, in one line: a target whose link closure carries the
    // ninfer_model_execution archive stops in the CUDA context/module path on this 15,209-object
    // artifact (main thread asleep in anon_pipe_read, one core pegged, RSS 3.7 MiB, 0 % GPU,
    // K-independent), while the forward and shape probes - ninfer_model_loading and ninfer_ops,
    // this file's closure - read the same manifest and build the same views in 0.2 s. So the
    // whole-tail arm is compiled only into that closure, and the curve below runs the tail's
    // five calls at the engine's own arguments, which is what dspine.cpp's body is.
#ifdef NINFER_TAIL_COST_WITH_EXECUTION
    Clock whole("whole");
    for (std::int32_t index = 0; index < warmup; ++index) {
        seed_round(probe, round, 0x50371153ULL + std::uint64_t(index));
        execution::dspine_propose_tail(view, inputs, outputs, arena, cuda);
    }
    cuda_ok(cudaStreamSynchronize(stream), "warmup settle");
    for (std::int32_t index = 0; index < iterations; ++index) {
        seed_round(probe, round, 0x50371153ULL + std::uint64_t(index));
        whole(stream, [&] { execution::dspine_propose_tail(view, inputs, outputs, arena, cuda); });
    }
    report("dspine_propose_tail", drafts, "ms", describe(whole.events));
    report("  same, host wall", drafts, "ms", describe(whole.wall));
#else
    std::printf("COST dspine_propose_tail    K=%-2d ms   not in this link closure (D-45: the "
                "execution archive in the closure stops the run; the five calls below are its "
                "body, at its arguments)\n", drafts);
#endif

    // ---- the same sequence, op by op ---------------------------------------------------------
    auto proposals = round.proposals;
    auto table     = round.table;
    auto head      = probe.weight(probe.head);
    auto norm_w    = probe.parameter(probe.final_norm_id);
    auto gate_bias = probe.parameter(probe.gate_bias_id);
    auto tg        = probe.weight(probe.tg_id);
    auto tsv       = probe.weight(probe.tsv_id);
    const auto grid_words = std::size_t(std::int64_t(kBatch) * kSlots * drafts);
    std::vector<std::int32_t> host_head, host_tail;
    std::vector<float> host_q;
    host_head.resize(grid_words);
    host_tail.resize(grid_words);
    host_q.resize(grid_words);

    Clock gather("gather"), topk("topk"), serial("serial_scores"), bridge("bridges"),
          tail("serial_tail");
    const auto step = [&](std::int32_t index) {
        seed_round(probe, round, 0x77AC0001ULL + std::uint64_t(index));
        gather(stream, [&] {
            const auto h = std::size_t(std::int64_t(probe.target.hidden_size) * drafts) * 2;
            const auto spitch =
                std::size_t(std::int64_t(probe.target.hidden_size) * round.width) * 2;
            cuda_ok(cudaMemcpy2DAsync(proposals.data, h,
                                      static_cast<std::byte*>(round.proposal_hidden.data)
                                          + std::size_t(probe.target.hidden_size) * 2,
                                      spitch, h, std::size_t(kBatch), cudaMemcpyDeviceToDevice,
                                      stream),
                    "proposal gather");
        });
        topk(stream, [&] {
            ops::linear_topk(proposals, head, kTokenDomain, round.out_candidates, round.scores,
                             arena, cuda);
            cuda_ok(cudaGetLastError(), "linear_topk launch");
        });
        serial(stream, [&] {
            score_table(probe, round, head, round.bridge_raw, round.bridge_rms,
                        round.bridge_message, gate_bias, norm_w, tg, tsv, round.out_candidates,
                        round.index, table);
        });
        bridge(stream, [&] {
            // The three crossings the accept op's contract forces (dspine.h's "three layouts"),
            // each a D2H, a host permute of 16*K words per row, and an H2D, each settling the
            // stream. Timed as one arm because the tail runs between the first and the other two.
            cuda_ok(cudaMemcpyAsync(host_head.data(), round.out_candidates.data,
                                    grid_words * sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                                    stream), "d2h candidates");
            cuda_ok(cudaStreamSynchronize(stream), "settle candidates");
            for (std::int32_t row = 0; row < kBatch; ++row)
                for (std::int32_t slot = 0; slot < drafts; ++slot)
                    for (std::int32_t cand = 0; cand < kSlots; ++cand)
                        host_tail[std::size_t(row) + std::int64_t(kBatch) *
                                        (std::int64_t(slot) + std::int64_t(drafts) * cand)] =
                            host_head[std::size_t(cand) + std::int64_t(kSlots) *
                                          (std::int64_t(slot) + std::int64_t(drafts) * row)];
            cuda_ok(cudaMemcpyAsync(round.tail_candidates.data, host_tail.data(),
                                    grid_words * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                    stream), "h2d candidates");
            cuda_ok(cudaStreamSynchronize(stream), "settle tail candidates");
        });
        tail(stream, [&] {
            ops::dspine_serial_tail(table, round.tail_candidates, policies, round.draw_positions,
                                     round.tail_drafts, round.tail_q, round.tail_index, stream);
            cuda_ok(cudaGetLastError(), "serial tail launch");
        });
    };
    for (std::int32_t index = 0; index < warmup; ++index) { step(index); }
    cuda_ok(cudaStreamSynchronize(stream), "step warmup settle");
    for (std::int32_t index = 0; index < iterations; ++index) { step(index); }
    report("  step proposal gather", drafts, "ms", describe(gather.events));
    report("  step linear_topk", drafts, "ms", describe(topk.events));
    report("  step serial_scores", drafts, "ms", describe(serial.events));
    report("  step layout bridges", drafts, "ms", describe(bridge.events));
    report("  step serial_tail", drafts, "ms", describe(tail.events));

    // ---- the anatomy of one score-table launch ----------------------------------------------
    Clock full("full"), flat_rank("rank1"), flat_head("tinyhead");
    auto tiny   = tiny_head(probe, round);
    const auto tiny_tg  = bf16_table(round.tiny_raw, kSlots, dimension(probe.config.bridge.rank));
    const auto tiny_tsv = bf16_table(round.tiny_message, kSlots,
                                    2 * dimension(probe.config.bridge.rank));
    const auto rank1 = std::int32_t(1);
    const auto rank1_tg  = bf16_table(round.rank1_tg, head.n, rank1);
    const auto rank1_tsv = bf16_table(round.rank1_tsv, head.n, 2);
    for (std::int32_t index = 0; index < warmup; ++index) {
        seed_round(probe, round, 0x13579ACEULL + std::uint64_t(index));
        full(stream, [&] {
            score_table(probe, round, head, round.bridge_raw, round.bridge_rms,
                        round.bridge_message, gate_bias, norm_w, tg, tsv, round.out_candidates,
                        round.index, table);
        });
        flat_rank(stream, [&] {
            score_table(probe, round, head, round.bridge_raw, round.bridge_rms,
                        round.rank1_message, round.rank1_gate_bias, norm_w, rank1_tg, rank1_tsv,
                        round.out_candidates, round.index, table);
        });
        flat_head(stream, [&] {
            score_table(probe, round, tiny, round.bridge_raw, round.bridge_rms,
                        round.bridge_message, round.tiny_gate_bias, norm_w, tiny_tg, tiny_tsv,
                        round.ids_small, round.index, table);
        });
    }
    cuda_ok(cudaStreamSynchronize(stream), "anatomy warmup settle");
    for (std::int32_t index = 0; index < iterations; ++index) {
        seed_round(probe, round, 0x13579ACEULL + std::uint64_t(index));
        full(stream, [&] {
            score_table(probe, round, head, round.bridge_raw, round.bridge_rms,
                        round.bridge_message, gate_bias, norm_w, tg, tsv, round.out_candidates,
                        round.index, table);
        });
        flat_rank(stream, [&] {
            score_table(probe, round, head, round.bridge_raw, round.bridge_rms,
                        round.rank1_message, round.rank1_gate_bias, norm_w, rank1_tg, rank1_tsv,
                        round.out_candidates, round.index, table);
        });
        flat_head(stream, [&] {
            score_table(probe, round, tiny, round.bridge_raw, round.bridge_rms,
                        round.bridge_message, round.tiny_gate_bias, norm_w, tiny_tg, tiny_tsv,
                        round.ids_small, round.index, table);
        });
    }
    report("  score table, real", drafts, "ms", describe(full.events));
    report("  score table, rank=1", drafts, "ms", describe(flat_rank.events));
    report("  score table, 16 rows", drafts, "ms", describe(flat_head.events));

    // ---- the head's top-k at the two arms' geometry ------------------------------------------
    // DFlash2's proposal head is the same op over a [248320, 5120] head on a (K+1)-wide masked
    // block; DSpine's is the same op on a K-wide proposal block. The column count is the K term.
    const auto hidden = dimension(probe.target.hidden_size);
    std::vector<std::int32_t> block_widths{2, 4, 8, drafts + 1};
    block_widths.erase(std::unique(block_widths.begin(), block_widths.end()),
                       block_widths.end());
    for (const auto columns : block_widths) {
        if (columns < 1 || columns > 16) { continue; }
        auto block = probe.buffer(std::size_t(hidden) * columns * kBatch * 2, DType::BF16,
                                   {hidden, columns, kBatch, 1});
        auto ids   = probe.buffer(std::size_t(kSlots) * columns * kBatch * 4, DType::I32,
                                  {kSlots, columns, kBatch, 1});
        auto rows  = probe.buffer(std::size_t(kSlots) * columns * kBatch * 4, DType::FP32,
                                  {kSlots, columns, kBatch, 1});
        world->upload(block, raw_for(probe, columns).data(),
                       std::size_t(hidden) * columns * kBatch * 2);
        Clock arm("topk columns");
        for (std::int32_t index = 0; index < warmup; ++index) {
            arm(stream, [&] {
                ops::linear_topk(block, head, kTokenDomain, ids, rows, arena, cuda);
                cuda_ok(cudaGetLastError(), "linear_topk launch");
            });
        }
        cuda_ok(cudaStreamSynchronize(stream), "topk warmup settle");
        for (std::int32_t index = 0; index < iterations; ++index) {
            arm(stream, [&] {
                ops::linear_topk(block, head, kTokenDomain, ids, rows, arena, cuda);
                cuda_ok(cudaGetLastError(), "linear_topk launch");
            });
        }
        char label[64];
        std::snprintf(label, sizeof(label), "topk %d-wide block", columns);
        report(label, drafts, "ms", describe(arm.events));
    }
}

std::vector<std::uint16_t> raw_for(Harness& probe, const std::int32_t columns) {
    std::vector<std::uint16_t> values(std::size_t(std::int64_t(probe.target.hidden_size)
                                                 * columns) * kBatch);
    std::uint64_t state = 0x2545F491ULL * (columns + 7);
    for (auto& word : values) {
        state ^= state >> 12; state ^= state << 25; state ^= state >> 27;
        const float value = float(((state * 0x2545F4914F6CDD1DULL) >> 40) / float(1ULL << 24))
                          * 3.0F - 1.5F;
        word = to_bf16(value);
    }
    return values;
}

std::vector<std::int32_t> parse_list(const std::string& text) {
    std::vector<std::int32_t> out;
    std::size_t begin = 0;
    while (begin <= text.size()) {
        const auto end = text.find(',', begin);
        const auto part = text.substr(begin, end == std::string::npos ? std::string::npos
                                                                      : end - begin);
        if (!part.empty()) { out.push_back(std::stoi(std::string(part))); }
        if (end == std::string::npos) { break; }
        begin = end + 1;
    }
    return out;
}
} // namespace

int main(const int argc, char** argv) {
    (void)argc;
    (void)argv;
    // A cost curve that is killed mid-run keeps the K rows it measured (D-34's lesson).
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    // The D-45 landmine is a target in this exact link closure that stops before its first line,
    // so every phase of the setup names itself as it is entered: a curve that dies is a curve
    // whose death point is known.
    std::printf("phase: main entered\n");
    const auto path = setting("NINFER_TEST_ARTIFACT");
    if (path.empty()) {
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
    const std::int32_t iterations = std::max(20, std::stoi(setting("NINFER_BENCH_ITERS", "24")));
    const std::int32_t warmup     = std::max(1, std::stoi(setting("NINFER_BENCH_WARMUP", "4")));
    auto ks                       = parse_list(setting("NINFER_BENCH_DRAFTS", "1,3,7,15"));
    try {
        std::printf("phase: a device is visible, opening the manifest %s\n", path.c_str());
        Harness probe(path, device);
        world = &probe;
        probe.bind();
        probe.materialize();
        const auto head = probe.weight(probe.head);
        std::printf("head: n=%d k=%d padded %dx%d qtype %d layout %d group %d; tg n=%d k=%d "
                    "qtype %d; tsv n=%d k=%d qtype %d; rank=%u\n",
                    head.n, head.k, head.padded_shape[0], head.padded_shape[1], int(head.qtype),
                    int(head.layout), int(head.group),
                    int(probe.weight(probe.tg_id).n),
                    int(probe.weight(probe.tg_id).k),
                    int(probe.weight(probe.tg_id).qtype),
                    int(probe.weight(probe.tsv_id).n),
                    int(probe.weight(probe.tsv_id).k),
                    int(probe.weight(probe.tsv_id).qtype),
                    probe.config.bridge.rank);
        std::printf("cost: %d timed iterations, %d warmup (module load excluded), batch %d\n",
                    iterations, warmup, kBatch);
        std::fflush(stdout);
        const auto bytes = std::size_t(96ULL << 20);
        auto arena_storage = DeviceBuffer(probe.device, bytes);
        MemoryArena arena(arena_storage.span());
        std::printf("phase: the arena is open, costing K in %s\n",
                    setting("NINFER_BENCH_DRAFTS", "1,3,7,15").c_str());
        std::fflush(stdout);
        for (const auto drafts : ks) {
            if (drafts < 1 || drafts > 15) { continue; }
            cost_k(probe, arena, drafts, iterations, warmup);
            std::printf("arena peak: %.3f MiB of %.3f MiB\n", arena.peak_used() / double(1<<20),
                        bytes / double(1<<20));
            arena.reset_peak();
        }
        return 0;
    } catch (const std::exception& error) {
        std::printf("RED %s\n", error.what());
        std::fflush(stdout);
        return 1;
    }
}
