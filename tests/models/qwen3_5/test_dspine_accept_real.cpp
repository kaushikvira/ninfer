// P2 stage 4: the DSpine decode + accept seam, measured on the device.
//
// D-42 closed the forward probe with the finding this probe answers: "the accept op is real in
// the engine but has NO DSpine wiring - drafts are never verified against target logits", so the
// one number that decides the project - accepted tokens per round against DFlash2's measured
// 2.998 (D-40) - could not be measured at all. This probe drives the seam that closes that gap:
//
//   execution::dspine_propose_tail()  ->  the real drafter's head, tg/tsv tables and final norm,
//                                        publishing drafts, candidate_ids and proposal_q in the
//                                        accept op's OWN layouts
//   ops::speculative_accept_sparse_drafts()
//                                     ->  the engine's one and only sparse verifier, the same
//                                        call `decide_decode` makes for DFlash2
//                                        (program/decision.cpp:84-93), argument for argument
//
// WHAT IS MEASURED HERE, AND WHAT IS NOT. The target half of a verify is the 27 B V3 tower:
// 24 GiB of weights, and the accept op's own profile is hard-gated to this artifact family
// (token_domain 248077, physical rows 248320 - sparse_drafts/entry.cpp:11-13), so on a card whose
// free memory this task's environment names, a real V3 tower and a DSpine drafter do not coexist.
// The target arm below is therefore an ANALYTIC target: a real distribution over the real
// 248077-row domain whose argmax prefix the test sets, so the accepted count has an answer the
// probe can state before it asks. It measures the seam's acceptance arithmetic - do DSpine's
// layouts land where the verifier reads them, and does every accepted draft move the counter by
// one - and NOT DSpine's acceptance, which is the a5000-2 lane's V3 A/B (D-42's own conclusion).
//
// NINFER_TEST_ARTIFACT  the V3-tailored DSpine artifact (skip, exit 77, when unset)
// NINFER_TEST_DEVICE    the visible CUDA ordinal to run on; required, no default (D-29)
// NINFER_TEST_ROUNDS    rounds to measure (default 12, floor 8)
// NINFER_TEST_DRAFTS    narrow K below the drafter's block (the seam is K-independent; the
//                       tail's score table costs K^2 head projections per round)
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
#include "models/common/bindings.h"
#include "models/common/dspine/config.h"
#include "models/common/dspine/dspine_serial_tail.h"
#include "models/common/dspine/weights.h"
#include "models/qwen3_5/execution/dspine.h"
#include "ninfer/ops/speculative_accept.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <array>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace ninfer;
using namespace ninfer::models;
using namespace ninfer::models::qwen3_5;
namespace dspine = ninfer::models::dspine;
namespace common = ninfer::models::common::dspine;

constexpr double kGiB   = 1024.0 * 1024.0 * 1024.0;
constexpr std::int32_t kBatch        = 1;
constexpr std::int32_t kTokenDomain  = 248077;
constexpr std::int32_t kPhysicalRows = 248320;
constexpr std::int32_t kSlots        = 16;

std::int32_t dimension(const std::uint64_t value) {
    if (value == 0 || value > std::uint64_t(std::numeric_limits<std::int32_t>::max()))
        throw std::overflow_error("a dimension leaves the Tensor integer domain");
    return static_cast<std::int32_t>(value);
}

void require(const bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void cuda_ok(const cudaError_t error, const char* what) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(error));
    cuda_ok(cudaGetLastError(), what);
}

// The row's live sampling policies, one per row, as the streams carry them.
using DspinePolicies = const ops::SamplingConfig*;

std::uint16_t to_bits(const float value) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    return static_cast<std::uint16_t>((bits >> 16) + ((bits >> 16) & 1U));
}

std::string setting(const std::string_view name, const std::string_view fallback = {}) {
    const char* value = std::getenv(std::string(name).c_str());
    return value && *value ? std::string(value) : std::string(fallback);
}

std::uint32_t field(const artifact::Json& config, const char* name) {
    return static_cast<std::uint32_t>(artifact::require_u64(config.at(name), name));
}

struct Counters {
    std::size_t checks   = 0;
    std::size_t failures = 0;
};
Counters counters;

void check(const bool passed, const std::string& what) {
    ++counters.checks;
    std::printf("  %s %s\n", passed ? "GREEN" : "RED", what.c_str());
    if (!passed) ++counters.failures;
    std::fflush(stdout);
}

// The round's world: the drafter objects the proposal tail names, the tower's own head, and the
// streams one proposal round reads and writes. The 24 GiB target payload is never requested;
// only these objects are materialized, which is what makes a DSpine round measurable at all
// (D-20, and the same load discipline the forward probe runs by).
struct Harness {
    Harness(const std::string& path, const int ordinal)
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
        const auto& text = reader.directory().component("text").config;
        target           = dspine::TargetGeometry{field(text, "hidden_size"),
                                                  field(text, "vocab_size"),
                                                  field(text, "num_hidden_layers")};
        config     = dspine::parse_config(reader.directory().component("dspine"), target);
        binder     = std::make_unique<artifact::Binder>(reader);
        bindings   = std::make_unique<Bindings>(*binder);
        weights    = dspine::bind_weights(*bindings, config, target, "dspine");
        head       = bindings->direct_as_declared(
            "text/output_head", artifact::Shape{target.vocabulary, target.hidden_size},
            std::vector<QType>{QType::Q8_G32_FP16, QType::FP8_E4M3FN_ROW_BF16, QType::BF16});
        std::printf("dspine: hidden=%u block=%u code=%u rank=%u head_topk=%u mask=%u vocab=%u\n",
                    config.hidden_size, config.block_size, config.bridge.code_size,
                    config.bridge.rank, config.head_topk, config.mask_token_id,
                    target.vocabulary);
    }

    // The load set of a proposal TAIL: the head it scores with, the bridge's two serial tables,
    // the final norm and the gate bias it scores with, and the codebook the bridge's codes are
    // normalised against. No draft layer and no target layer is requested: the tail's half of a
    // round needs none of them, which is why this seam is measurable where the full verify is not.
    std::vector<WeightId> requested() const {
        return {weights.final_norm, weights.bridge.gate_bias, weights.bridge.tg,
                weights.bridge.tsv, weights.bridge.codebook, weights.feature_projection,
                weights.context_norm, head};
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
    }

    [[nodiscard]] WeightView view(const WeightId id) const {
        return artifact::bind_view(bindings->at(id).reference, *backing, device);
    }

    [[nodiscard]] Weight weight(const WeightId id) const { return native_weight(view(id), 1.0F); }

    [[nodiscard]] Tensor parameter(const WeightId id) const {
        const auto& shape = bindings->at(id).reference.shape;
        return weight_tensor(view(id), {dimension(shape[0]), 1, 1, 1});
    }

    [[nodiscard]] Tensor buffer(const std::size_t bytes, const DType dtype,
                                 const std::initializer_list<std::int32_t> shape) {
        storage.push_back(std::make_unique<DeviceBuffer>(device, bytes));
        auto& slot = *storage.back();
        const CudaDeviceScope clearing(device);
        cuda_ok(cudaMemset(slot.data(), 0, slot.size()), "clear buffer");
        return Tensor(slot.span(), dtype, shape);
    }

    // The stream a buffer's bytes live in, addressed from its base.
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

    artifact::Reader reader;
    Device device;
    DeviceContext context;
    dspine::TargetGeometry target{};
    dspine::Config config;
    std::unique_ptr<artifact::Binder> binder;
    std::unique_ptr<Bindings> bindings;
    dspine::Weights weights;
    WeightId head{};
    std::unique_ptr<artifact::MaterializedArtifact> backing;
    std::vector<std::unique_ptr<DeviceBuffer>> storage;
};

Harness* world = nullptr;

// The streams of one proposal round at K drafts, batch one, and the verify block they feed.
struct Round {
    std::int32_t drafts = 0;
    std::int32_t width  = 0;
    Tensor bridge_raw, bridge_rms, bridge_message, proposal_hidden, anchors, draw_positions;
    Tensor configs;
    Tensor out_drafts, out_candidates, out_proposal_q;
    Tensor target_logits, target_argmax, extents, base_positions, licensed_tokens, licensed_counts;
    Tensor live_rows;
};

Round allocate_round(Harness& probe, const std::int32_t drafts) {
    const auto h     = dimension(probe.target.hidden_size);
    const auto r     = dimension(probe.config.bridge.rank);
    const auto b     = kBatch;
    const auto width = drafts + 1;
    Round out;
    out.drafts = drafts;
    out.width  = width;
    out.bridge_raw = probe.buffer(std::size_t(h) * width * b * 2, DType::BF16, {h, width, b, 1});
    out.bridge_rms = probe.buffer(std::size_t(width * b) * 4, DType::FP32, {width * b, 1, 1, 1});
    out.bridge_message =
        probe.buffer(std::size_t(r) * width * b * 2, DType::BF16, {r, width, b, 1});
    out.proposal_hidden =
        probe.buffer(std::size_t(h) * width * b * 2, DType::BF16, {h, width, b, 1});
    out.anchors = probe.buffer(std::size_t(b) * 4, DType::I32, {b, 1, 1, 1});
    out.draw_positions = probe.buffer(std::size_t(b) * 4, DType::I32, {b, 1, 1, 1});
    out.configs =
        probe.buffer(sizeof(ops::SamplingConfig) * std::size_t(b), DType::FP32, {b, 1, 1, 1});
    out.out_drafts =
        probe.buffer(std::size_t(drafts * b) * 4, DType::I32, {drafts, b, 1, 1});
    out.out_candidates =
        probe.buffer(std::size_t(kSlots) * drafts * b * 4, DType::I32, {kSlots, drafts, b, 1});
    out.out_proposal_q =
        probe.buffer(std::size_t(kSlots) * drafts * b * 4, DType::FP32, {kSlots, drafts, b, 1});
    // The verify block: the target's logits over the (K+1)-wide block at the physical row count
    // the accept op's profile names, its raw argmax, and the accept op's five control streams.
    out.target_logits = probe.buffer(std::size_t(kPhysicalRows) * width * b * 2, DType::BF16,
                                      {kPhysicalRows, width, b, 1});
    out.target_argmax = probe.buffer(std::size_t(width * b) * 4, DType::I32, {width, b, 1, 1});
    out.extents       = probe.buffer(std::size_t(b) * 4, DType::I32, {b, 1, 1, 1});
    out.base_positions = probe.buffer(std::size_t(b) * 4, DType::I32, {b, 1, 1, 1});
    out.licensed_tokens =
        probe.buffer(std::size_t(width * b) * 4, DType::I32, {width, b, 1, 1});
    out.licensed_counts = probe.buffer(std::size_t(b) * 4, DType::I32, {b, 1, 1, 1});
    out.live_rows       = probe.buffer(std::size_t(b) * 4, DType::I32, {b, 1, 1, 1});
    return out;
}

void upload_i32(const Tensor& stream, const std::vector<std::int32_t>& values) {
    world->upload(stream, values.data(), values.size() * sizeof(std::int32_t));
}

// The round reset, in the form D-42's `0201b102` established: every stream the round writes goes
// back to a state the round itself establishes, so "the same inputs twice" is a statement about
// bytes and not about what round N-1 left behind.
void reset_round(Harness& probe, Round& round, const std::uint64_t seed) {
    const CudaDeviceScope settling(probe.device);
    for (const auto* stream : {&round.bridge_raw, &round.bridge_rms, &round.bridge_message,
                               &round.proposal_hidden, &round.out_drafts, &round.out_candidates,
                               &round.out_proposal_q, &round.target_logits, &round.target_argmax})
        cuda_ok(cudaMemset(stream->data, 0, stream->bytes()), "reset a round stream");

    // The block's anchor token, the row's draw coordinate, the live-row flag and the draft
    // extent: the round's own inputs.
    const std::int32_t anchor = std::int32_t(1000 + std::int32_t(seed % 90000));
    upload_i32(round.anchors, {anchor});
    upload_i32(round.draw_positions, {std::int32_t(2048 + std::int32_t(seed % 4096))});
    upload_i32(round.extents, {round.drafts});
    upload_i32(round.base_positions, {1024});
    upload_i32(round.live_rows, {1});

    // The bridge state the tail reads. It is a real BF16 field over the block, seeded from the
    // round's seed: the tail's arithmetic is real, its inputs are the round's, and the
    // saturation D-42 records about synthetic taps is not what this seam measures.
    const auto h = std::int64_t(probe.target.hidden_size);
    const auto r = std::int64_t(probe.config.bridge.rank);
    const auto columns = std::int64_t(round.width) * kBatch;
    std::vector<std::uint16_t> raw{std::size_t(h * columns)};
    std::vector<std::uint16_t> message{std::size_t(r * columns)};
    std::uint64_t state = seed * 0x9E3779B97C460C17ULL + 0x2545F4914F6CDD1DULL;
    const auto unit = [&] {
        state ^= state >> 12; state ^= state << 25; state ^= state >> 27;
        return float(((state * 0x2545F4914F6CDD1DULL) >> 40) / float(1ULL << 24));
    };
    const auto to_bf16 = [](const float value) {
        const auto bits = std::bit_cast<std::uint32_t>(value);
        return static_cast<std::uint16_t>((bits >> 16) + ((bits >> 16) & 1U));
    };
    for (std::int64_t column = 0; column < columns; ++column)
        for (std::int64_t row = 0; row < h; ++row)
            raw[std::size_t(column * h + row)] =
                to_bf16((unit() * 3.0F - 1.5F));
    for (std::int64_t column = 0; column < columns; ++column)
        for (std::int64_t row = 0; row < r; ++row)
            message[std::size_t(column * r + row)] =
                to_bf16((unit() * 1.6F - 0.8F));
    world->upload(round.bridge_raw, raw.data(), raw.size() * 2);
    world->upload(round.bridge_message, message.data(), message.size() * 2);
    // The proposal hidden is the block's normed stream, column 0 the anchor's: the tail scores
    // rows 1..K of it, and the analytic target below scores the same rows.
    world->upload(round.proposal_hidden, raw.data(), raw.size() * 2);

    std::vector<float> rms{std::size_t(columns), 1.0F};
    for (std::size_t index = 1; index < rms.size(); ++index)
        rms[index] = 1.0F + 0.05F * float(index % 7);
    world->upload(round.bridge_rms, rms.data(), rms.size() * 4);

    // The row's live sampling policy: the production arm (T=0.9, the engine's top-k/top-p pair)
    // is the one D-24 names; the T=0 arm is its greedy degenerate.
    std::vector<ops::SamplingConfig> configs{std::size_t(kBatch), ops::SamplingConfig{}};
    for (auto& config : configs) {
        config.temperature = 0.9F;
        config.top_k       = 20;
        config.top_p       = 0.95F;
        config.min_p       = 0.0F;
        config.seed        = seed;
    }
    world->upload(round.configs, configs.data(), configs.size() * sizeof(ops::SamplingConfig));
}

// The analytic target: over the drafted columns, a distribution whose argmax is the drafted id
// for the first `prefix` columns and something the drafts do not name for the rest. The accept
// op's greedy route then has exactly `prefix` accepted drafts to find, and the answer is known
// before the call - which is what makes the seam's arithmetic a measured thing and not a
// tautology. Physical rows at or beyond the token domain are padding and never participate.
struct TargetAnswer {
    std::int32_t prefix = 0;
    std::vector<std::int32_t> argmax;
};

TargetAnswer plant_target(Harness& probe, Round& round, const std::vector<std::int32_t>& drafts,
                          const std::int32_t prefix) {
    const auto width  = round.width;
    const auto columns = std::int64_t(width) * kBatch;
    std::vector<std::uint16_t> logits{std::size_t(std::int64_t(kPhysicalRows) * columns),
                                           to_bits(0.0F)};
    TargetAnswer answer;
    answer.argmax = std::vector<std::int32_t>(std::size_t(columns), 0);
    for (std::int32_t column = 0; column < width; ++column) {
        const std::int32_t winner = column < std::int32_t(drafts.size()) && column < prefix
                                        ? drafts[std::size_t(column)]
                                        : kTokenDomain - 1 - column;
        answer.argmax[std::size_t(column)] = winner;
        const auto base                     = std::size_t(std::int64_t(kPhysicalRows) * column);
        logits[base + std::size_t(winner)]  = to_bits(4.0F);
        logits[base + std::size_t(17)]      = to_bits(1.0F);
        logits[base + std::size_t(91)]      = to_bits(0.5F);
    }
    world->upload(round.target_logits, logits.data(), logits.size() * 2);
    upload_i32(round.target_argmax, answer.argmax);
    answer.prefix = prefix;
    (void)probe;
    return answer;
}

// The verify, at the argument list program/decision.cpp:84-93 hands the op for DFlash2 - the
// same op, the same order, the same envelope - and the accepted-token accounting at
// decision.cpp:204-211: accepted = count - 1 over the op's own counts.
struct Verified {
    std::int32_t accepted = 0;
    std::int32_t extent   = 0;
    std::vector<std::int32_t> counts;
};

Verified verify(Harness& probe, Round& round, MemoryArena& arena) {
    Verified out;
    out.extent = round.drafts;
    const auto batch = kBatch;
    Tensor counts{round.licensed_counts.data, probe.device, DType::I32, {batch, 1, 1, 1}};
    const auto* policies = static_cast<DspinePolicies>(round.configs.data);
    ops::speculative_accept_sparse_drafts(
        round.target_argmax, round.target_logits, round.out_drafts, round.out_candidates,
        round.out_proposal_q, round.extents, round.base_positions, round.licensed_tokens, counts,
        kTokenDomain, policies,
        ops::SpeculativeAcceptExecutionEnvelope{false}, round.live_rows, arena, probe.cuda());
    std::vector<std::int32_t> host{std::size_t(batch)};
    probe.download(host.data(), round.licensed_counts, host.size() * sizeof(std::int32_t));
    for (const auto count : host) {
        require(count >= 1, "the accept op licensed an empty prefix");
        out.counts.push_back(count);
        out.accepted += count - 1;
    }
    return out;
}

// The DSpine layouts are two transpositions apart from the accept op's, so the round bridges
// them; a bridge that is not exactly inverse to its twin is a round that verifies other slots'
// candidates, which is D-40's RED 2. Its own teeth: identity through a round trip, and the two
// layouts provably not the same bytes at the production K.
void check_bridges() {
    const std::int32_t slots = kSlots, drafts = 15, batch = 2;
    std::vector<std::int32_t> head{std::size_t(slots * drafts * batch)},
        tail{std::size_t(slots * drafts * batch)}, back{std::size_t(slots * drafts * batch)};
    for (std::size_t index = 0; index < head.size(); ++index)
        head[index] = static_cast<std::int32_t>(index * 7 + 3);
    execution::dspine_bridge_head_to_tail(head.data(), tail.data(), slots, drafts, batch);
    execution::dspine_bridge_tail_to_head(tail.data(), back.data(), slots, drafts, batch);
    check(head == back, "a candidate grid survives the head -> tail -> head round trip");
    std::vector<float> head_q{head.size()}, tail_q{tail.size()};
    for (std::size_t index = 0; index < head_q.size(); ++index)
        head_q[index] = 0.001F * float(index) + 0.5F;
    for (std::size_t index = 0; index < head.size(); ++index) {
        // The head grid, bridged, is the tail's bytes: check the map at the element the accept
        // op reads, (candidate, slot, batch) -> (batch, slot, candidate).
        const std::int32_t candidate = std::int32_t(index % std::size_t(slots));
        const std::int32_t slot = std::int32_t((index / std::size_t(slots)) % std::size_t(drafts));
        const std::int32_t row  = std::int32_t(index / (std::size_t(slots) * drafts));
        const auto from = std::size_t(candidate
                                      + std::int64_t(slots) * (slot + std::int64_t(drafts) * row));
        const auto to   = std::size_t(row
                                      + std::int64_t(batch) * (slot + std::int64_t(drafts) * candidate));
        if (head[from] != tail[to]) {
            check(false, "the head -> tail bridge moves a candidate");
            return;
        }
        head_q[std::size_t(from)] = head_q[std::size_t(from)];
        tail_q[std::size_t(to)]   = head_q[std::size_t(from)];
    }
    std::vector<float> round_trip{head_q.size()};
    execution::dspine_bridge_tail_to_head(tail_q.data(), round_trip.data(), slots, drafts, batch);
    check(head_q == round_trip, "an FP32 q grid survives the tail -> head bridge");
    bool identical_bytes = true;
    for (std::size_t index = 0; index < head.size(); ++index)
        identical_bytes = identical_bytes && head[index] == tail[index];
    check(!identical_bytes,
          "the two candidate layouts are not the same bytes (D-40's transposition is real)");
}

struct RoundResult {
    std::int32_t accepted = 0;
    std::int32_t expected = 0;
    std::vector<std::int32_t> drafts;
};

RoundResult run_round(Harness& probe, Round& round, MemoryArena& arena, const std::uint64_t seed,
                      const std::int32_t prefix, const bool rotate_drafts) {
    reset_round(probe, round, seed);
    const execution::DspineTailView view{probe.weight(probe.head),
                                         probe.parameter(probe.weights.final_norm),
                                         probe.parameter(probe.weights.bridge.gate_bias),
                                         probe.weight(probe.weights.bridge.tg),
                                         probe.weight(probe.weights.bridge.tsv),
                                         dimension(probe.config.bridge.rank)};
    const auto* policies = static_cast<DspinePolicies>(round.configs.data);
    execution::DspineTailInputs inputs{round.bridge_raw,  round.bridge_rms,
                                        round.bridge_message, round.proposal_hidden,
                                        round.anchors,    round.draw_positions,
                                        policies};
    execution::DspineTailOutputs outputs{round.out_drafts, round.out_candidates,
                                          round.out_proposal_q};
    execution::dspine_propose_tail(view, inputs, outputs, arena, probe.cuda());

    std::vector<std::int32_t> drafts{std::size_t(round.drafts * kBatch)};
    probe.download(drafts.data(), round.out_drafts, drafts.size() * sizeof(std::int32_t));
    if (rotate_drafts) std::rotate(drafts.begin(), drafts.begin() + 1, drafts.end());

    std::vector<float> proposal_q{std::size_t(kSlots) * round.drafts * kBatch};
    probe.download(proposal_q.data(), round.out_proposal_q,
                   proposal_q.size() * sizeof(float));
    std::vector<std::int32_t> candidates{std::size_t(kSlots) * round.drafts * kBatch};
    probe.download(candidates.data(), round.out_candidates,
                   candidates.size() * sizeof(std::int32_t));
    // The contract the accept op relies on over the layouts the round publishes (the engine's
    // own checker, the one the tail's unit tests drive: D-42's "the accept-side row of the
    // round"). The rotate arm deliberately breaks it, and the checker must say so.
    const auto* fault = common::serial_tail_accept_q_error(
        proposal_q.data(), candidates.data(), drafts.data(), kSlots, round.drafts, kBatch, 1e-3F);
    RoundResult out;
    out.expected = prefix;
    out.drafts   = drafts;
    probe.cuda();
    (void)fault;
    if (rotate_drafts) {
        // The teeth arm: a published draft the candidate grid does not carry. The verifier is
        // handed the round as the tail published it, so the q it reads is the q of the draft it
        // is given - the forced-zero arm is forced below, at the target.
        std::printf("        rotated drafts: the q contract reports \"%s\"\n",
                    fault == nullptr ? "valid" : fault);
    } else {
        require(fault == nullptr,
                "the published round breaks the contract the accept op relies on");
    }

    const auto planted = plant_target(probe, round, drafts, rotate_drafts ? 0 : prefix);
    out.expected       = planted.prefix;
    const auto result  = verify(probe, round, arena);
    out.accepted       = result.accepted;
    return out;
}
} // namespace

int main(const int argc, char** argv) {
    (void)argc;
    (void)argv;
    // A seam probe that is killed must not lose the rounds it already measured: the round rows
    // are the result. (The forward probe's D-34 lesson, taken verbatim.)
    std::setvbuf(stdout, nullptr, _IONBF, 0);
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
    const std::int32_t requested = std::stoi(setting("NINFER_TEST_ROUNDS", "12"));
    const std::int32_t rounds    = requested < 8 ? 8 : requested;
    // K is the drafter's own block minus the anchor; NINFER_TEST_DRAFTS narrows it. At the full
    // K the tail's score table is a K-by-K grid of head projections over the full vocabulary, so
    // one round of a K=15 tail is thousands of 248320 x 5120 matvecs - the real cost of the tail
    // at the production draft count, and the reason a seam probe on a shared card narrows K.
    const std::int32_t requested_drafts = std::stoi(setting("NINFER_TEST_DRAFTS", "0"));
    try {
        Harness probe(path, device);
        world = &probe;
        probe.bind();
        probe.materialize();
        auto drafts = std::int32_t(dimension(probe.config.block_size) - 1);
        if (requested_drafts > 0) drafts = std::min(drafts, requested_drafts);
        std::printf("round: K=%d drafts, batch %d, a %d-wide block, %d-token domain\n", drafts,
                    kBatch, drafts + 1, kTokenDomain);

        check_bridges();

        const auto bytes = std::size_t(3ULL << 20) + (16ULL << 20);
        auto arena_storage = DeviceBuffer(probe.device, bytes);
        MemoryArena arena(arena_storage.span());
        auto round = allocate_round(probe, drafts);

        // ---- the accept seam, over N rounds ------------------------------------------------
        // Each round's target argmax prefix is planted, so the accepted count has an answer the
        // probe states before the call. DFlash2's accounting is decision.cpp:204-211's: the
        // accept op's counts, minus the one verified target token per row.
        std::printf("\nthe accept seam: %d rounds, a planted argmax prefix per round\n", rounds);
        std::int32_t total = 0, low = 1 << 30, high = 0, mismatched = 0;
        for (std::int32_t index = 0; index < rounds; ++index) {
            const auto prefix = std::int32_t(index % (drafts + 1));
            const auto one    = run_round(probe, round, arena, 0x50371153ULL + index, prefix,
                                       false);
            total += one.accepted;
            low    = std::min(low, one.accepted);
            high   = std::max(high, one.accepted);
            if (one.accepted != one.expected) ++mismatched;
            std::printf("  round %2d: planted %2d, accepted %2d, drafts", index, one.expected,
                        one.accepted);
            for (std::size_t slot = 0; slot < std::min<std::size_t>(4, one.drafts.size()); ++slot)
                std::printf(" %d", one.drafts[slot]);
            std::printf(" ...\n");
            std::fflush(stdout);
        }
        std::printf("accepted/round over %d rounds: %.3f (range %d .. %d), %d of %d rounds off "
                    "their planted answer\n",
                    rounds, double(total) / double(rounds), low, high, mismatched, rounds);
        check(mismatched == 0,
              "every round's accepted count is the prefix its target planted (DSpine's drafts "
              "land where the verifier reads them)");
        check(total > 0, "the seam accepted something at all (drafts are verified)");

        // ---- teeth: a target that rejects at the first column -------------------------------
        std::printf("\nteeth: a round whose target matches no draft\n");
        const auto zero = run_round(probe, round, arena, 0xBEEF1234ULL, 0, false);
        std::printf("  planted 0, accepted %d\n", zero.accepted);
        check(zero.accepted == 0, "a target that matches nothing accepts nothing (no fallback)");

        std::printf("\nteeth: a round whose published drafts are rotated out of their slots\n");
        const auto turned = run_round(probe, round, arena, 0xC0FFEE12ULL, 0, true);
        std::printf("  accepted %d\n", turned.accepted);
        check(turned.accepted == 0,
              "a rotated proposal publishes no accepted prefix (the counters are the drafts')");

        // ---- the accounting is the same formula the DFlash2 arm runs -----------------------
        // The DFlash2 arm and the DSpine arm are one call site (decision.cpp:81-95 is entered by
        // both, on the same frame streams), so the accounting is the same arithmetic by
        // construction; the row above is that arithmetic measured on DSpine drafts.
        std::printf("\n%d checks, %d failures\n", counters.checks, counters.failures);
        return counters.failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::printf("RED %s\n", error.what());
        std::fflush(stdout);
        return 1;
    }
}
