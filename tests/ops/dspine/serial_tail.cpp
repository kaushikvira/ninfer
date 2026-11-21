// DSpine serial tail - device op test.
//
// The op (ops::dspine_serial_tail) must reproduce two independent oracles:
//   1. the committed reference fixture (Tencent propose.py policy math + the engine's draw
//      stream), i.e. the same vectors the CPU contract test uses, and
//   2. the host core (models/common/dspine/dspine_serial_tail.cpp) entry by entry, which is
//      the executable specification the op mirrors.
// Plus: a CUDA-graph capture/replay path (the op must allocate and synchronize nothing), and
// a teeth case (a wrong inv_temp must disagree), so the comparison is not vacuous.
//
// Skips (exit 77) without a CUDA device. Needs a device with ~4 GiB free: run it with
// CUDA_VISIBLE_DEVICES pointing at the free card.

#include "ops/suite.h"
#include "cuda_fixture.h"
#include "core/buffer.h"
#include "core/decode_graph.h"
#include "ninfer/ops/dspine_bridge.h"
#include "models/common/dspine/dspine_serial_tail.h"
#include "ops/op_tester.h"
#include "dspine_serial_tail/fixture.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
namespace tail = ninfer::models::common::dspine;
namespace fixture = ninfer::test::dspine_serial_tail;

// The host core names the draw-stream purpose numerically to stay CUDA-free; the op uses the
// engine enum. They must be the same stream or a proposal is not replay-reproducible.
static_assert(tail::kSerialTailPurpose ==
                  static_cast<int>(ninfer::ops::kSamplePurposeDFlash2Proposal),
              "the serial tail draw stream drifted from the engine's sampling purpose");

namespace {

constexpr std::int32_t kSlots = 16;
constexpr float kQRelativeTolerance = 1e-5f;
constexpr float kQAbsoluteTolerance = 1e-8f;

float from_bits(std::uint32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

struct Case {
    const char* name;
    std::int32_t drafts;
    std::int32_t batch;
    float temperature;
    std::int32_t top_k;
    float top_p;
    float min_p;
    const std::uint32_t* table;
    const std::int32_t* cand;
    const std::int32_t* drafts_out;
    const std::uint32_t* q;
    const std::int32_t* idx;
    const std::uint64_t* seeds;
    const std::int32_t* positions;
};

#define CASE_FIELDS(name)                                                                       \
    fixture::k##name##Table, fixture::k##name##Cand, fixture::k##name##Drafts,                   \
        fixture::k##name##Q, fixture::k##name##Idx, fixture::k##name##Seed,                       \
        fixture::k##name##Position

#define CASE(name)                                                                              \
    Case {                                                                                      \
        #name, fixture::k##name##DraftsCount, fixture::k##name##Batch,                            \
            fixture::k##name##Temperature, fixture::k##name##TopK, fixture::k##name##TopP,        \
            fixture::k##name##MinP, CASE_FIELDS(name)                                             \
    }

std::vector<Case> cases() {
    return {
        CASE(prod_t09_minp005),  CASE(greedy_t0),      CASE(topk4_t10),
        CASE(topp08_t12),        CASE(minp02_t07),     CASE(lowtemp_t005),
        CASE(prod_batch2_drafts3),
    };
}

struct Result {
    std::vector<std::int32_t> drafts;
    std::vector<float> q;
    std::vector<std::int32_t> idx;
};

// Host core result: the specification the device op is compared against entry by entry.
Result host_result(const Case& test, float temperature_override = NAN) {
    Result result;
    const std::size_t table_size =
        static_cast<std::size_t>(kSlots) * kSlots * test.drafts * test.batch;
    std::vector<float> table(table_size);
    for (std::size_t index = 0; index < table_size; ++index) {
        table[index] = from_bits(test.table[index]);
    }
    tail::SerialTailPolicy policy;
    policy.temperature = std::isnan(temperature_override) ? test.temperature : temperature_override;
    policy.top_k       = test.top_k;
    policy.top_p       = test.top_p;
    policy.min_p       = test.min_p;
    const std::vector<tail::SerialTailPolicy> policies(static_cast<std::size_t>(test.batch),
                                                       policy);
    result.drafts.resize(static_cast<std::size_t>(test.drafts) * test.batch);
    result.q.resize(static_cast<std::size_t>(kSlots) * test.drafts * test.batch);
    result.idx.resize(static_cast<std::size_t>(kSlots) * test.drafts * test.batch);
    tail::serial_tail_run(table.data(), test.cand, kSlots, test.drafts, test.batch,
                          policies.data(), test.seeds, test.positions, result.drafts.data(),
                          result.q.data(), result.idx.data());
    return result;
}

Result device_result(const Case& test, float temperature_override = NAN, bool use_graph = false) {
    const std::size_t table_size =
        static_cast<std::size_t>(kSlots) * kSlots * test.drafts * test.batch;
    std::vector<float> table(table_size);
    for (std::size_t index = 0; index < table_size; ++index) {
        table[index] = from_bits(test.table[index]);
    }
    std::vector<std::int32_t> cand(test.cand,
                                   test.cand + static_cast<std::size_t>(kSlots) * test.drafts *
                                                   test.batch);
    std::vector<ops::SamplingConfig> configs(static_cast<std::size_t>(test.batch));
    for (auto& config : configs) {
        config.temperature = std::isnan(temperature_override) ? test.temperature
                                                              : temperature_override;
        config.top_k       = test.top_k;
        config.top_p       = test.top_p;
        config.min_p       = test.min_p;
    }
    for (std::int32_t b = 0; b < test.batch; ++b) configs[b].seed = test.seeds[b];
    std::vector<std::int32_t> positions(test.positions, test.positions + test.batch);

    DeviceBuffer d_table   = to_device(table);
    DeviceBuffer d_cand    = to_device(cand);
    DeviceBuffer d_configs = to_device(configs);
    DeviceBuffer d_positions = to_device(positions);
    GuardedDeviceBuffer d_drafts(std::size_t(test.batch) * test.drafts * sizeof(std::int32_t));
    GuardedDeviceBuffer d_q(std::size_t(test.batch) * test.drafts * kSlots * sizeof(float));
    GuardedDeviceBuffer d_idx(std::size_t(test.batch) * test.drafts * kSlots * sizeof(std::int32_t));
    d_drafts.fill(0xcd);
    d_q.fill(0xcd);
    d_idx.fill(0xcd);

    // First axis fastest: table [batch, 16 cand, 16 sender, drafts], cand/drafts/q/idx as
    // documented in ninfer/ops/dspine_bridge.h.
    Tensor table_tensor(d_table.span(), DType::FP32, {test.batch, kSlots, kSlots, test.drafts});
    Tensor cand_tensor(d_cand.span(), DType::I32, {test.batch, test.drafts, kSlots});
    Tensor position_tensor(d_positions.span(), DType::I32, {test.batch});
    Tensor drafts_tensor(d_drafts.span(), DType::I32, {test.batch, test.drafts});
    Tensor q_tensor(d_q.span(), DType::FP32, {test.batch, test.drafts, kSlots});
    Tensor idx_tensor(d_idx.span(), DType::I32, {test.batch, test.drafts, kSlots});
    const auto* device_configs =
        static_cast<const ops::SamplingConfig*>(d_configs.data());

    const auto launch = [&](CudaExecutionView cuda) {
        ops::dspine_serial_tail(table_tensor, cand_tensor, device_configs, position_tensor,
                                drafts_tensor, q_tensor, idx_tensor, cuda.stream());
    };
    const auto cuda = cuda_environment().context.execution_view();
    if (use_graph) {
        // Capture and replay: a hidden allocation or a host sync would either fail capture or
        // change the result on the second replay.
        DecodeGraphDefinition definition;
        definition.capture(cuda.stream(), [&] { launch(cuda); });
        DecodeGraphExecutable graph;
        graph.instantiate(definition);
        graph.upload(cuda.stream());
        for (int replay = 0; replay < 3; ++replay) {
            graph.launch(cuda.stream());
            cuda_synchronize();
        }
    } else {
        launch(cuda);
    }
    cuda_synchronize();

    Result result;
    result.drafts = from_device<std::int32_t>(d_drafts.data(), drafts_tensor.numel());
    result.q      = from_device<float>(d_q.data(), q_tensor.numel());
    result.idx    = from_device<std::int32_t>(d_idx.data(), idx_tensor.numel());
    return result;
}

void compare(const char* what, const Result& got, const Result& want, int& failures) {
    int draft_mismatches = 0, idx_mismatches = 0, q_mismatches = 0;
    float worst_q = 0.0f;
    for (std::size_t index = 0; index < got.drafts.size(); ++index) {
        if (got.drafts[index] != want.drafts[index]) ++draft_mismatches;
    }
    for (std::size_t index = 0; index < got.idx.size(); ++index) {
        if (got.idx[index] != want.idx[index]) ++idx_mismatches;
    }
    for (std::size_t index = 0; index < got.q.size(); ++index) {
        const float delta = std::fabs(got.q[index] - want.q[index]);
        worst_q           = std::max(worst_q, delta);
        if (delta > kQAbsoluteTolerance + kQRelativeTolerance * std::fabs(want.q[index])) {
            ++q_mismatches;
        }
    }
    if (draft_mismatches != 0 || idx_mismatches != 0 || q_mismatches != 0) {
        std::printf("  %s: %d draft, %d idx, %d q mismatches (worst q delta %.3g)\n", what,
                    draft_mismatches, idx_mismatches, q_mismatches, worst_q);
        failures += std::max({draft_mismatches, idx_mismatches, q_mismatches});
    }
}

int run_case(const Case& test, bool use_graph) {
    int failures = 0;
    const Result reference = [&] {
        Result result;
        result.drafts.assign(test.drafts_out,
                             test.drafts_out + static_cast<std::size_t>(test.drafts) * test.batch);
        result.idx.assign(test.idx,
                          test.idx + static_cast<std::size_t>(kSlots) * test.drafts * test.batch);
        result.q.reserve(static_cast<std::size_t>(kSlots) * test.drafts * test.batch);
        for (std::size_t index = 0; index < static_cast<std::size_t>(kSlots) * test.drafts *
                                                test.batch;
             ++index) {
            result.q.push_back(from_bits(test.q[index]));
        }
        return result;
    }();
    const Result host   = host_result(test);
    const Result device = device_result(test, NAN, use_graph);
    compare((std::string(test.name) + " device-vs-reference").c_str(), device, reference, failures);
    compare((std::string(test.name) + " device-vs-host-core").c_str(), device, host, failures);
    compare((std::string(test.name) + " host-core-vs-reference").c_str(), host, reference,
            failures);
    return failures;
}

int reference_parity() {
    int failures = 0;
    for (const Case& test : cases()) failures += run_case(test, false);
    return failures;
}

int graph_replay() {
    int failures = 0;
    for (const Case& test : cases()) failures += run_case(test, true);
    return failures;
}

int teeth_wrong_temperature() {
    // Feeding the tail 1/T instead of T (the classic inv_temp mistake) must disagree with the
    // reference on at least one case; otherwise the parity above proves nothing about policy.
    int caught = 0;
    for (const Case& test : cases()) {
        if (test.temperature <= 0.0f) continue;  // greedy rows ignore temperature
        const Result reference = host_result(test);
        const Result mutant    = device_result(test, 1.0f / test.temperature);
        bool differs           = false;
        for (std::size_t index = 0; index < mutant.drafts.size(); ++index) {
            if (mutant.drafts[index] != reference.drafts[index]) differs = true;
        }
        if (differs) ++caught;
    }
    if (caught == 0) {
        std::printf("  no case detected the wrong inv_temp\n");
        return 1;
    }
    return 0;
}

int rejects_bad_geometry() {
    // The entry validates its operands: a 15-wide candidate axis or a mismatched q must throw
    // rather than read out of bounds.
    int failures = 0;
    const std::int32_t batch = 1, drafts = 3;
    std::vector<float> table(static_cast<std::size_t>(kSlots) * kSlots * drafts * batch, 0.0f);
    std::vector<std::int32_t> cand(static_cast<std::size_t>(kSlots) * drafts * batch, 0);
    for (std::int32_t i = 0; i < drafts; ++i) {
        for (std::int32_t k = 0; k < kSlots; ++k) cand[k * drafts + i] = k + i;
    }
    std::vector<std::int32_t> positions(batch, 7);
    std::vector<ops::SamplingConfig> configs(batch);
    DeviceBuffer d_table = to_device(table);
    DeviceBuffer d_cand  = to_device(cand);
    DeviceBuffer d_pos   = to_device(positions);
    DeviceBuffer d_cfg   = to_device(configs);
    GuardedDeviceBuffer d_drafts(drafts * sizeof(std::int32_t));
    GuardedDeviceBuffer d_q(std::size_t(drafts) * kSlots * sizeof(float));
    GuardedDeviceBuffer d_idx(std::size_t(drafts) * kSlots * sizeof(std::int32_t));
    Tensor table_tensor(d_table.span(), DType::FP32, {batch, kSlots, kSlots, drafts});
    Tensor q_tensor(d_q.span(), DType::FP32, {batch, drafts, kSlots});
    Tensor idx_tensor(d_idx.span(), DType::I32, {batch, drafts, kSlots});
    const auto* device_configs = static_cast<const ops::SamplingConfig*>(d_cfg.data());
    const auto expect_throw   = [&](auto&& call, const char* what) {
        try {
            call();
        } catch (const std::exception&) { return; }
        std::printf("  expected a rejection: %s\n", what);
        ++failures;
    };
    expect_throw(
        [&] {
            Tensor narrow(d_cand.span(), DType::I32, {batch, drafts, 15});
            Tensor dft(d_drafts.span(), DType::I32, {batch, drafts});
            ops::dspine_serial_tail(table_tensor, narrow, device_configs,
                                    Tensor(d_pos.span(), DType::I32, {batch}),
                                    dft, q_tensor, idx_tensor, nullptr);
        },
        "15-wide candidate axis");
    expect_throw(
        [&] {
            Tensor dft(d_drafts.span(), DType::I32, {batch, drafts});
            Tensor q15(d_q.span(), DType::FP32, {batch, drafts, 15});
            ops::dspine_serial_tail(table_tensor, Tensor(d_cand.span(), DType::I32, {batch, drafts, kSlots}),
                                    device_configs, Tensor(d_pos.span(), DType::I32, {batch}),
                                    dft, q15, idx_tensor, nullptr);
        },
        "q with a 15-wide candidate axis");
    expect_throw(
        [&] {
            Tensor dft(d_drafts.span(), DType::I32, {batch, drafts});
            ops::dspine_serial_tail(table_tensor, Tensor(d_cand.span(), DType::I32, {batch, drafts, kSlots}),
                                    nullptr, Tensor(d_pos.span(), DType::I32, {batch}),
                                    dft, q_tensor, idx_tensor, nullptr);
        },
        "null sampling configs");
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    OpSuite suite;
    suite.add("dspine_serial_tail/reference_parity", reference_parity);
    suite.add("dspine_serial_tail/graph_replay", graph_replay, {{"execution", "graph"}});
    suite.add("dspine_serial_tail/teeth_wrong_temperature", teeth_wrong_temperature);
    suite.add("dspine_serial_tail/rejects_bad_geometry", rejects_bad_geometry);
    return suite.main(argc, argv);
}
