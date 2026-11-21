// The DSpine draft forward's linear geometries, read out of the compiled binary on the host.
//
// A linear family resolves (representation, rows, columns, token interval) against a registered
// shape table and refuses an uncompiled geometry on the host, before any launch. The proposal path
// of the DSpine drafter names nine of them; the forward probe (tests/models/qwen3_5) measures them
// on a real artifact, and this is the same gate with the artifact's numbers folded in, so it needs
// no device and no 26 GiB file to answer "can this step be issued at all".
//
// The numbers are the drafter release's own (dspine-b16-260a6c23/config.json, hidden 5120,
// 32 x 128 query, 8 x 128 key/value, 5 taps, code 1024, rank 512, 248320 vocab), i.e. the geometries
// the loader binds:
//
//   feature projection        [5120, 25600] nvfp4  (5 taps x 5120 columns)
//   attention q/k/v, packed   [6144,  5120] nvfp4  split into 4096 + 1024 + 1024
//   attention output          [5120,  4096] nvfp4
//   mlp gate/up, packed      [34816,  5120] nvfp4  (2 x 17408)
//   mlp down                  [5120, 17408] nvfp4
//   bridge read/gate/write   [1024,5120] [512,1024] [5120,512] bf16
//   proposal head           [248320,  5120] q8_g32_fp16
//
// T is both a context append (1) and a whole draft block (16, the release's block_size).
//
// Every line must read COMPILED: a REFUSED line is a proposal step that cannot be issued, and the
// prod tower's own geometries are in this list, so a regression in the shape tables fails here too.
//
// The 5090/SM120 build is the one that carries these tables (NINFER_TARGET=rtx5090); the shape
// lookup itself is target-independent for the geometries named here.

#include "core/platform.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_split.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/linear_topk.h"
#include "ninfer/ops/projection_weights.h"

#include <cstdio>
#include <exception>
#include <initializer_list>
#include <string>
#include <vector>

namespace {

using namespace ninfer;

struct Request {
    std::string_view family;
    std::string_view label;
    QType qtype;
    std::int32_t rows, columns;
    std::initializer_list<std::int32_t> segments{};  // non-empty = the split's output segmentation
};

int failures = 0;
int steps    = 0;

void probe(const Request& request, const std::int32_t tokens) {
    const auto policy = ops::LinearPolicy::A16Only;
    const auto attempt = [&]() -> std::size_t {
        if (request.family == "linear")
            return ops::linear_workspace_capacity_bytes(
                request.qtype, request.rows, request.columns, policy, tokens, tokens);
        if (request.family == "linear_add")
            return ops::linear_add_workspace_capacity_bytes(
                request.qtype, request.rows, request.columns, policy, tokens, tokens);
        if (request.family == "linear_swiglu")
            return ops::linear_swiglu_workspace_capacity_bytes(
                request.qtype, request.rows, request.columns, policy, tokens, tokens);
        if (request.family == "linear_topk")
            return ops::linear_topk_workspace_capacity_bytes(
                request.qtype, request.rows, request.columns, tokens, tokens);
        const ops::ProjectionProfile profile{
            ops::LinearProfile{
                request.qtype, request.rows, request.columns, policy, QuantLayout::Contiguous},
            {}};
        return ops::linear_split_workspace_capacity_bytes(
            profile, std::vector<std::int32_t>(request.segments), tokens, tokens);
    };
    const auto label = std::string(request.label) + " [" + std::to_string(request.rows) + ", " +
                       std::to_string(request.columns) + "]";
    ++steps;
    try {
        const auto bytes = attempt();
        std::printf("  COMPILED T=%-3d %s (workspace %zu B)\n", tokens, label.c_str(), bytes);
    } catch (const std::exception& error) {
        ++failures;
        std::printf("  REFUSED  T=%-3d %s : %s\n", tokens, label.c_str(), error.what());
    }
}

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const std::vector<Request> requests{
        {"linear", "feature projection (the taps)", QType::NVFP4, 5120, 25600, {}},
        {"linear_split", "attention query/key/value, packed", QType::NVFP4, 6144, 5120,
         {4096, 1024, 1024}},
        {"linear", "attention output", QType::NVFP4, 5120, 4096, {}},
        {"linear_swiglu", "mlp gate/up, packed", QType::NVFP4, 34816, 5120, {}},
        {"linear_add", "mlp down", QType::NVFP4, 5120, 17408, {}},
        {"linear", "bridge read", QType::BF16, 1024, 5120, {}},
        {"linear", "bridge gate", QType::BF16, 512, 1024, {}},
        {"linear", "bridge write", QType::BF16, 5120, 512, {}},
        {"linear_topk", "proposal head", QType::Q8_G32_FP16, 248320, 5120, {}},
    };
    constexpr std::string_view target =
        NINFER_TARGET_RTX5090 ? "rtx5090" : NINFER_TARGET_SPARK ? "dgx_spark" : "generic";
    std::printf("the linear geometries the DSpine proposal path names (target %s):\n",
                std::string(target).c_str());
    for (const std::int32_t tokens : {1, 16})
        for (const auto& request : requests) probe(request, tokens);
    std::printf("dspine shape gate: %d lookups, %d refused\n", steps, failures);
    return failures == 0 ? 0 : 1;
}
