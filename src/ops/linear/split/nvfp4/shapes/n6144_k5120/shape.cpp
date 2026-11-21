#include "ops/linear/split/nvfp4/shapes/n6144_k5120/launch.h"

namespace ninfer::ops::detail::linear_split::nvfp4_n6144_k5120 {
namespace {
bool quantized(LinearPolicy policy, int columns) { return allows_a4(policy) && columns >= 4; }

std::size_t capacity(LinearPolicy policy, std::int32_t, std::int32_t last) {
    return quantized(policy, last) ? nvfp4_a4_workspace_capacity_bytes(last, 5120) : 0;
}

void launch(const Tensor& x, Weights weights, std::span<const Tensor> outputs,
            MemoryArena* workspace, cudaStream_t stream) {
    if (!quantized(weights.policy, x.ne[1])) {
        launch_a16(x, weights.first, outputs, stream);
        return;
    }
    const auto scratch = allocate_nvfp4_a4_workspace(*workspace, x.ne[1], 5120);
    launch_a4(x, weights.first, outputs, scratch, stream);
}
} // namespace
} // namespace ninfer::ops::detail::linear_split::nvfp4_n6144_k5120

namespace ninfer::ops::detail::linear_split {
const SplitShape kNvfp4SplitN6144K5120{6144, 5120, kSplitRows<4096, 1024, 1024>,
                                       nvfp4_n6144_k5120::launch, nvfp4_n6144_k5120::capacity};
} // namespace ninfer::ops::detail::linear_split
