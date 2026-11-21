#include "ops/linear/projection/nvfp4/nvfp4_shapes.h"
#include "ops/linear/projection/nvfp4/nvfp4_launch.cuh"
#include "ops/linear/kernels/nvfp4/nvfp4_instances.cuh"

namespace ninfer::ops::detail {
namespace {
// The DSpine draft layer's tap feature projection: 5 target taps x 5120 hidden = 25600 columns.
using Geometry = Nvfp4Geometry<5120, 25600>;
using Gemv =
    Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>;
using C2      = Nvfp4A16SimtSchedule<4, 1, 2, 16, 2, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                     Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                     Nvfp4SimtBlockOrder::RowsContiguous, 1>;
using C4      = Nvfp4A16SimtSchedule<4, 1, 2, 16, 4, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                     Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                     Nvfp4SimtBlockOrder::RowsContiguous, 1>;
using C5      = Nvfp4A16SimtSchedule<4, 1, 2, 16, 5, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                     Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                     Nvfp4SimtBlockOrder::RowsContiguous, 1>;
using T32R64  = Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>;
using T32R128 = Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>;
using T64R128 = Nvfp4A4MmaSchedule<64, 128, 256, 4, 2, 2, 1>;
using T128R128Pipelined = Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 2, 1>;
using T128R128Resident  = Nvfp4A4MmaSchedule<128, 128, 256, 4, 2, 1, 2>;

Nvfp4Launch select_a16(int tokens) {
    if (tokens == 1) return nvfp4_linear_a16_gemv<Geometry, Gemv>;
    if (tokens <= 2) return nvfp4_linear_a16_simt<Geometry, 2, C2, true>;
    if (tokens <= 4) return nvfp4_linear_a16_simt<Geometry, 4, C4, false>;
    if (tokens == 5) return nvfp4_linear_a16_simt<Geometry, 5, C5, true>;
    if (tokens <= 8) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<8, 4, 2>>;
    if (tokens <= 16) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<16, 4, 2>>;
    if (tokens <= 24) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<32, 4, 2>>;
    if (tokens <= 32) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<32, 4, 1>>;
    if (tokens <= 48) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<32, 4, 1>>;
    if (tokens <= 64) return nvfp4_linear_a16_sliced_k<Geometry, Nvfp4SlicedInstance<64, 2, 2>>;
    if (tokens <= 128)
        return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<32, 64, 128, 32, 16, 2, 2>>;
    return nvfp4_linear_a16_mma<Geometry, Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>;
}

void launch_a16(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    select_a16(x.ne[1])(x, w, y, stream);
}

Nvfp4A4Route select_a4(std::int32_t tokens) {
    // No TMA branch, and the interval below is capped by uses_a4: on this card the A4 contraction
    // at this K measures a 34 % error from T=1024 up (nvfp4/5120x25600/s717/.../a4/t1024 and /t1025,
    // one sampled coordinate, deterministic), while the same ladder measures clean at 1023 and for
    // the K=6144, K=17408 and split K=5120 shapes at 1024 and 1025. 100 K-tiles is the widest A4
    // problem in the table; the defect is in that route at that width, not in this geometry, so the
    // route is bounded here and the geometry is not registered as a TMA problem. Above the cap the
    // dispatch takes the A16 route, which measures clean at 1024 (nvfp4/5120x25600/s713/.../t1024).
    if (tokens <= 64) return nvfp4_a4_mma_route<Geometry, T32R64>();
    if (tokens <= 128) return nvfp4_a4_mma_route<Geometry, T32R128>();
    if (tokens <= 192) return nvfp4_a4_mma_route<Geometry, T64R128>();
    if (tokens <= 384) return nvfp4_a4_mma_route<Geometry, T128R128Resident>();
    if (tokens <= 512) return nvfp4_a4_mma_route<Geometry, T128R128Pipelined>();
    return nvfp4_a4_mma_route<Geometry, T128R128Resident>();
}

bool uses_a4(std::int32_t, std::int32_t max_tokens) {
    return max_tokens >= 8 && max_tokens <= 1023;
}
} // namespace

const Nvfp4LinearShape kNvfp4N5120K25600{5120, 25600, launch_a16, launch_nvfp4_a4<select_a4>,
                                         uses_a4};
} // namespace ninfer::ops::detail
