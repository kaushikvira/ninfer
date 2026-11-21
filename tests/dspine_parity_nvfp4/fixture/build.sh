#!/bin/bash
# Build the P2 NVFP4 parity fixture against THIS tree's build (read-only).
# Requires: g++ (c++), CUDA headers, build/<preset>/lib/{libninfer_ops,libninfer_core}.so
set -euo pipefail

TREE=${TREE:-$(cd "$(dirname "$0")/../../.." && pwd)}
BUILD=${BUILD:-$TREE/build/rtx5090-ops}
CUDA=${CUDA:-/usr/local/cuda}
HERE=$(cd "$(dirname "$0")" && pwd)

if [[ ! -f "$BUILD/lib/libninfer_ops.so" ]]; then
  echo "missing $BUILD/lib/libninfer_ops.so (configure: cmake --preset rtx5090-ops)" >&2
  exit 1
fi

TARGETS="$CUDA/targets/x86_64-linux/include"
[[ -d $TARGETS ]] || TARGETS="$CUDA/include"

set -x
c++ -std=gnu++20 -O1 -Wall \
  -I "$TREE/include" -I "$TREE/src" \
  -isystem "$TARGETS" -isystem "$TARGETS/cccl" \
  "$HERE/dspine_nvfp4_fixture.cpp" \
  -o "$HERE/dspine_nvfp4_fixture" \
  -L "$BUILD/lib" -lninfer_ops -lninfer_core \
  -L "$CUDA/targets/x86_64-linux/lib" -lcudart -lcuda \
  -Wl,-rpath="$BUILD/lib" -Wl,-rpath="$CUDA/targets/x86_64-linux/lib"
set +x
echo "built $HERE/dspine_nvfp4_fixture"
