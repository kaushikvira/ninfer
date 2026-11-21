#!/bin/bash
# Build the P0b parity fixture against the P0a engine build (read-only).
# Requires: g++ (c++), CUDA headers, /tmp/ninfer-dspine-port/build/p0a/lib/*.so
set -euo pipefail

PORT=${PORT:-/tmp/ninfer-dspine-port}
BUILD=${BUILD:-$PORT/build/p0a}
CUDA=${CUDA:-/usr/local/cuda}
HERE=$(cd "$(dirname "$0")" && pwd)

if [[ ! -f "$BUILD/lib/libninfer_ops.so" ]]; then
  echo "missing $BUILD/lib/libninfer_ops.so (P0a build)" >&2
  exit 1
fi

TARGETS="$CUDA/targets/x86_64-linux/include"
[[ -d $TARGETS ]] || TARGETS="$CUDA/include"

set -x
c++ -std=gnu++20 -O1 -Wall \
  -I "$PORT/include" -I "$PORT/src" \
  -isystem "$TARGETS" -isystem "$TARGETS/cccl" \
  "$HERE/dspine_fixture.cpp" \
  -o "$HERE/dspine_fixture" \
  -L "$BUILD/lib" -lninfer_ops -lninfer_core \
  -L "$CUDA/targets/x86_64-linux/lib" -lcudart -lcuda \
  -Wl,-rpath="$BUILD/lib" -Wl,-rpath="$CUDA/targets/x86_64-linux/lib"
set +x
echo "built $HERE/dspine_fixture"
