# How we rebased onto `64d10fbf` (2026-10-17) — and how to do the next one

The `64d10fbf` wave was the hardest rebase yet: a ~75-commit ops/MoE/context
reorg + new Qwen4Exp/Flash-Next + MoE CPU-offload executor. This records the
non-obvious parts so the next resync is faster. The general workflow lives in
`v-llm-gateway/.pi/skills/ninfer-resync/SKILL.md`; this is the *gotchas* layer.

## 1. Survey before you rebase — decide what to DROP
The reorg **superseded** three carried patches. Check each carried patch
against the new tree *before* cherry-picking, or you'll port dead code:

| Dropped | Why the reorg made it obsolete |
|---|---|
| #355 (nvfp4 tiled-prefill KV split) | New per-dtype `plan.cpp` + `CausalKvPartition` subsumes it |
| #365 (nvfp4 linear_add verify-width tile) | New nvfp4 linear-add schedule family (sliced-K MMA T≤32, TMA) covers it |
| mxfp8 shared KV-split helper | Reorg moved the wave-search into per-dtype `plan.cpp`; target file gone |

Rule: if upstream rewrote the subsystem your patch touches, the patch is
probably dead. Confirm by grepping the new tree for the behavior, then drop +
record in `PATCHES.md` (re-measure via perf bench; re-port only on regression).

## 2. Cherry-pick, don't `git rebase` the whole range
`git rebase upstream/master` replayed 21 commits and tangled the superseded
ones in. Cleaner: reset a scratch branch to `upstream/master` and
`git cherry-pick` only the commits you're keeping, in dependency order.

**Dependency gotcha:** #309's candidate loop depends on the dup-param commit's
`QwenToolRegionParser` class + `duplicate_parameters_repaired` diagnostic.
Pick the dup-param commit **first**, or #309 won't compile.

## 3. Auto-merge duplicate-field artifacts (the sneaky one)
When a cherry-pick adds a field/constant that upstream *also* added in the
reorg, the 3-way merge silently produces a **duplicate declaration**:
- `EngineOptions::image_token_budget` (types.h) — added twice
- `kPixelsPerVisionToken` (frontend.cpp) — added twice

These compile-fail with "redefinition". Grep the touched files for the symbol
after each pick; a `grep -c` > 1 means a dup. Delete the extra copy.

## 4. API shape changes in the reorg (field → method)
The reorg turned `CheckpointState::frontier` from a **field** into a
**`frontier()` method**. The #403 port used the field form →
"invalid use of member function". Fix: call `frontier()`. The reclaim scaling
test's synthetic `SynthState` also needed a `frontier()` method to match.
General rule: after a big reorg, expect field→method / signature shifts in the
types your patches touch; the compiler names them precisely.

## 5. Test-container environment gaps (NOT code breaks)
Running full ctest in an ad-hoc `nvidia/cuda:13.1.2-devel` container hits
environmental walls that look like failures but aren't:

| Symptom | Cause | Fix |
|---|---|---|
| `nvtx3/nvToolsExt.h: No such file` | CUDA include not on g++ default path | `-e CPLUS_INCLUDE_PATH=/usr/local/cuda/targets/x86_64-linux/include` |
| `io_uring: Operation not permitted` (read_queue, artifact_backing, ple_disk) | container blocks the io_uring syscall | environmental — accept, or run with the capability |
| `No module named 'jinja2'/'jsonschema'` | Python deps not in the image | `apt-get install python3-jinja2` (+ `python3-jsonschema` / pip) |
| `no CUDA-capable device` on a GPU test | `CUDA_VISIBLE_DEVICES=1` inside a single-`--gpus device=1` container (re-indexed to 0) | unset it, or match the container's visible index |

Distinguish environmental from semantic: **instant failures (<0.05s) with an
OS/module/CUDA error = environmental.** A real logic failure runs longer and
asserts. The 7 failures here were all environmental; every test covering a
carried patch passed.

## 6. Build/test command that worked
```bash
docker run --rm --gpus device=1 -v $PWD:/src -w /src \
  -v ~/.ccache-ninfer:/ccache \
  -e CPLUS_INCLUDE_PATH=/usr/local/cuda/targets/x86_64-linux/include \
  nvidia/cuda:13.1.2-devel-ubuntu24.04 bash -c '
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends cmake ninja-build pkg-config \
    python3 python3-jinja2 libavcodec-dev libavformat-dev libavutil-dev \
    libcurl4-openssl-dev libhwloc-dev libswscale-dev liburing-dev ccache
  export CCACHE_DIR=/ccache CCACHE_MAXSIZE=20G
  cmake -S . -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=ON \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache
  cmake --build /build --parallel
  ctest --test-dir /build --output-on-failure --timeout 120'
```
Notes: `liburing-dev` is now a hard build dep (PLE io_uring path) — the stock
upstream Dockerfile is missing it (our Dockerfile patch adds it). Point GPU
tests at the freer card (`device=1`); the 5090 runs prod. `--timeout 120` so a
hung GPU test can't stall the whole run.

## 7. The Dockerfile fork patch (single commit, not three)
Fold curl + liburing + ccache into **one** Dockerfile commit over stock
upstream (the old separate curl/ccache commits predate the reorg). curl =
container healthcheck (a stock image with no curl shows "unhealthy" in docker
even when the API is fine); liburing = won't build without it; ccache = fast
rebuilds.
