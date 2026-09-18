# Patches carried by this fork (drop as upstream merges)

## Fork policy — stay close to upstream

This fork exists to *ship* what upstream hasn't merged yet, not to diverge:

- **Always rebase onto upstream master.** The delta should stay tiny and readable.
  After each upstream pull, `git log origin/master..main` should show only the
  commits below.
- **Prefer upstream PRs** over own inventions: if an open upstream PR does it,
  cherry-pick it (or port it) and track it here instead of writing our own —
  every such commit is *temporary*, to be dropped when the PR merges.
- **Own commits** (the nvfp4full registration, the graft tool, small fixes on
  top of a cherry-pick) stay only if upstream genuinely lacks them; re-check
  each rebase and propose upstream when worthwhile.
- **Never drift**: no big rewrites, no third-party infra, no vendor lock-in.
  If we need something upstream rejects, keep it minimal and documented here.

## Carried commits (as of 2026-09-18 — rebased onto v3 `f76e19c0`)

This fork = upstream `Neroued/ninfer` master (`f76e19c0`, v3 model/weight
decoupling) + the commits below. The v3 rebase **dropped two carried commits**:
the C++ `nvfp4full` weights-profile registration (v3's loader is data-driven —
the enum/binder became dead code) and the PR #160 tile-contiguous scales port
(superseded by upstream `1d8587bc`). `nvfp4full` now lives entirely in the
artifact, upgraded to a v3 container with the extended `tools/upgrade_ninfer_v2_to_v3.py`
(see below) — no engine fork needed for it.

| Local commit | Source | Upstream PR | What | Drop trigger |
|---|---|---|---|---|
| `50e04987` | **our own (tools/docs)** | — | `tools/artifact/graft_dflash2_w8.py` (z-lab DFlash2 W8G32/BF16 module graft), `FORK.md`, sanitized serving example | Keep forever (not engine) |
| `9dcdce39` | cherry-pick `193dab17` | **#148** (Sha1rholder) | OpenAI Responses API: accept `include: reasoning.encrypted_content` + `reasoning.summary` | When #148 merges |
| `25de38a5` | **our own** | (derived from #148) | skip summary/encrypted-only reasoning **input** Items (Inspect AI multi-turn); upstream #148 only handled the create side | Keep until #148 lands + re-diff |
| `3ca1a001` | port of `03df31d5` | **#97** (DuncanBetts) | ccache + BuildKit cache mount in Dockerfile (incremental builds) | When #97 merges |
| `f26621be` | cherry-pick `eb413c76` | **#61** (Sociopacific) | `--image-token-budget N` per-image Vision-token ceiling (re-anchored onto v3's `processor_options`/`FrontendOptions` chain). Our original validator fix became moot — v3 dropped the strict registered-pixel-bounds check. | When #61 merges |
| `cba96bcc` | **our own (build)** | — | curl in the runtime image (container healthcheck support) | Keep (build, not engine) |
| `775f1e0b` | **our own (docs)** | — | this patch registry + fork policy (`PATCHES.md`) | Keep (docs) |
| `3550b95f` | **our own (docs)** | — | `FORK.md` carried-commit delta | Keep (docs) |
| *(this commit)* | **our own (tools)** | — | `tools/upgrade_ninfer_v2_to_v3.py`: add `qwen3.8-27b/nvfp4full` to `KNOWN_COUNTS` (1259 plain / 1325 with DFlash2 graft) so our artifact upgrades to a v3 container with weight bytes preserved | When upstream registers `nvfp4full` (then the entry is upstream) |

**Serving note (v3 cutover, 2026-09-18):** the on-box artifact was upgraded to
`qwen3_8_27b_nvfp4full-dflash2.v3.ninfer` (+289 KB metadata, same 18.7 GiB
device footprint). Gate on the rebased build: needle 12/12, tool 10/10, decode
**162.5 tok/s** (baseline `default.json` re-saved to v3 numbers; first cold-start
probe read 141 and settled at 162.5). Profile: `bench/configs/ninfer-nvfp4full-dflash2-v3.cfg`.

## How to re-sync after an upstream merge

```bash
git fetch origin master
# after dropping a merged commit from `main` (e.g. cherry-pick -x or rebase):
git log --oneline origin/master..main   # should be only fork-native commits left
```

Checklist per drop: re-run `bench/tests/gate.sh` (needle + perf + tool) + the
relevant probe (image budget → one image request; #160 → perf.py prefill) and
verify the engine boots with the base cfg unchanged.

## Upstream PRs we reviewed and did NOT take

- **#173** rk2v4-e8 compressed KV (208 B/head-token): huge re-port (4131 lines); would ~double
  on-device KV but 2-bit K is a quality risk vs our k8v4 (12/12 needle). Revisit only with a
  full needle/quality gate.
- **#167** FP8 A8 TMA GEMM: +2.7-4.7% prefill on FP8-row paths; our base is W8G32 — benefit small.
- **#194** drop guarded expf in SwiGLU epilogue: 1.8-5.4% on one operator, likely <1% end-to-end;
  author withdrew a claim. Skipped as marginal.
- **#152** automatic shared-prefix write at system/developer frontier: nice for clients that
  don't send prompt_cache_breakpoint; our clients do — low value here.
- **#195/#107/#183/#197/#163/#162/#54/#84/#59** — not applicable (no corresponding profile /
  Windows / metadata conveniences).
