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

## Carried commits (as of 2026-09-07)

This fork = upstream `Neroued/ninfer` master (`487f8977`) + the commits below.

| Local commit | Source | Upstream PR | What | Drop trigger |
|---|---|---|---|---|
| `2eb59dbc` | **our own (engine)** | — (proposal upstream-worthy) | register `qwen3.8-27b/nvfp4full` weights profile (4-file port of cometkim `feat/qwen3.8-nvfp4full` — Fused nvfp4full binder for the cometkim artifact; upstream only has groupwise-int + nvfp4) | When upstream adds `nvfp4full` (none open; PR #107 covers only Ostrfella profile) |
| `65bbf8b5` | **our own (tools/docs)** | — | `tools/artifact/graft_dflash2_w8.py` (z-lab DFlash2 W8G32/BF16 module graft), `FORK.md`, sanitized serving example | Keep forever (not engine) |
| `71b4c9b2` | cherry-pick `193dab17` | **#148** (Sha1rholder) | OpenAI Responses API: accept `include: reasoning.encrypted_content` + `reasoning.summary` | When #148 merges |
| `bcc6261e` | **our own** | (derived from #148) | skip summary/encrypted-only reasoning **input** Items (Inspect AI multi-turn); upstream #148 only handled the create side | Keep until #148 lands + re-diff; then only if upstream adds the input-side handling |
| `e202c53b` | port of `03df31d5` | **#97** (DuncanBetts) | ccache + BuildKit cache mount in Dockerfile (incremental builds) | When #97 merges |
| `7e8ad2e9` | cherry-pick `545f64b0` | **#160** (MichaelDementii) | NVFP4 TMA route reads activation scales tile-contiguous (prefill +1-2.5% on our box) | When #160 merges |
| `4ac61fa2` | cherry-pick `eb413c76` | **#61** (Sociopacific) | `--image-token-budget N` per-image Vision-token ceiling + **our validator fix** (allow policy-lowered `image_max_pixels`) | When #61 merges (verify our validator hunk is included; we posted it as a comment) |

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
