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

## Never-drop files (survive rebases)

- `tools/convert/nvfp4full-build/` — all-NVFP4 artifact build pipeline
  (quantize_all_nvfp4.py, normalize_groups.py, recipe_swift_full.py,
  nvfp4full_qconfig.json, README with full runbook). Upstream has no
  equivalent; it produced both published kaushikvira v3 artifacts
  (swift-abliterated 74c97213…, swift15 16f313c0…). On every rebase:
  `git log origin/master..main -- tools/convert/nvfp4full-build/` must be
  non-empty-of-commits i.e. the directory still exists, and re-push the
  backup branch: `git push -f gh main:contrib/nvfp4full-build`.
  (Lost-file precedent: `tools/convert/qwen3_8_27b/dflash2_recipe.py` was
  dropped during the 2026-09-16 squash-rebase — don't repeat that.)

## Carried commits (as of 2026-10-17 — rebased onto `64d10fbf`)

This fork = upstream `Neroued/ninfer` master (`64d10fbf`, the ops/MoE reorg
+ Qwen4Exp/Flash-Next + MoE CPU-offload wave) + the commits below. The
2026-10-17 rebase re-anchored the carried set onto the reorg: the qwen
frontend moved to `src/models/common/qwen/frontend/`, the linear/attention
families were reorganized, and four old carried patches were **dropped as
superseded** (see below). #309's candidate loop now rides on the dup-param
commit's `QwenToolRegionParser` class (picked first, on purpose).

| Local commit | Source | Upstream PR | What | Drop trigger |
|---|---|---|---|---|
| `6038507d` | cherry-pick `7092f747` | **#299** | keep-last-value on duplicate tool-call param + `QwenToolRegionParser` class + `duplicate_parameters_repaired` diagnostic. Upstream now has the *behavior* (non-strict keep-last) but not the class/diagnostic that #309's loop depends on — carried for the structure | When upstream's parser gains the class + diagnostic, or #309 drops |
| `cf0de389` | cherry-pick `63c1e547` | **#309** | fix(frontend): keep quoted `</think>` closes and later `<tool_call>` markers (candidate marker loop), re-anchored onto `common/qwen/frontend/` | When #309 merges |
| `188de8d9` | cherry-pick `bdedac0b` | **#309** | parser in the #309 candidate loop must be non-const | With #309 |
| `9a4b042a` | cherry-pick `57e09c80` | **#309** | reconcile #309 close-scan with upstream constrained decoding (grammar-active exact framing) | With #309 / when constrained decoding changes |
| `aa6c95fc` | cherry-pick `45be95b0` | **#368** (rebase of #152) | automatic shared-prefix write at the system/developer frontier (agent siblings share the long head); `--no-auto-system-shared-prefix` opt-out | When #368/#152 merges |
| `4664eab5` | cherry-pick `92f0f23e` | **#368** | tests + docs for the system-frontier candidate | With #368 |
| `a5400c9c` | cherry-pick `8adf4615` | **#368** | wire `--no-auto-system-shared-prefix` into Chat Completions | With #368 |
| `dff5f983` | cherry-pick `cc79701e` | **#403** (open, fixes #402) | O(1) survivor membership in `recovery_loss` (cubic→quadratic; 147x at N=2048). Re-anchored onto `qwen3_5/program/transactions/reclaim.{cpp,h}` | When #403 merges upstream |
| `2b734237` | **our own (tools/docs)** | — | `FORK.md`, `PATCHES.md`, `tools/convert/nvfp4full-build/` (nvfp4full artifact pipeline), `tools/artifact/graft_dflash2_w8.py` | Keep forever (not engine) |
| `a46d1de3` | **our own (build)** | — | Dockerfile fork patch: curl (healthcheck CLI) + liburing (PLE io_uring dep, stock Dockerfile doesn't build) + ccache/BuildKit cache mount (#97) | Keep (build, not engine). **liburing lines are upstream PR #407** (open, mergeable) — drop them from this patch when #407 merges; **curl** suggested on #407 (comment 2026-10-17) — drop when it lands; drop ccache part when #97 merges |

**Dropped as superseded by the `64d10fbf` reorg (2026-10-17):**

- **#355** (nvfp4 tiled-prefill KV split across CTAs) — upstream's new
  per-dtype plan architecture (`dense/causal_cache/*/plan.cpp` +
  `CausalKvPartition`) subsumes it. Re-measure via the perf bench; re-port
  only if the 27B gate regresses.
- **#365** (nvfp4 linear_add verify-width deep tile) — upstream's new nvfp4
  linear-add schedule family (sliced-K MMA for T≤32, TMA variants) covers the
  small-token verify case. Same re-measure rule.
- **mxfp8 shared KV-split helper** (`5d6d2fc6`/`a0ef6807`) — upstream's
  softmax-attention reorg moved the wave-minimizing search into per-dtype
  `plan.cpp`; the helper's target file no longer exists.
- **standalone curl + ccache Dockerfile commits** — folded into the single
  Dockerfile fork patch (`a46d1de3`).

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

### Reviewed 2026-09-24, not taken

- **#300** RFC agent-workload bundle (+10% dflash2 agent): explicitly "not for merge"; bundles
  #177/#178/#179/#208/#251 context-cache fixes — revisit once upstream stabilizes the pieces.
- **#311/#292** Q4/Q5 K-split routes: not our quant (we are NVFP4); target MTP3 groupwise serving.
- **#284** Q6 gate/up: not our quant.
- **#286–290** NVFP4 sparse-MoE series: Qwen3.8-27B is dense.
- **#307** logprobs logsumexp: we never request logprobs.
- **#304** speed-of-light bench estimator: offline tooling, no engine surface (revisit for A/B analysis).
- **#282** GGUF conversion source: not our conversion path (cometkim nvfp4full + graft).
- **#235** CUDA 12.9 floor: we build on 13.1.
- **#221** MTP draft >5 startup failure: we run DFlash2 K=7; MTP fallback profile is K=4 (below the limit).
- **#294** XGrammar structured output: not maintainer-pre-approved, vendors a grammar lib; we don't use
  `response_format` (tool calls go through the #309 parser).
- **#199** sparse-MoE Q4 quads: MoE.
- **#295** (rebase of #148): we already carry the feature (rows above) — adopt #295 *instead* of our
  two #148-derived commits when it merges, then re-diff.
