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

## Carried commits (as of 2026-09-24 — rebased onto `bace20dc`)

This fork = upstream `Neroued/ninfer` master (`bace20dc`, incl. the silu
accuracy fix `c4ae8a9c`) + the commits below. The 2026-09-24 rebase added
the five `port/2026-09-21-six-prs` ports (#297 #274 #299 #264 #268, A/B'd on
that branch) plus two new ports (#309 #305). #309 conflicted with our #299
port in `tool_call_parser.cpp` and was merged by hand: #309's candidate
marker loop keeps #299's `duplicate_parameters_repaired` capture (the parser
is loop-scoped, so the flag is captured on the winning parse).

| Local commit | Source | Upstream PR | What | Drop trigger |
|---|---|---|---|---|
| `f8c5bd3b` | **our own (tools/docs)** | — | `tools/artifact/graft_dflash2_w8.py` (z-lab DFlash2 W8G32/BF16 module graft), `FORK.md`, sanitized serving example | Keep forever (not engine) |
| `8af0e5f7` | cherry-pick `193dab17` | **#148** (Sha1rholder) → re-based as **#295** | OpenAI Responses API: accept `include: reasoning.encrypted_content` + `reasoning.summary` | When #295 (rebase of #148) merges — adopt #295, then re-diff the next row against it |
| `40652a3a` | **our own** | (derived from #148) | skip summary/encrypted-only reasoning **input** Items (Inspect AI multi-turn); upstream #148 only handled the create side | Keep until #148 lands + re-diff |
| `ac0b55d1` | port of `03df31d5` | **#97** (DuncanBetts) | ccache + BuildKit cache mount in Dockerfile (incremental builds) | When #97 merges |
| `c4114051` | cherry-pick `eb413c76` | **#61** (Sociopacific) | `--image-token-budget N` per-image Vision-token ceiling (re-anchored onto v3's `processor_options`/`FrontendOptions` chain). Our original validator fix became moot — v3 dropped the strict registered-pixel-bounds check. | When #61 merges |
| `fb7a631a` | **our own (build)** | — | curl in the runtime image (container healthcheck support) | Keep (build, not engine) |
| `036f3621` | **our own (docs)** | — | this patch registry + fork policy (`PATCHES.md`) | Keep (docs) |
| `78054dab` | **our own (docs)** | — | `FORK.md` carried-commit delta | Keep (docs) |
| `3a8401ef` | **our own (tools)** | — | `tools/upgrade_ninfer_v2_to_v3.py`: add `qwen3.8-27b/nvfp4full` to `KNOWN_COUNTS` (1259 plain / 1325 with DFlash2 graft) so our artifact upgrades to a v3 container with weight bytes preserved | When upstream registers `nvfp4full` (then the entry is upstream) |
| `df34f428` | cherry-pick | **#297** | fix(core): preserve workspace layout state after allocation overflow | When #297 merges |
| `458d06eb` | cherry-pick | **#274** | fix(runtime): default shared-prefix catalog sized for one request's full candidate set (7 candidates > old `max(concurrency,4)` default — the eviction bug behind our `--max-shared-prefixes 16` cfg workaround) | When #274 merges |
| `c8a1aac0` | cherry-pick | **#299** | fix(frontend): keep the last value on a duplicate tool-call parameter (`duplicate_parameters_repaired` diagnostic) | When #299 merges |
| `0e6fa957` | cherry-pick | **#264** | perf(nvfp4): fused SwiGLU TMA route takes a partial last M tile (ragged widths stop falling back to linear + silu_mul) | When #264 merges |
| `05a98aeb` | cherry-pick | **#268** | perf(attention): fold the sigmoid gate into the causal reduce epilogue (one graph node instead of two) | When #268 merges |
| `81feb389` | cherry-pick `-x`, **conflict resolved by hand** | **#309** | fix(frontend): keep quoted `</think>` closes and later `<tool_call>` markers (candidate marker loop). Merged with our #299 port — see note above | When #309 merges (re-check the #299 merge if #299 lands first) |
| `3b1c42f5` | cherry-pick `-x` | **#305** | perf(ops): fuse attention RMSNorm + NVFP4 activation quant on the T≥1024 TMA route (our prefill chunk is 4096 — hits our path) | When #305 merges |

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
