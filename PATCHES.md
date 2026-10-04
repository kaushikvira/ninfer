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

## Rebase record 2026-09-26 — onto `e31bc99b` (template unification)

Upstream landed the linear-ops template unification (q4–q8/fp8/nvfp4/bf16 a16_mma
+ sliced_k_mma + TMA routes), KDA (Kimi delta attention), and the GDN two-stage
rewrite. **Two carried commits dropped:**

- `0e6fa957` (port of **#264**, SwiGLU partial last M tile) — superseded:
  master's `fc3993d8` routes swiglu `tokens >= 256` through the shared
  `launch_nvfp4_a4_tma_mma` kernel with `div_up` grids. #264 itself is still
  open but now proposes only the narrower 512 ragged floor on top.
- `3b1c42f5` (cherry-pick of #305 head `1f2ab4b5`, RMSNorm+NVFP4 quant fusion) —
  **#305 closed unmerged 2026-09-25 as a recorded falsified direction**:
  author's SoL math + stage measurements show ~5% at stage level but ~0.04%
  end-to-end prefill (compute-bound), and eligibility (`tokens >= 1024`) meant
  it never ran at decode. Re-port would have cost a full w4a4→a4 API rename.

Also noted: #148 closed unmerged — successor is #295 (Macasacker); keep our
`8af0e5f7` + `40652a3a` until #295 merges. #309 (quoted reasoning closes)
closed unmerged with no comments — still carrying `81feb389` + `3090a5b1`
(functional tool-call fix; re-evaluate if #318 lands).

## Sync record 2026-10-04 — PR heads re-checked, 3 ports updated (base unchanged: `d44ab584`)

Upstream master unchanged (`d44ab584` — none of the carried PRs merged). All 8 carried
PR branches re-fetched; 5 heads identical to what we ported (#61 `eb413c76`, #268
`2961dac6`, #274 `0f5f96f4`, #299 `d3a44d21`, #309 `71304581`). Three were stale and
updated on branch `kv/pr-sync`:

- **#97** grew 1→7 content commits: pulled `ec1efc4b..bc3a8633` (config-invalidation fix
  `9d89a654` + bounded config trees `bc3a8633` + build-cache docs). Branch-sync merge
  `c0392a3b` skipped (content already in master).
- **#297** force-pushed: author rewrote the fix as a minimal cursor-ordering change
  (`6d72167c`, Sep 27) replacing the original restructure (`1d4a6162`, Sep 20). Re-ported;
  same bug, smaller diff (8 lines vs 106/37).
- **#148 → #295**: adopted #295's single rebased commit `d9f4c8fb` in place of our
  `193dab17` port (superset: +reasoning.summary, expanded tests/docs). Our own
  `422dc28f` (Responses input-item skip) retained after it.

## Rebase record 2026-09-30 — onto `d44ab584` (attention reorg + fp8 linear tuning)

Upstream landed a 23-commit perf push: per-dtype causal-attention reorg
(bf16/fp8/int8/k8v4/nvfp4 kernels split from the monolithic
`prompt_*`/`small_t*` files into per-dtype dirs with shared
`common/causal_*` primitives), split-kv fp8/k8v4 prefill, fp8 linear TMA
tuning for qwen3.8 shapes, native nvfp4 a16 decode + fp8→bf16 conversion,
DFlash per-chunk prefill-control binding (`4201b5d2`), and a configurable
`--kv-dtype bf16|int8|fp8|nvfp4|k8v4` bench runner (`c1c48a6a`).

**One carried commit re-landed by hand:** `15447ad7` (#268, sigmoid gate
into the causal reduce epilogue) conflicted with the attention reorg — the
shared BF16/INT8 small-T reducer it fused into no longer exists (each
dtype now has its own reduce kernel). Re-landed in the standalone form:
the gate is a `const Tensor*` on the public `causal_softmax_attention`
API, applied by `sigmoid_mul` after the per-storage dispatch (bit-
identical to the pre-fold behaviour; the ~1 µs/node fusion is deferred
until a per-dtype epilogue exists to fold into). Bench gate extensions
(`--gate off|standalone|fused`) dropped from the bench file in favour of
upstream's `582c9a8f`/`909fb087` bench rework; gate coverage stays in the
`causal_cache.cpp` oracle. No commits dropped; no PR ports merged into
the delta (all direct upstream commits).

## Carried commits (as of 2026-09-30 — rebased onto `d44ab584`)

This fork = upstream `Neroued/ninfer` master (`d44ab584`, the 2026-09-27→29
attention reorg + fp8 linear tuning, on top of `e31bc99b`), the 2026-09-26
linear-ops template unification + KDA + GDN two-stage, incl. the silu accuracy
fix `c4ae8a9c`) + the commits below. The 2026-09-24 rebase added the five
`port/2026-09-21-six-prs` ports (#297 #274 #299 #264 #268, A/B'd on that
branch) plus two ports (#309 #305). #309 conflicted with our #299 port in
`tool_call_parser.cpp` and was merged by hand: #309's candidate marker loop
keeps #299's `duplicate_parameters_repaired` capture (the parser is
loop-scoped, so the flag is captured on the winning parse).

**2026-09-26 rebase dropped two of those ports** — `0e6fa957` (#264,
superseded by upstream's shared-kernel swiglu route) and `3b1c42f5` (#305
closed as a falsified direction). The e31bc99b build is perf-flat vs
bace20dc on our serving mix but fails the extended needle ladder
(empty response at 128k chars/depth 0.9, twice; rollback passes 15/15 —
record: v-llm-gateway `docs/NINFER_A_B.md` 2026-09-26). Prod image stays on
the bace20dc build until upstream fixes it; candidate kept as Docker tag
`ninfer-master:e31bc99b-candidate`. The 2026-09-30 rebase onto `d44ab584`
drops nothing; the new candidate is Docker tag `ninfer-master:d44ab584-candidate`.

| Local commit | Source | Upstream PR | What | Drop trigger |
|---|---|---|---|---|
| `465cff38` | **our own (tools/docs)** | — | `tools/artifact/graft_dflash2_w8.py` (z-lab DFlash2 W8G32/BF16 module graft), `FORK.md`, sanitized serving example | Keep forever (not engine) |
| `99798b17` | cherry-pick `d9f4c8fb` | **#295** (rebase of #148; adopted 2026-10-04 in place of the `193dab17` port) | OpenAI Responses API: accept `include: reasoning.encrypted_content` + `reasoning.summary` | When #295 merges — then re-diff the next row against it |
| `422dc28f` | **our own** | (derived from #148) | skip summary/encrypted-only reasoning **input** Items (Inspect AI multi-turn); upstream #148 only handled the create side | Keep until #295 lands + re-diff |
| `94795422..83f04521` | port of `03df31d5..bc3a8633` (7-commit stack, 2026-10-04) | **#97** (DuncanBetts) | ccache + BuildKit cache mount in Dockerfile (incremental builds) + config-invalidation fix + bounded config trees | When #97 merges |
| `8ab2b38d` | cherry-pick `eb413c76` | **#61** (Sociopacific) | `--image-token-budget N` per-image Vision-token ceiling (re-anchored onto v3's `processor_options`/`FrontendOptions` chain). Our original validator fix became moot — v3 dropped the strict registered-pixel-bounds check. | When #61 merges |
| `64a0995f` | **our own (build)** | — | curl in the runtime image (container healthcheck support) | Keep (build, not engine) |
| `14f81eec` | **our own (docs)** | — | this patch registry + fork policy (`PATCHES.md`) | Keep (docs) |
| `b70936cc` | **our own (docs)** | — | `FORK.md` carried-commit delta | Keep (docs) |
| `da706aaa` | **our own (tools)** | — | `tools/upgrade_ninfer_v2_to_v3.py`: add `qwen3.8-27b/nvfp4full` to `KNOWN_COUNTS` (1259 plain / 1325 with DFlash2 graft) so our artifact upgrades to a v3 container with weight bytes preserved | When upstream registers `nvfp4full` (then the entry is upstream) |
| `f1f3e3df` | cherry-pick `6d72167c` (re-port 2026-10-04; author force-pushed a cleaner minimal fix Sep 27) | **#297** | fix(core): preserve workspace layout state after allocation overflow | When #297 merges |
| `2e18151b` | cherry-pick | **#274** | fix(runtime): default shared-prefix catalog sized for one request's full candidate set (7 candidates > old `max(concurrency,4)` default — the eviction bug behind our `--max-shared-prefixes 16` cfg workaround) | When #274 merges |
| `3c951b36` | cherry-pick | **#299** | fix(frontend): keep the last value on a duplicate tool-call parameter (`duplicate_parameters_repaired` diagnostic) | When #299 merges |
| `15447ad7` | cherry-pick | **#268** | perf(attention): fold the sigmoid gate into the causal reduce epilogue (one graph node instead of two) | When #268 merges |
| `7152abdd` | cherry-pick `-x`, **conflict resolved by hand** | **#309** | fix(frontend): keep quoted `</think>` closes and later `<tool_call>` markers (candidate marker loop). Merged with our #299 port — see note above | When #309 merges (re-check the #299 merge if #299 lands first; #309 closed unmerged 2026-09-25 with no comments — re-evaluate against #318) |

*(SHA column = post-2026-09-26-rebase SHAs. The dropped `0e6fa957` #264 and `3b1c42f5` #305 rows are recorded in the rebase section above — no longer carried.)*

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
