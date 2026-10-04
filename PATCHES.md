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

## Carried commits (as of 2026-10-06 — rebased onto `abb7f14f`) This fork = upstream `Neroued/ninfer` master (`abb7f14f`, context-cache runtime
+ Prometheus + SM-derived launch plans) + the commits below. The 2026-10-06
rebase **dropped the #335 hybrid-prefix-cache patch** (the upstream runtime
supersedes it — A/B PASS, `v-llm-gateway/docs/PLAN-upstream-cache-test.md`)
and with it #274 (shared-prefix catalog — superseded by the runtime's
`prefix_index`) and the 335-only publication-streams fix. The #148 Responses
reasoning pair was **dropped 2026-10-06 (no longer helps)**: nothing in the
serving stack or the eval runner uses the Responses API with
`reasoning.encrypted_content` (evals use the OpenAI-compatible
chat-completions surface; prod ran the upstream-cache base without #148 with
no breakage). Re-land from history if a client ever needs it. | Local commit | Source | Upstream PR | What | Drop trigger |
|---|---|---|---|---|
| `c5683148` `160cc982` `f671c381` | cherry-pick trio | **#355** | nvfp4 tiled split-KV prefill (gridZ 1→7) + shared tiled KV-split helper + mxfp8 route — prefill-200k +5.8% on our box | When #355 merges |
| `e9bedea6` | **our own (build)** | — | curl in the runtime image (container HEALTHCHECK + watchdog support; stock image ships only libcurl) | Keep (build, not engine) |
| `270439f5` | **our own (tools/docs)** | — | fork-local build/quant tooling: `FORK.md`, this registry, `tools/convert/nvfp4full-build/` pipeline, `graft_dflash2_w8`, `strip_dflash2.py`, `upgrade_ninfer_v2_to_v3.py` `KNOWN_COUNTS` line | Keep forever (not engine); drop the upgrade-tool line when upstream registers `nvfp4full` |
| `9849047b` | cherry-pick `-x` | **#297** | fix(core): preserve workspace layout state after allocation overflow | When #297 merges |
| `37171caa` | cherry-pick `-x`, **re-anchored 2026-10-06** | **#268** | perf(attention): fold the sigmoid gate into the causal reduce epilogue — re-landed onto the `DeviceExecutionView` API (gate applied via `execution.stream` after the storage route; the #355 merge had restored the caller-side `sigmoid_mul`, so the fold re-removes it — no double application) | When #268 merges |
| `a7e1b363` | cherry-pick `-x` | **#299** | fix(frontend): keep the last value on a duplicate tool-call parameter (`duplicate_parameters_repaired` diagnostic) | When #299 merges |
| `24e4eedd` + `944f2eb9` | cherry-pick `-x`, **conflict resolved by hand** | **#309** | fix(frontend): keep quoted `</think>` closes and later `<tool_call>` markers; the #309 candidate loop keeps #299's non-const parser | When #309 merges (re-check the #299 merge if #299 lands first) |
| `841c137a` | cherry-pick `-x`, **re-anchored 2026-10-06** | **#61** (Sociopacific) | `--image-token-budget N` per-image Vision-token ceiling (closes the prod gap: images were uncapped on the upstream-cache base) | When #61 merges |
| `518223c2` | port of `03df31d5` | **#97** (DuncanBetts) | ccache + BuildKit cache mount in Dockerfile (incremental builds) | When #97 merges | **Dropped at the 2026-10-06 rebase (do not re-land without new evidence):**
#335 hybrid prefix cache (superseded by the upstream runtime), #274
(superseded by `prefix_index`), 335-only publication-streams fix (superseded
by upstream `064965c7`), #148 Responses-reasoning pair (unused API surface —
see note above), #264 (superseded by upstream shared-kernel swiglu route),
#305 (closed unmerged upstream — falsified direction). 