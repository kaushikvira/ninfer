# PLAN — DSpine speculative-decoding backend for NInfer

status: active · updated: 2026-11-21 · branch: feat/dspine-spec-backend (base main 70afbf33)

## READ-FIRST (compaction re-entry)

This file is the **durable context store**. The session transcript and
observational memory are **disposable** — both lanes (opus-5, sonnet-5) are
262k and a 10–14 day build crosses compaction. On resume: read the **STATUS**
block → the phase section → the **DECISION LEDGER** (esp. the `⚠` entries),
**then act**. Do NOT re-derive from conversation. Append to the ledger; never
rewrite a locked line — supersede it with a dated `⚠` line.

## STATUS (cursor — update + commit at every gate)

- phase: **P0 (de-risk spike) — not started**
- done: plan written; branch `feat/dspine-spec-backend` created
- next: **P0a** port `dspine_bridge` ops to a worktree of prod base
  `416-ebdc67e5`; **P0b** PyTorch bit-exact vectors vs
  `Tencent/DSpine/dspine_mlx/.../reference.py`; **P0c** fleet gate
  (`lane down a5000-11` → GPU-1 ≥ 35 GiB → `lane up a5000-2` → stock-target
  probe). Kill-switch: P0b not bit-exact ⇒ STOP.
- open-risks: R5 upstream-drift, R3 Swift-1.5-tower acceptance (both P0-gated)
- last-gate: — (none passed)

## DECISION LEDGER (append-only; `⚠` = a correction, obey it)

- **D-2026-11-21-01** — DSpine, not EAGLE-4. EAGLE-4 is source-available with a
  **anti-porting clause** ("…porting… placing material Eagle 4 technology in
  another framework") — legally unusable as a NInfer backend. DSpine = Apache-2.0.
- **D-2026-11-21-02** — `⚠` **Model→lane map (LOCKED, router-authoritative).**
  `litellm.l.vira.rocks` is an external LiteLLM router; **`models.json` `name:`
  fields are descriptive metadata, NOT routing truth.** Authoritative map (user,
  owner of the router): `claude-sonnet-5`→**5090-1 prod 27B**;
  `claude-opus-5`→**basalt (Flash Next)**; `claude-haiku-5-5`→a5000-11 dgemma.
  **Do not re-derive from names** (a prior session mapped opus-5→a5000-2 from the
  name and built a lane-swap plan on it — deleted). opus-5/basalt = the backbone,
  alive in every phase (basalt = never-down policy).
- **D-2026-11-21-03** — GPU geometry. End-state **5090 (32 GB) only**:
  fixed 22.8 (V3 artifact) + 1.35 (DSpine NVFP4 tables) + ~2.5 → KV absorbs the
  rest → ~31.5 GiB, **fits** (fork measured 8.59 GiB free on a 5090). KV
  830k→~680k tokens = **2.6× max concurrent ctx — accepted**. Dev/P0–P2 on **GPU 1**
  (a5000-2 test lane; `lane down a5000-11` → 30 GiB free, basalt stays up). 5090
  window (P3): prod → **a5000-2 (identical V3 artifact**, `model_id
  a5000-2-qwen27`, model-id flip in pi-config, no nginx change).
- **D-2026-11-21-04** — `--spec dspine` flag; draft-tokens range [1,15] mirroring
  dflash2 (`src/product/speculative_options.h`); verify/accept via the EXISTING
  `speculative_accept.h` (q = filtered softmax(logits·inv_temp), matches the
  DSpine `serial_sample_chain` contract — verified in `propose.py`).
- **D-2026-11-21-05** — **T=0.9 acceptance is LOW risk (de-risked):** DSpine's
  serial tail natively samples under the request policy
  (`serial_sample_chain(inv_temp, top_k, top_p, min_p)` in
  `dspine_mlx/dspine_mlx/propose.py:114`); prod T=0.9 + min-p 0.05 maps directly.
- **D-2026-11-21-06** — Acceptance measured on **Swift-1.5 V3 natively**: the
  `attach_dspine_head.py` computes the tower-specific TG/TSV tables from the
  target's lm_head, so a V3-fused artifact is buildable (fork ships two-tower
  variants). The **stock-target** numbers are only a **cheap kill-switch probe**.
- **D-2026-11-21-07** — The fork `mengwuzhuanshou/ninfer-windows-dspine` ships the
  bridge ops + converter as **public**; the `models/` execution dir + tests are
  **private**. **Do not depend on the author.** Public refs for the exec path:
  `Tencent/DSpine` (`dspine_mlx`, `sglang_dspine`) + merged vLLM PR #67094
  (`qwen3_dspine.py`, `dspine/speculator.py`, ~600 LOC) = the known-good design.

## OBJECTIVE

DSpine parallel drafter (Tencent, Apache-2.0, arXiv 2609.36173) as a NInfer
spec backend, serving **Swift-1.5 V3 on the 5090**. Target: >~5 tok/round at
T=0.9 on our traffic (prod DFlash2 = 4.75 @ K=7), beat DFlash2 on decode tok/s.

## PHASES (gates = human decisions; commit at each)

| Phase | Work | Gate (human) |
|---|---|---|
| **P0 de-risk** | port bridge ops → worktree of `416-ebdc67e5`; PyTorch bit-exact vectors; **stock-target accept probe** (vLLM lane, zero engine work) | kernels bit-exact **AND** probe accept ≥ ~5 tok/round, else **STOP** |
| **P1 dev** | port bridge ops (portability) + `--spec dspine` + converter; bridge-op parity tests | parity green |
| **P2 validate** | load side → draft forward + serial tail → `speculative_accept` wiring → state/graph; T=0 byte-exact vs reference + T=0.9 distribution; **V3-native A/B on a5000-2** (dflash2 K7 vs dspine K sweep) | **promote / shelve** |
| **P3 5090 window** (2–3 h) | prod → a5000-2; DSpine gate on the 5090; all agents → opus-5 (sonnet-5 down) | window **GO** (prod-downtime start) |
| **P4 promote** | point 5090-1 cfg at DSpine build; gate; re-enable; **resync a5000-2 (stale `77c2e5e2`) to prod**; docs | restore |

## MODEL ROUTING (D-02 map)

`opus-5` (basalt) = **main agent, alive all phases**. `sonnet-5` (5090) = coders
P0–P2, **relaunch on opus-5 in P3** (5090 down). `haiku-5-5` (dgemma) = P2 relief.
`deepseek-flash` = **paid, rare** — the capability escape-valve, NOT forced (the
prior "P3 deepseek-forced" plan rested on the D-02 `⚠` error; deleted). In P3,
subagent concurrency = 1 (all ride basalt).

## GPU (D-03)

- P0b/P1 (tests): GPU 1, a5000-2 test lane, **dgemma down** (30 GiB free), **basalt up**.
- P3/P4 window: prod = a5000-2 (a5000-11 down for slot); 5090 = gate target; restore both.
- GPU-1 `LnkSta` Gen5 x16 check **before** any heavy bench (Gen1 down-grade gotcha).
- Policy: **basalt never down**; one lane action at a time; RAM guard on `lane up`.

## RISKS

| # | risk | sev | phase |
|---|---|---|---|
| R1 | T=0.9 accept math | **LOW** (verified D-05) | P2 tests |
| R2 | T=0 byte-exact vs reference | LOW | P1–P2 |
| R3 | Swift-1.5-tower acceptance low | MED | P0c + P2 |
| R4 | accept < ~5 tok/round (whole build wasted) | LOW (P0c gate) | P0c |
| R5 | 7-week upstream drift (fork base ≈2026-09-30) | MED | P0a |

## COMPACTON PROTOCOL (the anti-loss method)

1. **Git tree = context store.** STATUS + ledger + phase detail live here; commit
   at every gate. The session is disposable; the tree is not.
2. **STATUS cursor first.** Every agent (incl. a post-compaction fresh subagent):
   FIRST tool call = read STATUS + phase + ledger `⚠` lines; LAST = update
   STATUS + commit. One read reconstitutes full state.
3. **Ledger append-only.** Locked facts immutable; supersede with a dated `⚠`.
   Records the *corrections* (D-02) so a post-compaction session cannot re-derive
   the basalt/opus-5 error.
4. **Subagent briefs self-contained.** The task prompt names this file + the
   outcome/frontier/locked (D-02…D-07); a compacted subagent re-reads STATUS.
5. **Commit cadence.** Every gate + every phase exit ⇒ every compaction boundary
   lands on a committed, resumable state.
6. **Keep STATUS small** (~15 lines); detail lives in the phase sections, read
   on demand. Do not use `recall` for load-bearing state — files are the source.

## REFERENCES

fork: github.com/mengwuzhuanshou/ninfer-windows-dspine (bridge ops,
`attach_dspine_head.py`, `dspine_recipe.py`; `models/` private) ·
Tencent/DSpine (`dspine_mlx` = rounding reference, `sglang_dspine` kernels) ·
vLLM #67094 (merged DSpine design, ~600 LOC) · draft weights
`HF tencent/Qwen3.8-27B-DSpine-b16` (pinned rev `30de2dad`) ·
prod lane `lanes/5090-1/config.cfg` (DFlash2 K7) · `docs/engines/NINFER_A_B.md`.
