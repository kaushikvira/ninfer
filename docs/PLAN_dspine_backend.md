# PLAN — DSpine speculative-decoding backend for NInfer

status: active · updated: 2026-11-22 · branch: feat/dspine-spec-backend (base main 70afbf33)

## SESSION START — BOOTSTRAP (you are the MAIN agent; run to final completion)

You are the MAIN (control-plane) agent for THIS build. Read **this whole
file** in the **READ-FIRST** order, then drive the phase ladder **P0 → P4 to
final completion** = DSpine serving prod on the 5090, docs landed, lanes
restored. You NEVER read bulk code / build / bench / grep / run long loops
(EXECUTION MODEL) — heavy work goes to **subagents on `opus-5`**; you launch
them, read their **digests**, gate, run lane ops, and keep the durable store
current (**update STATUS + commit at every gate**).

SETUP (first bash): `git -C /home/kv/work/ninfer checkout
feat/dspine-spec-backend` · `[ -d /tmp/ninfer-dspine-review ] || git clone
--depth 1 https://github.com/mengwuzhuanshou/ninfer-windows-dspine
/tmp/ninfer-dspine-review` · confirm `hostname`=turin (run local, do NOT ssh).

**FIRST GATE = P0c** (HANDOFF): (a) fresh vLLM + NVFP4 27B / (b) 8B / (c)
skip→P2. **Ask only THIS** — the 5090 KV gate is **MOOT by D-14** (drafter-swap
−1.06 GiB, KV **GAINS** 0.35×→0.41×). Do NOT self-decide gates. **Never self-approve a promotion** (P2→P3, P4) — the
user decides at each gate. Then run PHASES; launch fan-out with the
COPY-PASTE subagent prompts; fan-out rides **`opus-5`** (m2: the 5090 runs
prod, `sonnet-5` is NOT idle), GPU-1 serialized, one writer/file, check running
agents before launch. Obey every `⚠` in the ledger. On any compaction you
resume from STATUS (you should rarely compact — you are control-plane).

## READ-FIRST (compaction re-entry)

This file is the **durable context store**. The session transcript and
observational memory are **disposable** — both lanes (opus-5, sonnet-5) are
262k and a 10–14 day build crosses compaction. On resume: read the **STATUS**
block → the phase section → the **DECISION LEDGER** (esp. the `⚠` entries),
**then act**. Do NOT re-derive from conversation. Append to the ledger; never
rewrite a locked line — supersede it with a dated `⚠` line.

## STATUS (cursor — update + commit at every gate)

- phase: **P0 — LAUNCHED** (P0a port-ops + P0b parity-ref, both on `opus-5`);
  P0c **SKIPPED by user decision (D-17)** → the **P2 V3-native A/B is the
  accept gate**
- done: plan; 4 reviews (self / verifier `mwpxt2ws` / D-14 drafter-swap /
  `mwpya6mp` coherence); spine verified (D-01 license, D-05 T=0.9, D-02 map,
  5090-fits, D-14 drafter-swap); fixes → **D-10..14**
- next: **P0 gate = P0b parity** (relaunch `mwpzha6z-2324`; DIVERGENT ⇒ STOP)
  → P1 dev (exec path + converter)
- P0a: **DONE/green** — `feat/dspine-port` `1138e8b2` (14 files, +746/−5),
  full host build 743/743 `sm_120a`, 6 ops linked, `--spec dspine` [1,15]
  validates; R5 drift = 5 rename families, survivable. **Not yet merged.**
- fail-safe: P0c skipped ⇒ **no cheap early accept abort**; the first accept
  numbers now arrive at **P2 V3-native**. If R4 looks real, the K-sweep sweep
  on a5000-2 (P2) is the earliest abort point.
- open-risks: R5 drift (8–10 wk, MED) · R3 V3-tower accept (P2) · R4 accept
  <~5 tok/round (P2 now owns it)
- last-gate: **P0c decided 2026-11-22 → (c) skip** (`D-2026-11-21-17`)
- artifacts: port worktree `/tmp/ninfer-dspine-port` (branch `feat/dspine-port`
  off `f9c00d50`, = main + docs) · fork clone `/tmp/ninfer-dspine-review` ·
  Tencent reference clone `/tmp/dspine-ref`

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
  `⚠` **SUPERSEDED by D-14 (whole entry)** — its additive fixed-cost
  decomposition is void; the canonical VRAM model is D-14 below. (The old
  830k/680k/2.6× figures were the stale-8.9-KiB/tok bug.)
  **CANONICAL 5090 budget (D-14 drafter-swap, USE THIS):** DSpine **replaces**
  DFlash2 → weights `22.80 − (DFlash2 2.407) + (DSpine 1.35) = 21.74 GiB` +
  `~3.0` graphs/ctx → **KV ≈ 7.6 GiB → ~429k tok (0.41× concurrent, ABOVE
  today's 0.35×)**. Net **−1.06 GiB** vs prod. The 26.7 / additive figures
  below are a **CONSERVATIVE FLOOR, not the net cost**. Dev/P0–P2 on **GPU 1**
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

- **D-2026-11-21-08** — GPU P3/P4 co-sizing. **P3/P4 prod-on-a5000-2 sizing =
  a5000-2 KV-capped ≈30** (a5000-2 30 + basalt 39 + tts 3 = 72/72 tight; drop
  tts → 69 fits). NOT the 36 dev/test slice.
- **D-2026-11-21-09** — **Resync a5000-2 to prod `416-ebdc67e5` before the
  P3/P4 window** (prod-on-a5000-2 = prod-parity, not the stale `77c2e5e2`); P4
  resyncs a5000-2 back to the DSpine build.

- **D-2026-11-21-10** — `⚠` **RE-BASE D-03** (supersedes its 830k/680k/2.6×,
  stale 8.9 KiB/tok). Verified: 18,432 B/tok (16 full-attn of 64 × 4 kv-heads ×
  2 (k+v) × (128 data + 16 scale) nvfp4); **live** device pool
  `9,011,277,824 B / 18,432 = 489k` tok (host 53.7 GiB). End-state
  `32.6−(22.8+1.35+~0.7)=5.2 GiB ≈ 300k tok`. Concurrent set = 4×262,144 = 1.05M
  → today **0.46×**, end-state **0.28×** (NOT 2.6×). **`⚠` the `0.28×` +
  `26.7 < 32 GiB` are the ADDITIVE FLOOR — the 300k / 0.28× are SUPERSEDED by
  D-14's swap (net −1.06 GiB → ~429k / 0.41×). The "fits" verdict SURVIVES —
  more comfortably under the swap.**
- **D-2026-11-21-11** — `⚠` **P0c infeasible as written** (B1). a5000-2 is a
  NInfer/`dflash2` lane (no DSpine in any NInfer build → STATUS's "lane up
  a5000-2 = the probe" self-contradicts). DSpine merged vLLM **2026-11-16**;
  fleet vLLM nightly ≈2026-10-24 → P0c needs a **fresh nightly (≥Nov-16,
  ≥34 GB pull)**; the PR validates **BF16 on H100** (54 GiB, fits no GPU) → P0c
  needs an **NVFP4 27B + FLASHINFER (SM120, unvalidated → fallback plan)**; the
  **131072 ctx cap** = probe-validity limit. Fix: P0c = explicit fresh-vLLM-
  nightly + NVFP4 target + 131k, **NOT zero-cost**.
- **D-2026-11-21-12** — **P3 5090-window co-sizing** (refines BLOCKER-1). Measured
  basalt 47 + tts 1 (not 39/3), a5000-11 down → GPU-1 free **28.5** (not 30/35).
  a5000-2 ~22 non-KV + 2 headroom + **4.5 KV = 250k tok = 0.24× concurrent (half
  prod's 489k)** + 23 GiB host cache → **fits 70.8/71.7, no tts drop**. The prod
  stand-in runs **HALF prod's device KV** (more preemption/recompute) — state
  explicitly + P4 restore check.
- **D-2026-11-21-13** — Re-pin draft weights to rev **`260a6c23`**
  (`DSpineDraftModel` config); `30de2dad` declares `DFlashDraftModel` (stale).

- **D-2026-11-21-14** — `⚠` **DRAFTER-SWAP — D-10's additive model is WRONG.**
  DSpine **replaces** DFlash2; measured (artifact bindings, physical bytes, 0
  overlap): **DFlash2 drafter 2.407 GiB** (2.074 + 0.333 proposal head) vs
  **DSpine 1.35 GiB** → **−1.06 GiB — DSpine is CHEAPER than prod, not
  costlier.** So the 22.80 artifact **shrinks** → device KV **GROWS** ~62k tok
  (measured model: today ~367k / live-pool 489k → **~429k, 0.35×→0.41×** the
  concurrent set). **DSpine end-state is BETTER than DFlash2 today** on VRAM +
  KV. Closes the KV gate (D-10 re-open).

- **D-2026-11-21-15** — `⚠` **THE V3 DFLASH2 DRAFTER IS STOCK, NOT TAILORED TO
  V3.** Decisive: `pipeline/dflash2-retrain/README.md` is a pipeline built to
  retrain a drafter for "the NInfer Swift-1.5 V3 27B target" → it is NOT RUN
  (no `LOG.md`/`results/`), no promoted-retrained-drafter record → prod runs
  the **stock** drafter. (a) z-lab card = "mirror of `incoai/Qwen3.8-27B-
  DFlash2`", base `Qwen/Qwen3.8-27B`; (b) our taps `[5,19,33,47,61]` = the
  *retrain recipe's* stock geometry = `dflash.py:120` `2560×num_hidden_layers(64)`
  = **stock 27B's 64 layers**; (c) v3 recipe = quantize→convert→`dflash2 +
  proposal head` = **repackage**, no drafter-train; (d) the retrain README
  itself: *"the **unmodified drafter** scored 0.3-1.1% acceptance"* on a
  fine-tuned tower (stock drafters **degrade on post-towers**).
- **D-2026-11-21-16** — `⚠` **SUCCESS ODDS RAISED (supersedes the 55-65% read).**
  Both drafters stock → the **stock→V3 gap hits DSpine AND the DFlash2 baseline
  EQUALLY** → the **relative delta transfers → R3 is NOT decisive** (my
  "tailored baseline is a harder bar" claim is void). **`block_size` asymmetry:**
  the z-lab card's DFlash2 base drafter = **`block_size 8` (K≈7)**; prod
  `NINFER_DRAFT_TOKENS` 7→15 (2026-10-16) → **prod runs a K≈7-trained drafter at
  K=15 → under-trained beyond ~7 → DFlash2 headroom-limited.** **DSpine
  (block-16-trained, card 5.58/5.11) is K-matched → headroom DFlash2 lacks.**
  SUCCESS ~**65-75%**. **Bonus: `pipeline/dflash2-retrain/` = a stock→V3 retrain
  capability (retrain the drafter for V3) — a DFlash2 path AND the DSpine
  ceiling-raiser.**

- **D-2026-11-21-17** — **P0c DECIDED by the user (gate): option (c) — SKIP the
  stock-target vLLM probe.** No fresh vLLM nightly, no NVFP4-27B/FLASHINFER
  SM120 gamble. The **P2 V3-native A/B on a5000-2 is the accept gate**.
  Consequence (accept for it explicitly): R4 has **no cheap early abort** — the
  first real accept numbers arrive only after P1/P2 exec-path work lands.
  P0 is therefore **parity-only** (P0b bit-exact is still the kill-switch for a
  broken fork).

- **D-2026-11-21-18** — **P0b oracle = the COMPILED OP, not a CUDA transcription.**
  P0b attempt 1 (`mwpykgtv-4116`, opus-5) spent its budget writing
  `fork_model.py` = a line-cited **hand transcription** of `kernels.cu`, then
  comparing it to a FP64 SGLang reference — i.e. **model-vs-model**, and it
  ended **55 fail / 4 pass, uncommitted**. Its headline "structural `finish`
  divergence (85 % of bf16 writes, `saturated`/`norm_eps_zero`)" is therefore
  **inconclusive by construction**: the transcription is as much a suspect as
  the kernel. Locked: **P0b links `libninfer_ops.so` from the P0a build
  (`1138e8b2`, `build/p0a`) and calls the real kernels on GPU 1**, vs
  `Tencent/sglang_dspine/kernels/reference.py`. Per-op verdicts are
  **BIT-EXACT / EQUIVALENT-ORDER (named reduction-order diff, in-band) /
  DIVERGENT** — only DIVERGENT is a kill-switch. Relaunched `mwpzha6z-2324`.
  Rule added: **a red parity suite is never greened by widening a tolerance**;
  and subagents **commit while red** (attempt 1 lost all its work).

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

## EXECUTION MODEL (context discipline)

**Main agent = control plane.** Decision, gates, lane ops, STATUS updates,
launching + reading digests, and the rare one-line inspection. It stays
**nimble** — small context, rarely compacts, so the *decision* context never
drifts. Main agent NEVER: reads bulk C++/CUDA/reference files, runs build/test
loops, dumps large tool output, runs the bench, greps the tree.

**Subagents = data plane.** Every context-heavy unit → subagent (own 262k, does
the bulk, returns a digest). The heavy work burns *their* context; mine stays
small.

**Digest contract (the anti-compaction lever).** Subagents return:
`DONE · FILES(file:line) · DECISIONS(why) · RESULT(num/green/red) ·
NEXT · BLOBS(file paths, NOT content)`. NO raw logs/build output/file dumps.
I read the digest + the STATUS/commit — never the bulk. BLOBS are pointers: the
work lives in the worktree/commit, readable on demand.

**Per-phase map:**

| Subagent (role) | model | work | returns |
|---|---|---|---|
| **port-ops** | `opus-5` (basalt — `sonnet-5` runs prod, NOT idle: m2) | port `dspine_bridge` to a worktree of `416-ebdc67e5`; patch drift; build | build green + drift notes + commit |
| **parity-ref** | sonnet-5 | PyTorch reference, bit-exact vectors, test file | pass/fail + files |
| **impl-<stage>** | sonnet-5 | exec path: load/forward/tail/accept (1 stage each) | diff + test green + decisions |
| **test-runner** | sonnet-5 | build+test loop; fix; re-run | digest w/ final result |
| **bench-runner** | sonnet-5 | run gate+A/B, parse jsonl | numbers + pass/fail |
| **verifier** | opus-5 (basalt) | accept-math review | pass/fail + reasoning |

**Rules.** Fan-out read-only; one writer per file; worktree isolation on code
(squash-merge by me). Long loops (build, bench, test-iterate) = subagent —
NEVER main. **If main reads >1k lines of bulk, it broke the contract.**

**Main's loop:** `read STATUS → fan-out subagents (parallel, worktree) → read
digests → commit (subagents commit worktree; I squash-merge) → update STATUS →
gate (human)`. One read + N digests per phase — bounded main context.

## HANDOFF — instructions for the next main agent (READ-FIRST)

**Re-entry:** STATUS → the ledger `⚠` lines (corrections — **obey**) →
**EXECUTION MODEL** → **HANDOFF** → act. You are the control plane: launch
subagents, read digests, gate, run lane ops. You do NOT read bulk code, build,
bench, grep, or run long loops (EXECUTION MODEL).

**State.** Plan is durable on `feat/dspine-spec-backend` (base `main`
`70afbf33`), docs committed. Spine VERIFIED true — do NOT re-litigate: D-01
(EAGLE-4 license anti-porting = blocking), D-05 (T=0.9 via native
`serial_sample_chain`), D-02 (model→lane map, self-consistent), 5090-fits
**(D-14 drafter-swap model: 21.74 weights → ~429k KV / 0.41×; the 26.7 additive
figure is a conservative FLOOR — size P3 off D-14, not 26.7)**. **Do NOT
re-derive the map from `models.json`** — the
`name:` fields are stale metadata, NOT routing truth (the D-02 `⚠` correction).
Obey every `⚠`: D-02 map · D-10 (2.6× FALSE — units bug; **additive model
SUPERSEDED by D-14**) · D-11 (P0c infeasible as written) · D-12 (P3 stand-in =
half prod's KV) · D-13 (pin rev `260a6c23`) · **D-14 (drafter-SWAP: DSpine
1.35 < DFlash2 2.407 GiB = −1.06 → DSpine CHEAPER than prod, KV GAINS)**.

**Model routing (D-02 + m2).** `opus-5`(basalt) = backbone, alive ALL phases.
`sonnet-5`(5090) runs PROD through P0–P2 → **NOT idle — do NOT schedule fan-out
on it.** P0–P2 fan-out rides **`opus-5`** (shared with the main agent); `haiku-
5-5`(a5000-11) is DOWN in P0–P2. `deepseek-flash` = paid, rare. GPU-1 is a
**shared, serialized** resource (one lane-agent at a time).

**FIRST ACTION — ONE gate: P0c (D-11) (do NOT self-decide; do NOT ask the KV
gate — it's MOOT).**

> **VRAM/KV gate — CLOSED (no sign-off needed).** D-14: DSpine *replaces*
> DFlash2 (2.407 GiB) with **1.35 GiB** = **−1.06 GiB** → device KV **GROWS**
> (0.35×→0.41×, above today). **DSpine is CHEAPER than prod on VRAM.** (Supersedes
> the D-10 additive model.)

1. **P0c (D-11).** Infeasible as written (a5000-2 is a NInfer/dflash2 lane — no
   DSpine; the probe must be vLLM; fleet nightly ≈Oct-24 < the Nov-16 merge;
   BF16/H100 fits no GPU). **ASK, options:**
   - **(a)** invest the probe (fresh vLLM nightly ≥Nov-16 + **NVFP4 27B** +
     FLASHINFER/SM120 [**unvalidated** → 8B fallback] + 131k ctx cap)
   - **(b)** 8B stock-target fallback (weaker kill-switch)
   - **(c)** **skip P0c → the P2 V3-native A/B is the gate** (main's lean: P0c's
     signal is weak, 131k cap + unvalidated SM120; P2 measures the thing that
     matters, V3-native)

**Then, on go → launch P0:**
- `port-ops` (model `opus-5`) — port the public DSpine **1596 LOC** → a
  worktree of `416-ebdc67e5`, patch drift, build. *(R5 drift)*
- `parity-ref` (model `opus-5`, parallel) — PyTorch bit-exact vectors vs
  `Tencent/DSpine/dspine_mlx/.../reference.py`. **Kill-switch: not bit-exact ⇒
  STOP.**
- fan-out = `opus-5` only; **check running agents before launch; GPU-1
  serialized; one writer per file.** P0c follows per the decision.
- **P0 gate = parity green** (P0b) ⇒ P1. Commit STATUS at every gate.

## SELF-REVIEW (2026-11-21, main) — fixes to apply

- **BLOCKER-1 (D-03 GPU co-sizing).** P3/P4 prod-on-a5000-2 full-ctx `36 +
  basalt 39 + tts 3 = 78 > 72 GiB` → **doesn't fit.** The doc's "30 GiB free"
  is the *dev* rule (a5000-2 test). Fix: P3/P4 run a5000-2 **KV-capped ≈30**
  (`30+39+3=72` tight; `drop tts → 3` → `69` fits). P3/P4 sizing =
  "a5000-2 ≈30, drop tts if tight", NOT the 36 test-slice.
- **MAJOR-2 (P0c vLLM pin).** DSpine merged vLLM 2026-11-16; fleet vLLM lane
  images are older → P0c needs a **newer vLLM nightly (budget)**; and the
  DSpine `--context-length 131072` caps the **stock-target probe at 131k ctx**
  (V3-native P2 carries the 262k measurement).
- **MAJOR-3 (P4 sequencing).** P3 prod-on-a5000-2 = stale `77c2e5e2` (3 builds
  behind) → prod degraded the window. **Resync a5000-2 to `416-ebdc67e5`
  before** the window; P4 resyncs it back to the DSpine build.
- **MINOR-4.** Confirm the bench computes tok/round **identically** for
  DFlash2 (4.75) vs DSpine, same corpus.
- **SOUND:** D-02 map (self-consistent, opus-5=basalt survives every phase),
  D-05 T=0.9, EAGLE license, compaction/context model.

## VERIFIER REVIEW (2026-11-21, `mwpxt2ws-a170`, opus-5, fresh context)

- **MAJOR-1 = HALLUCINATION — DISCARDED.** The verifier cited
  `~/work/seiso-vira-rocks-gateway/docker/litellm/config.yml` (with line refs)
  refuting D-02. **That repo does not exist on turin** (gateway = vira.rocks
  box, per AGENTS.md, NOT turin); no litellm `config.yml` in any repo; the only
  in-repo reference is `models.json` `name:` (D-02 says metadata). D-02
  reaffirmed — owner-given, not repo-verifiable. (This is why the verifier is
  itself verified: a fabricated refutation of a locked owner fact.)
- **ADOPTED minors:**
  - **#1** DSpine `--context-length 131072` vs our 262144 = **2×** → the stock
    target probe **cannot run the 262k slice** (OOM). P0c: probe ≤131k only;
    262k lives at P2 V3-native.
  - **#2** The gate (`bench/tests/gate.py`) checks needles/tools — **no
    acceptance length.** P2 gate = **A/B decode result**, not `gate.py`.
  - **#3** P4 "resync a5000-2 to prod" is **incoherent** (prod build = the
    DSpine build). Fix: **rebuild a5000-2 from `feat/dspine-spec-backend`**
    (full image build) — that IS the promotion; a resync is a no-op.
  - **#4** **LOC undercounted.** Public DSpine = **1596 LOC** (not 720; that
    was the bridge subset). Our DFlash2 draft = `draft.cpp/.h` **589 LOC**
    peer + **234 dflash refs across 41 files** = the exec path (589 peer +
    41-file wiring). Effort **10–14 → ~15–20**; P0 de-risks, it does not
    guarantee a number. **P0 = go/no-go, not ~15–20 as a promise.**
- **VERDICT:** no blockers; D-02 reaffirmed; **plan SOLID** with the committed
  fixes (self-review BLOCKER-1/MAJOR-2-3, verifier #1–#4). Ready for P0.

## VERIFIER DETAIL (2026-11-21, full report — supersedes the summary)

Confirmed: **M1** (2.6× false → D-10), **B1** (P0c infeasible → D-11), **M2**
(P3 stand-in KV-starved → D-12). Minors (adopted): free **28.5** GiB not
30/35; "5090 idle" false (it runs prod through P0–P2); **haiku-5-5
unavailable as P2 relief** (a5000-11 is down in P0–P2 — use sonnet-5 for P2
fan-out); **4.75@K=7** = the K=15 number (K=7 ≈ 4.0–4.5; live accept 37.7% →
~2.6 tok/round on a hot-cache window → the ">~5" bar is a **bigger gap**);
R5 = **8–10 wk** drift (not 7). **Fan-out rule (m6):** main checks running
agents (`task_control list`/STATUS) before launch — no duplicate fan-out on a
resource; **GPU-1 serialized**; ledger is the sole append-only section.
EFFORT 10–14 = **optimistic**, schedule risk = **P0c + R5**, not the port.

## COPY-PASTE PROMPTS (for the next agent — prompts ride the doc, not the chat)

### P-MAIN — bootstrap the next main agent (new pi session)

```
You are the MAIN agent for the DSpine spec-decoding build.

Branch `feat/dspine-spec-backend`, repo `/home/kv/work/ninfer`, host `turin`
(run local, do NOT ssh).

READ-FIRST, in order, before acting:
1. `docs/PLAN_dspine_backend.md` -> STATUS, then DECISION LEDGER (obey every
   `⚠`), then HANDOFF, then EXECUTION MODEL, then PHASES.
2. Do NOT re-derive the model->lane map from `pi-config/models.json` (its
   `name:` fields are metadata, NOT the router; see the D-02 `⚠`).
3. Do NOT re-litigate the verified spine: D-01 (EAGLE-4 license blocks
   porting), D-05 (T=0.9 native), D-02 (map), 5090-fits. For any VRAM/KV sizing
   USE D-14 (drafter-swap: 21.74 weights, ~429k KV / 0.41×, ABOVE today); the
   26.7 additive figure is a conservative FLOOR, not the net cost.

FIRST GATE: ask the user ONE decision - P0c (D-11): (a) fresh vLLM + NVFP4
27B, (b) 8B fallback, or (c) skip -> P2 is the gate (lean). Do NOT ask the KV
gate - it is MOOT (D-14: drafter-swap -1.06 GiB, KV GAINS). Do NOT self-decide.

Model routing (m2): fan-out rides `opus-5` (basalt) - `sonnet-5` = the 5090
runs PROD (NOT idle); `haiku-5-5` down in P0-P2; `deepseek-flash` paid/rare.
GPU-1 serialized; check running agents before launching; one writer/file.

You are CONTROL-PLANE only: launch subagents, read digests, gate, lane ops.
You NEVER read bulk code, build, bench, grep, run long loops. Bounded context:
you should rarely compact.

On go: launch `port-ops` + `parity-ref` (P0a/P0b prompts above) as subagents
on `opus-5`. P0 gate = parity bit-exact, else STOP. Commit STATUS at every gate.
```

### P0a — `port-ops` subagent

```
You are a READ+WRITE coder subagent (P0a).

GOAL: port the DSpine bridge ops into a new Git WORKTREE of commit
`416-ebdc67e5` and get them to compile - proof the 8-10wk base drift is
survivable. Do NOT touch the main tree, the lane, the prod lane, a build, a
lane, a service, a systemd unit, a cron, a Docker image.

SOURCE (fork, public, on disk): `/tmp/ninfer-dspine-review` (master) -
- `src/src/ops/dspine_bridge/` (dspine_bridge.cpp, kernels.cu, launch.h)
- `src/include/ninfer/ops/dspine_bridge.h`
- `src/src/ops/linear/bf16/shapes/dspine_bridge.cu`
- `converter/tools/convert/qwen3_8_27b/dspine_recipe.py`,
  `scripts/attach_dspine_head.py`
TARGET: the worktree of `416-ebdc67e5` (same repo). C++/CUDA engine; mirror
the surrounding op conventions (self-contained op, CUDA-graph-capturable, no
internal alloc).

TASK: port the bridge ops + the linear shape + the CMake wiring + the
`--spec dspine` flag into the worktree. Resolve the 8-10wk API drift
(tensor/weight/layout/graph APIs). Compile on Linux; report a clean build.
If a kernel uses a Windows/CUDA-13.1-specific construct, port it to CUDA-12
`sm_120a`.

RETURN a DIGEST (<=25 lines) ONLY - no file dumps/logs:
DONE · FILES(worktree paths) · DRIFT(resolved API breaks, file:line) ·
BUILD(clean/failed + the failing symbol) · RISKS(kernel constructs that may
not port) · NEXT. Commit in the worktree; report the SHA.
```

### P0b — `parity-ref` subagent

```
You are a READ+WRITE test subagent (P0b). The P0 KILL-SWITCH.

GOAL: prove the fork's DSpine bridge kernels are bit-exact against the
reference. If NOT bit-exact => the fork is broken => STOP the project. Report
PASS/FAIL per op.

REFERENCE (the rounding points the kernels must match):
- `dspine_mlx/dspine_mlx/propose.py` (topk_ids, filter_probs, serial_sample_chain)
- `dspine_mlx/dspine_mlx/sglang/kernels/reference.py` (+ `bridge.py`,
  `serial.py`, `batched_tail.py`)
Kernels under test (the ops to match):
`/tmp/ninfer-dspine-review/src/include/ninfer/ops/dspine_bridge.h` -
`dspine_prepare_read`, `dspine_normalize_codes`, `dspine_message`,
`dspine_finish`, `dspine_serial_scores`, `dspine_serial_chain`.

TASK: a PyTorch (CPU, BF16) reference of each op + test vectors (fixed seed,
realistic shapes); state the exact BF16 rounding point per op. Output: a
runnable parity test (PyTorch ref vs the documented op math). If feasible,
link the compiled CUDA op (`416-ebdc67e5` build) and compare on the SAME
vectors (bit-exact BF16) - the CUDA-vs-reference comparison is the real test.

RETURN a DIGEST (<=25 lines): DONE · FILES · PARITY(per-op bit-exact Y/N) ·
VERDICT(kernels sound/NOT - the kill-switch) · RISKS(rounding points where
the fork may deviate). No file dumps.
```

## REFERENCES

fork: github.com/mengwuzhuanshou/ninfer-windows-dspine (bridge ops,
`attach_dspine_head.py`, `dspine_recipe.py`; `models/` private) ·
Tencent/DSpine (`dspine_mlx` = rounding reference, `sglang_dspine` kernels) ·
vLLM #67094 (merged DSpine design, ~600 LOC) · draft weights
`HF tencent/Qwen3.8-27B-DSpine-b16` (pinned rev `30de2dad`) ·
prod lane `lanes/5090-1/config.cfg` (DFlash2 K7) · `docs/engines/NINFER_A_B.md`.
