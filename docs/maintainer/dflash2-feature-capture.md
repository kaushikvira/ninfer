# DFlash2 feature-capture / dump mode (spec)

Status: **implemented** — per-layer feature + per-position target logits (top-k) + token ids +
position capture, default-OFF, env-gated. Tracking: issue #1 on `kaushikvira/ninfer`.

## Summary

NInfer computes the DFlash2 teacher features internally (via `DFlashFeatureSink`) on every
prefill, but discards them after feeding the drafter. There is **no way to export them**.
This blocks training a custom / workload-matched DFlash2 drafter against the exact served
target.

## Implementation

- **`src/models/qwen3_5/execution/dflash_tfdump.{h,cpp}`** — a model-agnostic dump writer
  (ported from capy-flash `qwen4_tfdump`, MoE-specific bits dropped). Writes the tfdump/1
  layout: `hidden_LXX.bin` (per tapped layer, compact records), `positions.npy`, `tokens.npy`,
  `topk.bin` / `logits/pNNNNNN.npy`, `meta.json`.
- **Wired into the dense `DFlashFeatureSink` tap** (the qwen3.5 27B path, *not* the qwen4 MoE
  path). The sink is a passive 4-field POD plus an optional `DFlashTfDump* dump` (null = off);
  the program (`RankProgramImpl::tf_dump`) owns the dump object.
  - `TextRun::capture_features(layer)` (`execution/text.cpp`) D2H-copies each tapped layer
    residual to disk after the existing feature copy/scatter (the residency_slots path is
    untouched).
  - `RankProgramImpl::begin_target` (`program/decode.cpp`) creates the dump lazily on the
    first generation prefill and attaches it to the prefill sink **only**, so the hooks never
    fire under CUDA-graph capture (graphs run decode/verify only) and never on the scoring
    path.
  - `RankProgramImpl::launch_decision` (`program/decision.cpp`) captures the per-position
    target logits (the supervision) for the whole chunk: the standard prefill projects only
    the *last* position, so the dump path projects ALL positions through the output head and
    writes `topk.bin` (or full `logits/pN.npy`), appends the chunk's token ids to
    `tokens.npy`, and writes `meta.json`.
- **Default-OFF**: `DFlashTfDump::from_environment` returns null unless
  `NINFER_DFLASH_TFDUMP=<dir>` is set; every hook is null-checked, so a prod lane carries
  zero overhead and the flag is never needed.
- **Env**: `NINFER_DFLASH_TFDUMP=<dir>` (on), `_LAYERS` (default all tapped; `all`/`none`/
  comma list of layer ids), `_TOPK=<k>` (top-k logprobs, default 0 = full logits),
  `_COMPACT=1` (default), `_LOGITS=0` (disable logits — note: enabled unless exactly `"0"`).

## Why

- The shipped DFlash2 drafter (`incoai/Qwen3.8-27B-DFlash2`) was trained against **vanilla
  Qwen3.8-27B**. Serving a fine-tune (e.g. Swift-1.5, NVFP4) introduces a target mismatch:
  the drafter reads tap-layer features from a target whose hidden states differ from the ones
  it was trained on.
- Training our own drafter against the **served target + our workload** needs, per sequence:
  (a) tap-layer hidden states `[5,19,33,47,61]`, (b) target next-token logits, (c) token ids,
  (d) positions, (e) a sequence id.
- Capturing from **NInfer itself** (vs a vLLM/SGLang capture server) yields features
  **bit-identical to what we serve** → zero train/serve mismatch. A separate-framework capture
  server introduces a subtle NVFP4 kernel/quant mismatch.

## Reference implementation (already exists in a fork)

`ValerioDolci/capy-flash` (serves Qwen3.8-27B + Flash-Next) implements this for **MTP**:

- `src/models/qwen3_5/execution/qwen4_tfdump.{h,cpp}` — env-gated, off-by-default, per-position
  `.npy` dump ("tfdump/1"): `tokens.npy`, `logits/pN.npy` (f32 [V]), `hidden/LXX/pN.npy`
  (f32 [S*H] residual after layer XX), `hidden/final/pN.npy`, `meta.json`.
  Knobs: `NINFER_QWEN4_TFDUMP=<dir>`, `..._LAYERS=<comma-list>|all|none`, `..._TOPK`,
  `..._COMPACT`, `..._LOGITS`. **Hidden taps run only in eager prefill** (each tap syncs the
  stream); gated behind `NINFER_QWEN4_VALIDATION=1` so it cannot reach a serving process.
  Hooks the same tap infrastructure (`Qwen4Taps` / `capture_layer`) as `DFlashFeatureSink`.
- `tools/mtp_refit/` — full `gen_dump → train → make_artifact → bench` pipeline. Measured
  **+4.6% to +18.4% tok/s** from refitting the spec head against the engine's own numerics;
  lossless identity check (drafts on == drafts off) passes.

## Port notes (this fork)

- capy-flash's `qwen4_tfdump` is **coupled to the MoE path** (`Qwen4Taps` in the Qwen4Exp /
  Flash-Next forward); it does not fire on the dense 27B path. We do not port the MoE tap
  wiring — this fork's dense path already has the equivalent tap: `DFlashFeatureSink`
  (`execution/text.h`) consumed by `TextRun::capture_features` (`execution/text.cpp`).
- The dump is an **additive consumer** of that tap: it D2H-copies each tapped residual as it
  is captured and writes it to disk. It never changes the generated tokens.
- The per-layer D2H happens on the execution stream and syncs it, so capture runs
  **eager-prefill only** by construction (the dump attaches to the generation prefill sink;
  decode/verify — the only graph-eligible phase — carries a null dump pointer).

## Constraints / correctness

- **Default OFF**; every hook is gated on the dump pointer being non-null.
- Keep the dump out of the hot path when disabled (no overhead when off).
- Known limitation: the compact `hidden_LXX.bin` files are truncated on first write per
  process and appended afterwards; `positions.npy` is derived from the on-disk bytes
  (relative 0..T-1) and is not reset between separate prefill requests in one process —
  fine for a single capture run, which is the intended use.

## Open questions

- Dump-format → SpecForge/speculators input-schema converter (the one genuinely new piece).
  vLLM `speculators` + SGLang `SpecForge` both train DFlash2; SpecForge ships a
  `qwen3.8-27b-dflash2` config + recipe. (MTP refit ≠ DFlash2 block-diffusion objective, so
  reuse an off-the-shelf DFlash2 trainer, not capy-flash's MTP `train.py`.)
- License/provenance for ported capy-flash dump code (check capy-flash's LICENSE; its MTP
  layer credits `ursuciprian/…` Apache-2.0 — the dump writer may be independently authored).
- Sample-rate / max-sequences gating for live-traffic (workload-matched) capture.

## Acceptance criteria

- A dump flag/env produces, per sequence: 5-layer tap feature + target top-k logprobs + token
  ids + positions.
- Dumped features are **bit-identical** to the drafter's serve-time features (identity check).
- Default OFF; zero overhead when disabled.
- End-to-end: dump → DFlash2 train → artifact → bench, with a measured acceptance/tok-s delta.
