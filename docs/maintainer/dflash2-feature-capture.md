# DFlash2 feature-capture / dump mode (spec)

Status: **implemented (v1)** — per-layer feature + position capture, default-OFF, env-gated. Per-position target logits (the supervision) are the v2 step (dedicated dump path). Tracking: issue #1 on `kaushikvira/ninfer`.

## Summary

NInfer computes the DFlash2 teacher features internally (via `DFlashFeatureSink`) on every
prefill, but discards them after feeding the drafter. There is **no way to export them**.
This blocks training a custom / workload-matched DFlash2 drafter against the exact served
target.

## Implementation (v1)

- **`src/models/qwen3_5/execution/dflash_tfdump.{h,cpp}`** — a model-agnostic dump writer
  (ported from capy-flash `qwen4_tfdump`, MoE-specific bits dropped). Writes the tfdump/1
  layout: `hidden_LXX.bin` (per tapped layer, compact records), `positions.npy`, `tokens.npy`,
  `topk.bin` / `logits/pNNNNNN.npy`, `meta.json`.
- **Wired into the dense `DFlashFeatureSink`** (the qwen3.5 27B path, *not* the qwen4 MoE
  path): `capture_layer` D2H's each tapped layer residual to disk; `capture_positions` writes
  `positions.npy`. The sink owns a `std::unique_ptr<DFlashTfDump>` (null = off).
- **Default-OFF**: `DFlashTfDump::from_environment` returns null unless
  `NINFER_DFLASH_TFDUMP=<dir>` is set; the hooks are null-checked, so a prod lane on
  `ninfer-clean:local` carries zero overhead and the flag is never needed.
- **Env**: `NINFER_DFLASH_TFDUMP=<dir>` (on), `_LAYERS` (default all tapped), `_TOPK=<k>`
  (top-k logprobs), `_COMPACT=1` (default), `_LOGITS=0` (disable logits).
- **v2 (next)**: per-position target logits (the supervision) — the standard prefill only
  projects the *last* position to logits, so a dedicated dump path (project all positions,
  like capy-flash `score-windows`) is needed. `finalize_dump()` is already stubbed on the sink
  for this.

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

## Port plan (VERIFIED against capy-flash + this fork, 2026-10-25)

### Compatibility verdict
- capy-flash's `qwen4_tfdump` is **coupled to the MoE path**: `Qwen4Taps` lives in
  `qwen4_forward.h` (Qwen4Exp / Flash-Next), invoked at `qwen4_forward.h:442/467`. It does
  **not** fire on the dense 27B path. Do NOT port the MoE tap wiring.
- **This fork's dense path already has the equivalent tap**: `DFlashFeatureSink`
  (`text.h:51`), `capture_layer` (`text.cpp:167`), invoked at `text.cpp:1139` (eager) and
  `text.cpp:1143-1152` (CUDA-graph `feature` lambda). It is **graph-compatible** (captures
  into a pre-allocated buffer, no in-graph D2H) — cleaner than capy-flash's eager-only taps.
- **The prefill signature already threads everything the dump needs**: `prefill_chunk(...,
  Tensor& logits, Tensor& target_tokens, DFlashFeatureSink& sink)` (`text.h:151`). So
  features + logits + token ids + positions are all in scope at one site.

### Therefore the port is: reuse our tap, port only the model-agnostic dump writer
1. **Port the dump serializer** from `qwen4_tfdump.cpp` (model-agnostic, ~300 lines):
   `write_npy` (.npy v1.0), `from_environment` (env parse), per-position `hidden/LXX/pN.npy`,
   `write_tokens` (`tokens.npy`), top-k logprobs (`topk.bin`, `..._TOPK`), `write_meta`
   (`meta.json`). Rename env `NINFER_QWEN4_TFDUMP*` → `NINFER_DFLASH_TFDUMP*`.
2. **Add a post-pass dump hook**: after prefill completes (the `DFlashFeatureSink.features`
   buffer is populated on GPU), do the D2H copy + write to disk. This is the ONLY D2H; the
   in-graph capture stays as-is. Serializes: 5-layer feature + target top-k logprobs + token
   ids + positions + seq id.
3. **5-layer feature**: `DFlashFeatureSink.features` already holds the 5 tap layers
   `[5,19,33,47,61]`. Confirm the on-buffer layout (concat `[S*H,5]` vs `[S,H*5]`) and emit it
   bit-identical to the drafter's serve-time feature.
4. **Gate**: env-gated, **default OFF**. (Post-pass D2H does not change generation, so it is
   safer than capy-flash's in-graph sync; the strict `NINFER_QWEN4_VALIDATION` opt-in is
   optional here, default-OFF is mandatory.)
5. **Bridge to a DFlash2 trainer**: vLLM `speculators` + SGLang `SpecForge` both train
   DFlash2; SpecForge ships a `qwen3.8-27b-dflash2` config + recipe. Write a small converter
   from our `.npy` dump → the trainer's input schema. (MTP refit ≠ DFlash2 block-diffusion
   objective, so reuse an off-the-shelf DFlash2 trainer, not capy-flash's MTP `train.py`.)

### File-by-file (this fork)
- **ADD** `src/models/qwen3_5/execution/dflash_tfdump.{h,cpp}` — the ported dump writer
  (serializer + env parse + post-pass D2H), a `DFlashTfDump` sink.
- **MODIFY** `src/models/qwen3_5/execution/text.{h,cpp}` — wire `DFlashTfDump` as an
  additional consumer at the existing `DFlashFeatureSink`/`capture_layer` site; add the
  post-pass dump call after prefill (features + logits + target_tokens + positions in scope
  per `text.h:151`).
- **MODIFY** CLI/env plumbing (`apps/cli/options.cpp` or the env reader) — add the
  `NINFER_DFLASH_TFDUMP*` knobs, default OFF.
- **ADD** `tools/dflash2_refit/` (or a converter script) — dump → SpecForge/speculators input.

### Why this is cleaner than capy-flash's approach
- Our tap is **graph-compatible** (capture under the graph, dump in a post-pass) → no need to
  force `--no-cuda-graph` for capture; only the dump post-pass is a D2H.
- We reuse the existing dense `DFlashFeatureSink` (already wired for the 27B) instead of
  porting the MoE `Qwen4Taps`.
- The prefill already threads `logits`+`target_tokens`+`sink` together → single hook site.

## Constraints / correctness

- Taps must run in **eager prefill only** — CUDA-graph capture is incompatible with the D2H
  sync → the dump driver runs `--no-cuda-graph`.
- **Default OFF**; gated so it cannot reach a serving process by accident.
- Keep the dump out of the hot path when disabled (no overhead when off).

## Open questions

- ~~qwen4 vs qwen3_5 tap-site~~ **ANSWERED**: capy-flash's tap is MoE-only; we use our own
  dense `DFlashFeatureSink` (already wired for the 27B).
- Exact `DFlashFeatureSink.features` on-buffer layout (confirm concat order) → must be
  bit-identical to the drafter's serve-time feature.
- Dump-format → SpecForge/speculators input-schema converter (the one genuinely new piece).
- License/provenance for ported capy-flash dump code (check capy-flash's LICENSE; its MTP
  layer credits `ursuciprian/…` Apache-2.0 — the dump writer may be independently authored).
- Sample-rate / max-sequences gating for live-traffic (workload-matched) capture.
- Post-pass D2H scheduling when the flag is ON (async/batched so it doesn't stall serving).

## Acceptance criteria

- A dump flag/env produces, per sequence: 5-layer tap feature + target top-k logprobs + token
  ids + positions + seq id.
- Dumped features are **bit-identical** to the drafter's serve-time features (identity check).
- Default OFF; zero overhead when disabled.
- End-to-end: dump → DFlash2 train → artifact → bench, with a measured acceptance/tok-s delta.

## Effort (revised, 2026-10-25 — smaller than the original estimate)

- Port the model-agnostic dump writer (`dflash_tfdump.{h,cpp}`): ~0.5–1 d
- Wire the post-pass dump hook at the `DFlashFeatureSink` site (features+logits+tokens+pos):
  ~1–2 d
- Env/flag plumbing + default-OFF gate: ~0.5 d
- 5-layer layout confirmation + bit-identity check vs serve-time feature: ~0.5 d
- Bridge dump → DFlash2 trainer (converter): ~1–2 d
- **Total: ~3–6 d** for the engine-side capture + bridge (full E2E incl. a trained drafter +
  bench adds the trainer run).
