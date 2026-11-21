# DSpine forward-walk reference comparison (P2 stage 3b)

`run_forward_reference.py` compares the **composed** proposal walk that
`tests/models/qwen3_5/test_dspine_forward_real.cpp` (`NINFER_TEST_STAGE=forward`) publishes
against the reference math `tests/dspine_parity/reference.py` carries (transcribed from Tencent's
`sglang_dspine/models/dspine.py` + `sglang_dspine/kernels/dspine.py`).

Run:

```
# 1. walk it on the free card and dump the chain
CUDA_VISIBLE_DEVICES=1 NINFER_TEST_DEVICE=0 NINFER_TEST_STAGE=forward \
NINFER_TEST_ARTIFACT=/home/kv/models/dspine-v3/qwen3_8_27b_swift15_v3_dspine-nvfp4tables.ninfer \
NINFER_TEST_DUMP=/tmp/walk-dump.bin \
  build/rtx5090-dev/tests/ninfer_qwen3_5_dspine_forward_real_test

# 2. recompute the chain on CPU from the dumped bytes
/home/kv/venvs/qquant/bin/python run_forward_reference.py /tmp/walk-dump.bin
```

The script recomputes `prepare_read`, `normalize_codes` (non-anchor columns), `message`, `finish`
and the `serial_chain` walk of the 15 T=0 drafts.

## What it deliberately does not cover

* `serial_scores` — it indexes the codebook, `tg`, `tsv` (NVFP4 in the sized artifact) and the
  tower head (`q8_g32_fp16`). Dumping those tables feeds the comparison the device's own
  dequantization, which is the vacuity a reference exists to exclude. Their gather parity is
  P0b's (`normalize_codes` 0 ulp, `serial_scores` idx 240/240 bit-exact) and D-37's (the Q8 head's
  `idx` 240/240 bit-exact; `argmax(table)` == device `idx` bit-exact) — **inherited, not
  re-measured here**.
* `normalize_codes`' anchor columns — the codebook rows, the same exclusion.
* The T=0.9 draw — the counter-based draw stream is the engine's, not the reference's.

## Status of the row

The comparison runs and prints, and on the synthetic tap field the walk currently carries it is
**not a parity row**: a 5-layer walk over synthetic activations and real trained weights
saturates (the read-RMS reaches ~2.9e4 and the refined residual ~2.3e6), and in that regime the
per-op BF16 rounding points diverge by ~1.4 relative-Frobenius. The bit-exact parity rows are
the P0b per-op rows on well-scaled fixtures. Closing this row as a *parity* row needs a
real-prompt activation fixture (a captured prefill), which is the follow-up.
