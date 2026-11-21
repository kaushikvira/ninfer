#!/usr/bin/env python3
"""P2 stage 3b: the DSpine forward walk's device outputs against the torch/numpy reference.

The device probe (tests/models/qwen3_5/test_dspine_forward_real.cpp, NINFER_TEST_STAGE=forward,
NINFER_TEST_DUMP=<path>) writes the LAST code bridge's chain - its inputs and the bytes the
device published - and the proposal round's score table, candidate rows, serial_scores idx and
the tail's drafts. This script recomputes, on CPU, from the SAME bytes:

  prepare_read      -> normalized, rms
  normalize_codes   -> codes            (the non-anchor columns; see the note below)
  message           -> message
  finish            -> residual_out, normed
  serial_chain      -> drafts          (the T=0 arm of dspine_serial_tail)

The formulas are the ones tests/dspine_parity/reference.py carries: they are transcribed from
Tencent's reference (sglang_dspine/models/dspine.py + sglang_dspine/kernels/dspine.py, whose
per-kernel rounding points reference.py cites by line), not hand-written for this comparison, so
this is the composed-chain comparison of a row that P0b measured op by op.

`serial_scores` is NOT recomputed here: it indexes the codebook, tg, tsv (NVFP4 in the sized
artifact) and the tower head (q8_g32_fp16), and dumping those tables would feed the comparison
the device's own dequantization - the vacuity the reference exists to exclude. The gather parity
of those tables is P0b's (normalize_codes 0 ulp, serial_scores idx 240/240) and D-37's (the Q8
head's idx 240/240 bit-exact, argmax(table) == device idx bit-exact), and is reported as
inherited, not re-measured. The T=0 drafts are compared through serial_chain, which is integer
only: that row is a full reference comparison of the 15 drafts.

normalize_codes' ANCHOR columns (block column 0 of each batch row) are the codebook rows, so they
are the same exclusion; the non-anchor columns are the rms path and are compared.

Usage:  ./home/kv/venvs/qquant/bin/python run_forward_reference.py <dump>
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "dspine_parity"))

import reference  # noqa: E402  (tests/dspine_parity/reference.py)
from numeric import bf16_bits_to_f32, bf16_round  # noqa: E402

DTYPES = {"f4": "<f4", "i4": "<i4", "u2": "<u2", "f8": "<f8"}
WIDTHS = {"f4": 4, "i4": 4, "u2": 2, "f8": 8}


def load(path: str) -> dict:
    records: dict = {}
    with open(path, "rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            name, dtype, count = line.decode().split()
            count = int(count)
            width = WIDTHS[dtype]
            payload = handle.read(count * width)
            if len(payload) != count * width:
                raise EOFError(f"{name}: short record")
            records[name] = np.frombuffer(payload, dtype=DTYPES[dtype]).copy()
    return records


def bf16(bits: np.ndarray) -> np.ndarray:
    return bf16_bits_to_f32(bits.astype(np.uint16))


def compare(name: str, device: np.ndarray, expected: np.ndarray) -> tuple:
    """The delta is read two ways: the bf16 bit pattern (the parity rows' criterion) and the
    FP32 value spread (what a non-bit-exact row is worth)."""
    a = device.astype(np.uint16) if device.dtype == np.uint16 else device
    b = expected.astype(np.uint16) if expected.dtype == np.uint16 else expected
    if a.shape != b.shape:
        a = a.reshape(b.shape)
    if a.dtype == np.uint16 and b.dtype == np.uint16:
        differing = int(np.count_nonzero(a != b))
        fa, fb = bf16(a), bf16(b)
    else:
        differing = int(np.count_nonzero(a != b))
        fa, fb = a.astype(np.float64), b.astype(np.float64)
    delta = float(np.max(np.abs(fa - fb))) if fa.size else 0.0
    scale = float(np.max(np.abs(fb))) or 1.0
    verdict = "BIT-EXACT" if differing == 0 else "delta"
    print(f"  {name:<26} {differing:>8}/{a.size} differing   "
          f"max |d| {delta:.3e}   rel-Fro {rel_fro(fa, fb):.3e}   {verdict}")
    return differing, delta


def rel_fro(a: np.ndarray, b: np.ndarray) -> float:
    difference = float(np.linalg.norm((a - b).astype(np.float64)))
    base = float(np.linalg.norm(b.astype(np.float64)))
    return difference / base if base > 0 else difference


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    r = load(sys.argv[1])
    (hidden, width, batch, columns, code_size, rank, taps, head_dim,
     query_heads, kv_heads) = [int(v) for v in r["shape"]]
    bridge_eps = float(r["bridge_eps"][0])
    rms_norm_eps = float(r["rms_norm_eps"][0])
    print(f"dump: hidden={hidden} block={width} batch={batch} code={code_size} rank={rank} "
          f"taps={taps} head_dim={head_dim} heads={query_heads}/{kv_heads}")
    print(f"      bridge eps={bridge_eps:g} rms_norm eps={rms_norm_eps:g}")
    print("the composed bridge chain and the T=0 drafts, device vs the reference:")

    failures = 0
    read = reference.prepare_read(r["residual_pre"].astype(np.uint16), hidden, bridge_eps)
    failures += compare("prepare_read.normalized", r["normalized"],
                        read.outputs["normalized"]) [0]
    failures += compare("prepare_read.rms", r["rms"], read.outputs["rms"])[0]

    anchors = r["anchors"].astype(np.int64)
    # The codebook is NVFP4 in the sized artifact, so its rows are the device's own
    # dequantization and are not dumped: the anchor columns are excluded below. A
    # zero-filled stand-in lets the reference compute the non-anchor columns.
    codes = reference.normalize_codes(r["codes_pre"].astype(np.uint16), anchors,
                                      np.zeros((int(anchors.max()) + 1, code_size),
                                               dtype=np.uint16),
                                      code_size, width, bridge_eps)
    anchor_columns = [column for column in range(columns) if column % width == 0]
    body = [column for column in range(columns) if column % width != 0]
    device_codes = r["codes"].astype(np.uint16).reshape((code_size, columns))
    expect_codes = codes.outputs["codes"].reshape((code_size, columns))
    failures += compare("normalize_codes (non-anchor)",
                        device_codes[:, body].reshape(-1),
                        expect_codes[:, body].reshape(-1))[0]

    message = reference.message(r["packed"].astype(np.uint16),
                                r["gate_bias"].astype(np.uint16), rank, width, batch)
    failures += compare("message", r["message"], message.outputs["message"])[0]

    finish = reference.finish(r["residual_pre"].astype(np.uint16), r["rms"],
                             r["delta"].astype(np.uint16),
                             r["final_norm"].astype(np.uint16), hidden, width, rms_norm_eps)
    failures += compare("finish.residual_out", r["residual_post"],
                        finish.outputs["residual_out"])[0]
    failures += compare("finish.normed", r["normed"], finish.outputs["normed"])[0]

    cand = r["tail_cand"].astype(np.int32).reshape((batch, width - 1, 16))
    idx = r["idx"].astype(np.int32).reshape((16, width - 1, batch))
    # The tail's cand is [batch, slot, candidate]; serial_chain reads (candidate, slot, batch).
    chain = reference.serial_chain(
        np.ascontiguousarray(np.swapaxes(cand, 0, 2), dtype=np.int32), idx)
    reference_drafts = chain.outputs["drafts"].reshape(-1)
    device_greedy = r["drafts_greedy"].astype(np.int32).reshape(-1)
    identical = bool(np.array_equal(reference_drafts, device_greedy))
    print(f"  {'serial_chain drafts (T=0)':<26} reference = {list(map(int, reference_drafts))}")
    print(f"  {'':<26} device   = {list(map(int, device_greedy))}")
    print(f"  {'':<26} drafts identical: {'Y' if identical else 'N'}")
    failures += 0 if identical else 1
    print(f"  {'serial_tail drafts (T=0.9)':<26} device = "
          f"{list(map(int, r['drafts_sampled'].astype(np.int32).reshape(-1)))}  "
          f"(the sampled draw is not a reference row: the draw stream is the engine's)")

    print(f"reference comparison: {'GREEN' if failures == 0 else f'{failures} RED ROWS'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
