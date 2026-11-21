"""The Q8_G32_FP16 proposal head: encode (the quantizer side) and decode (the gather side).

`dspine_serial_scores` scores the 16 candidates of a proposal row against the tower's own output
head, which the drafter REUSES (`models/qwen3_5/definition.cpp:99` - the DSpine Weights'
`output_head` *is* `weights.text.output_head`). The prod tower head is
`q8_g32_fp16 [248320, 5120]`, so the op has to dequantize that representation on gather: a copy of
the head in the FP8 format P0b verified is 1.18 GiB of codes + 76 MiB of scales (2.5 GiB in BF16)
that the D-14 drafter sizing (-1.06 GiB net) does not contain.

This module is the reference side of that dequant:

  encode   `int8_fp16_scale` / `int8_quant_code` (ops/common/quant/integer.cuh:357-373):
           one FP16 scale per (row, 32-column group) = RNE(absmax / 127), codes = RNE(x / scale)
           clamped to [-127, 127], stored as signed bytes.
  decode   `embed_gather_q8_kernel` (ops/embedding/kernel.cuh:148-166), the in-tree gather of
           this same format, in the element order `serial_scores_kernel` uses:
               scale = fp16(scales[row * (padded / 32) + col / 32])
               code  = int8(codes[row * padded + col])
               value = bf16(fp32(code) * scale)
           `padded` is `align_up(k, 128)` (core/weight_view.cpp:168-190, RowSplit): the code plane
           is laid out at the PADDED row stride, so a reader that assumes the packed stride reads
           every row after the first from the wrong address. Prod's 5120 is a multiple of 128, so
           prod never shows that - the padded geometry in the driver is what covers it.

The four HEAD_MUTANTS are authoring errors of that decode, applied to the REFERENCE only: the
device reads the real bytes, so every one of them must land outside the band.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import numeric as num

GROUP = 32            # Q8RowSplitStorage::kGroupK
PADDING = 128         # weight_view.cpp RowSplit: padded_columns = align_up(k, 128)
CODE_RANGE = 127      # int8_quant_code clamps to [-127, 127]


def align_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def f16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    """The FP16 scale plane (`scale_dtype = FP16`) widened to the fp32 the kernel multiplies in."""
    return np.asarray(bits, dtype=np.uint16).view(np.float16).astype(np.float32)


@dataclass
class Q8Head:
    """A Q8_G32_FP16 weight: signed code bytes at the padded stride + FP16 group scales."""

    codes: np.ndarray     # uint8 (int8 storage) [rows, padded]
    scales: np.ndarray    # uint16 (fp16 bits)   [rows, padded / 32]
    k: int                # logical columns (the head's hidden)
    padded: int           # code row stride, == align_up(k, 128)

    @property
    def rows(self) -> int:
        return int(self.codes.shape[0])

    @property
    def groups(self) -> int:
        return self.padded // GROUP

    def bytes(self) -> dict:
        return {"codes": self.codes.size, "scales": self.scales.size * 2}


def encode_head(values: np.ndarray, pad: bool = True) -> Q8Head:
    """Quantize a [rows, k] head the way the int8 / FP16 group quantizer does."""
    rows, k = values.shape
    padded = align_up(k, PADDING) if pad else k
    if padded % GROUP:
        raise ValueError(f"k={k} needs a padded column count divisible by {GROUP}")
    columns = np.zeros((rows, padded), dtype=np.float32)
    columns[:, :k] = np.asarray(values, dtype=np.float32)
    groups = padded // GROUP
    blocks = columns.reshape(rows, groups, GROUP)
    absmax = np.max(np.abs(blocks), axis=2).astype(np.float32)
    scale16 = (absmax / np.float32(CODE_RANGE)).astype(np.float16)
    scale = scale16.astype(np.float32)
    live = (scale > 0)[:, :, None]
    safe = np.where(scale > 0, scale, np.float32(1.0))[:, :, None]
    quantized = np.where(live, blocks / safe, np.float32(0.0))
    codes = np.clip(np.rint(quantized), -CODE_RANGE, CODE_RANGE)
    return Q8Head(codes.reshape(rows, padded).astype(np.int8).astype(np.uint8),
                  scale16.view(np.uint16), k, padded)


def dequant_head(head: Q8Head, *, scale_group_offset: int = 0, ignore_sign: bool = False,
                 packed_stride: bool = False, scale_row_broadcast: bool = False) -> np.ndarray:
    """FP32 [rows, k] of the stored head, each element BF16-rounded exactly as the kernel rounds.

    The keyword flags are the teeth: a scale plane read one group off, a code plane read at the
    packed (unpadded) stride, a dropped sign, a row-scale broadcast in place of the group scales.
    The defaults are the faithful decode.
    """
    rows = head.rows
    stride = head.k if packed_stride else head.padded
    flat = head.codes.astype(np.int8).astype(np.float32).reshape(-1)
    address = (np.arange(rows, dtype=np.int64)[:, None] * stride
                + np.arange(head.k, dtype=np.int64)[None, :])
    codes = np.take(flat, address.ravel()).reshape(rows, head.k)
    if ignore_sign:
        codes = np.abs(codes)
    scales = f16_bits_to_f32(head.scales)
    if scale_group_offset:
        scales = np.roll(scales, scale_group_offset, axis=1)
    if scale_row_broadcast:
        scales = np.repeat(scales[:, :1], head.groups, axis=1)
    scale_grid = np.repeat(scales, GROUP, axis=1)[:, :head.k]
    return num.bf16_round(codes * scale_grid)


# Authoring errors of the head gather, applied to the REFERENCE decode. The device reads the real
# bytes, so each one has to move the verdict. `packed_stride_read` is inert unless the head is
# padded (padded > k), which the driver reports rather than hiding.
HEAD_MUTANTS = {
    "scale_group_offset": dict(scale_group_offset=1),
    "sign_ignored": dict(ignore_sign=True),
    "packed_stride_read": dict(packed_stride=True),
    "row_scale_broadcast": dict(scale_row_broadcast=True),
}


def profile(head: Q8Head) -> dict:
    values = dequant_head(head)
    return {"rows": head.rows, "k": head.k, "padded": head.padded,
            "code_bytes": head.bytes()["codes"], "scale_bytes": head.bytes()["scales"],
            "min": float(np.min(values)), "max": float(np.max(values))}
