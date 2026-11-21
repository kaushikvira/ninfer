"""bf16 numeric helpers for the DSpine host forward (P2 stage 1).

A local, torch-free copy of the P0b helpers (``tests/dspine_parity/numeric.py``,
merged ``4f... f3846a56``) — deliberately duplicated rather than imported:
``run_parity.py`` sits in the P0b tree with a ``sys.path.insert``, and the CMake
discovered interpreter (the one the ``real`` suite runs under) has no torch, so an
import of the P0b module would make this test one of the P1 suite's 8
python-dependency skips.  The formulas are identical bit for bit.
"""

from __future__ import annotations

import numpy as np


def f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even fp32 -> bf16 bit patterns (__float2bfloat16_rn)."""

    f32 = np.ascontiguousarray(values, dtype=np.float32)
    bits = f32.view(np.uint32).astype(np.uint64)
    lsb = (bits >> 16) & 1
    rounded = ((bits + 0x7FFF + lsb) >> 16).astype(np.uint16)
    special = np.isnan(f32) | np.isinf(f32)
    if np.any(special):
        rounded = np.where(special, ((bits >> 16) | 0x0040).astype(np.uint16), rounded)
    return rounded.astype(np.uint16)


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint16)
    return (bits.astype(np.uint32) << 16).view(np.float32)


def bf16_round(values: np.ndarray) -> np.ndarray:
    """fp32 -> bf16 -> fp32: the value a bf16 store keeps."""

    return bf16_bits_to_f32(f32_to_bf16_bits(values))


def bf16_view(f32: np.ndarray) -> np.ndarray:
    """Bit patterns of an fp32 array as bf16 storage."""

    return f32_to_bf16_bits(f32)


def f32_bits(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)


def quantize_rows(values: np.ndarray, bits: int = 8, groups: int = 32) -> np.ndarray:
    """Group-wise symmetric row quantization, the ``q8_g32_fp16`` family.

    ``row_scale`` is a power of two so the dequantized code
    ``code * scale`` is exactly representable in binary32 — the property that makes
    ``A16Only`` linear routes admissible for it (the same reason DFlash2's
    ``fp8_e4m3fn_row_bf16`` head is admissible).
    """

    source = np.ascontiguousarray(values, dtype=np.float32)
    rows, columns = source.shape
    if columns % groups:
        raise ValueError(f"columns {columns} is not a multiple of the group size {groups}")
    codes = np.zeros((rows, columns), dtype=np.int8)
    for begin in range(0, rows, 4096):
        end = min(begin + 4096, rows)
        block = source[begin:end]
        high = (1 << (bits - 1)) - 1
        low = -(1 << (bits - 1))
        grouped = block.reshape(end - begin, columns // groups, groups)
        amax = np.max(np.abs(grouped), axis=-1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.where(amax > 0, amax / high, np.float32(1.0))
        scale = np.exp2(np.floor(np.log2(scale))).astype(np.float32)
        quantized = np.rint(grouped / scale)
        np.clip(quantized, low, high, out=quantized)
        codes[begin:end] = quantized.reshape(end - begin, columns).astype(np.int8)
    return codes


def dyadic(shape: tuple[int, ...], seed: int, exp_lo: int = -7, exp_hi: int = 3,
           mant_levels: int = 9, density: float = 0.75) -> np.ndarray:
    """Exact-in-bf16 values: sign * (m/8) * 2**e, m in [0, mant_levels-1].

    Products of two dyadic draws stay exact in fp32 (and, in this exponent band, in
    bf16), so a sum over them is order-independent: a reduction-order difference
    cannot masquerade as a logic defect and any real defect shows as a non-zero ulp.
    (P0b ``numeric.dyadic`` — same construction, same purpose.)
    """

    rng = np.random.default_rng(seed)
    exponent = rng.integers(exp_lo, exp_hi + 1, size=shape)
    mantissa = rng.integers(0, mant_levels, size=shape)
    sign = np.where(rng.random(shape) < 0.5, -1.0, 1.0)
    values = sign * (mantissa.astype(np.float32) / 8.0) * np.ldexp(
        np.ones(shape, np.float32), exponent
    )
    mask = rng.random(shape) < density
    return np.where(mask, values, np.float32(0.0)).astype(np.float32)
