"""bf16 numeric helpers for the P0b parity harness.

Every comparison is at the storage precision the ops use (BF16). "ulps" here means
representable BF16 steps between the two stored values (0 == bit-exact, 1 == one
rounding step at the reference's own fp32 rounding scale).
"""

from __future__ import annotations

import numpy as np

BF16 = np.dtype("uint16")


def f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even fp32 -> bf16 bit patterns (matches __float2bfloat16_rn)."""
    f32 = np.ascontiguousarray(values, dtype=np.float32)
    bits = f32.view(np.uint32).astype(np.uint64)
    lsb = (bits >> 16) & 1
    rounded = ((bits + 0x7FFF + lsb) >> 16).astype(np.uint16)
    special = np.isnan(f32) | np.isinf(f32)
    if np.any(special):
        # Keep infinities/NaNs as bf16 infinities/NaNs instead of overflowing the exponent.
        rounded = np.where(special, ((bits >> 16) | 0x0040).astype(np.uint16), rounded)
    return rounded.astype(np.uint16)


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint16)
    widened = (bits.astype(np.uint32) << 16).view(np.float32)
    return widened


def bf16_round(values: np.ndarray) -> np.ndarray:
    """fp32 -> bf16 -> fp32: the exact value a bf16 store would have kept."""
    return bf16_bits_to_f32(f32_to_bf16_bits(values))


def bf16_view(f32: np.ndarray) -> np.ndarray:
    """Bit patterns to write to the fixture as raw bf16 storage."""
    return f32_to_bf16_bits(f32)


def ulp_distance(bits_a: np.ndarray, bits_b: np.ndarray) -> np.ndarray:
    """Signed ulp distance (uint16 bf16 or uint32 fp32) via the two's-complement
    -> order-preserving key map (flip all bits for negatives, else flip the sign bit)."""
    a = np.ascontiguousarray(bits_a)
    b = np.ascontiguousarray(bits_b)
    if a.dtype not in (np.uint16, np.uint32) or a.dtype != b.dtype:
        raise ValueError(f"ulp_distance needs matching uint16/uint32, got {a.dtype} {b.dtype}")

    def key(x: np.ndarray) -> np.ndarray:
        sign = np.left_shift(np.ones(x.shape, dtype=x.dtype), x.itemsize * 8 - 1)
        allbits = ~np.zeros(x.shape, dtype=x.dtype)
        return np.where(x & sign, x ^ allbits, x ^ sign)

    return key(a).astype(np.int64) - key(b).astype(np.int64)


def summary(bits_a: np.ndarray, bits_b: np.ndarray) -> dict:
    delta = ulp_distance(bits_a, bits_b)
    return {
        "n": int(delta.size),
        "exact": int(np.count_nonzero(delta == 0)),
        "mismatch": int(np.count_nonzero(delta != 0)),
        "max_abs_ulp": int(np.max(np.abs(delta))) if delta.size else 0,
        "p99_abs_ulp": int(np.percentile(np.abs(delta), 99)) if delta.size else 0,
        "worst_index": int(np.argmax(np.abs(delta))) if delta.size else -1,
    }


def dyadic(shape: tuple[int, ...], seed: int, exp_lo: int = -7, exp_hi: int = 3,
            mant_levels: int = 9, density: float = 0.75) -> np.ndarray:
    """Exact-in-bf16 values: sign * (m/8) * 2**e with m in [0, mant_levels-1].

    Products of two dyadic draws stay exact in fp32 and (within the exponent band) in
    bf16, so a sum over them is order-independent: a reduction-order difference cannot
    masquerade as a logic defect, and any real defect shows up as a non-zero ulp.
    """
    rng = np.random.default_rng(seed)
    exponent = rng.integers(exp_lo, exp_hi + 1, size=shape)
    mantissa = rng.integers(0, mant_levels, size=shape)
    sign = np.where(rng.random(shape) < 0.5, -1.0, 1.0)
    values = sign * (mantissa.astype(np.float32) / 8.0) * np.ldexp(np.ones(shape, np.float32),
                                                                  exponent)
    # np.where (not values * mask) so masked-out entries are +0.0, not signed zeros -
    # a signed-zero input is a data artifact that shows up as a 1-ulp sign-bit difference
    # in the output and is not a rounding defect.
    mask = rng.random(shape) < density
    return np.where(mask, values, np.float32(0.0)).astype(np.float32)


def fp32_exact(shape: tuple[int, ...], seed: int, exp_lo: int = -3,
               exp_hi: int = 2) -> np.ndarray:
    """fp32 values with a 2-bit mantissa: products with bf16 stay exact in fp32."""
    rng = np.random.default_rng(seed)
    exponent = rng.integers(exp_lo, exp_hi + 1, size=shape)
    mantissa = rng.integers(4, 8, size=shape)  # 1.00, 1.25, 1.5, 1.75 -> 2 mantissa bits
    sign = np.where(rng.random(shape) < 0.5, -1.0, 1.0)
    return (sign * (mantissa.astype(np.float32) / 4.0)
            * np.ldexp(np.ones(shape, np.float32), exponent)).astype(np.float32)
