"""NVFP4 gather-table encode/decode for the P2 parity harness.

CPU side of the NVFP4 table path. Nothing here invents a format:

* the **encoder** is the converter's own profile
  `tools/convert/quantization/nvfp4_maxabs.py` (`NVFP4_MAXABS_DIVISOR_RNE_V1`), and the
  scale plane is laid out by `tools/artifact/codecs/nvfp4.py::swizzle_nvfp4_scales`
  (`block_scale_k16_m128x4_v1`) - the same bytes the P1 nvfp4tables artifact carries;
* the **decoder** mirrors the device contract byte for byte:
  `ops/common/quant/nvfp4.cuh::decode_nvfp4_e2m1x2` (E2M1 nibbles, low nibble = the
  smaller column), `ops/common/quant/fp8.cuh::decode_e4m3` (reached through
  `tests/dspine_parity/reference.py::decode_e4m3`, the P0b copy) and
  `ops/dspine_bridge/kernels.cu::table_elem_bf16`:

      bf16( (e2m1(code) * decode_e4m3(scale_byte)) / weight_scale )

  with the scale byte addressed by `ops/common/quant/nvfp4.h::nvfp4_scale_byte_offset`
  (identical to `core/weight_view.cpp::weight_scale_offset`).

`dequant_bits` is therefore the "CPU reference dequantizing the same NVFP4 bytes" the
verdicts compare against: it is *the* decode contract, so a device/reference delta is a
delta in the kernel's gather path, not a delta in two dialects of the same format.
"""

from __future__ import annotations

from dataclasses import dataclass

import math
import sys
from pathlib import Path

import numpy as np

TREES = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TREES))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dspine_parity"))

from reference import decode_e4m3  # P0b harness copy of fp8.cuh:50 decode_e4m3
from numeric import bf16_view

from tools.artifact.codecs.nvfp4 import swizzle_nvfp4_scales
from tools.convert.quantization.nvfp4_maxabs import (  # noqa: E402
    ENCODER_PROFILE,
    quantize_nvfp4,
)

# E2M1 magnitudes + the sign bit, from ops/common/quant/nvfp4.h::Nvfp4Tables
# (`codes[]{0, .5, 1, 1.5, 2, 3, 4, 6}`, sign = code & 8).
E2M1 = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


@dataclass(frozen=True, slots=True)
class Nvfp4Table:
    """One gather table as the engine's Weight sees it."""

    packed: np.ndarray      # uint8 [n, k/2] code plane (qdata)
    scales: np.ndarray      # uint8 [n/128 * k/64 * 512] swizzled E4M3 plane
    divisor: float          # Weight::weight_scale (the FP32 matrix scalar)
    shape: tuple

    @property
    def code_bytes(self) -> int:
        return int(self.packed.size)

    @property
    def scale_bytes(self) -> int:
        return int(self.scales.size)


def encode_table(values: np.ndarray) -> Nvfp4Table:
    """Encode an [n, k] FP32/BF16-value matrix with the converter's NVFP4 profile."""
    import torch

    n, k = int(values.shape[0]), int(values.shape[1])
    if n % 128 or k % 64:
        raise ValueError(f"NVFP4 gather table needs n%128==0, k%64==0 (got {n}x{k})")
    words = quantize_nvfp4(torch.from_numpy(np.ascontiguousarray(values, np.float32)),
                           device="cpu")
    swizzled = swizzle_nvfp4_scales(words.natural_scales, (n, k))
    return Nvfp4Table(
        packed=np.ascontiguousarray(words.packed_codes.numpy(), np.uint8),
        scales=np.ascontiguousarray(swizzled.numpy(), np.uint8),
        divisor=float(np.float32(words.weight_scale)),
        shape=(n, k),
    )


def scale_offsets(n: int, k: int) -> np.ndarray:
    """nvfp4_scale_byte_offset(row, group, k) for every (row, group), int64 [n, k/16]."""
    rows = np.arange(n, dtype=np.int64)[:, None]
    groups = np.arange(k // 16, dtype=np.int64)[None, :]
    return ((((rows // 128) * (k // 64) + groups // 4) * 512)
            + (rows % 32) * 16 + ((rows % 128) // 32) * 4 + (groups % 4))


def natural_scale_offsets(n: int, k: int) -> np.ndarray:
    """The row-major [n, k/16] addressing - NOT what the layout registers. Used by the
    `scale_natural_offset` teeth mutant only."""
    rows = np.arange(n, dtype=np.int64)[:, None]
    groups = np.arange(k // 16, dtype=np.int64)[None, :]
    return rows * (k // 16) + groups


def dequant_bits(table: Nvfp4Table, *, swap_nibbles: bool = False,
                 natural_scales: bool = False, drop_divisor: bool = False,
                 ignore_sign: bool = False) -> np.ndarray:
    """BF16 bit patterns [n, k] of the registered NVFP4 dequant (kernels.cu table_elem_bf16).

    The keyword flags are the teeth mutants: each is a plausible authoring error in the
    gather path, applied to the REFERENCE decode only, so the device must disagree.
    """
    n, k = table.shape
    low = (table.packed & 0xF).astype(np.int64)
    high = (table.packed >> 4).astype(np.int64)
    if swap_nibbles:
        low, high = high, low
    codes = np.stack((low, high), axis=2).reshape(n, k)
    values = E2M1[codes]
    if ignore_sign:
        values = np.abs(values).astype(np.float32)

    offsets = (natural_scale_offsets(n, k) if natural_scales
               else scale_offsets(n, k))
    scale_bytes = np.zeros(int(offsets.max()) + 1, dtype=np.uint8)
    used = min(table.scales.size, scale_bytes.size)
    scale_bytes[:used] = table.scales[:used]
    scale_words = scale_bytes[offsets]
    scales = np.array([decode_e4m3(int(word)) for word in scale_words.ravel()],
                      dtype=np.float32).reshape(scale_words.shape)

    scaled = values.astype(np.float32) * np.repeat(scales, 16, axis=1).astype(np.float32)
    out = scaled if drop_divisor else (scaled / np.float32(table.divisor)).astype(np.float32)
    return bf16_view(out)


def dequant_f32(bits: np.ndarray) -> np.ndarray:
    from numeric import bf16_bits_to_f32

    return bf16_bits_to_f32(bits)


def profile() -> dict:
    return {
        "encoder": ENCODER_PROFILE,
        "layout": "block_scale_k16_m128x4_v1",
        "e2m1_values": [float(value) for value in E2M1],
        "fp8_e4m3_probe": [decode_e4m3(word) for word in (0x00, 0x01, 0x38, 0x40, 0x7E)],
        "sqrt2": math.sqrt(2.0),
    }
