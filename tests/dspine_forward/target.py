"""Materialise the two target tensors the DSpine draft forward borrows.

The draft owns no head and no embedding — it borrows ``text/output_head`` and
``text/token_embedding`` (``src/models/qwen3_5/definition.cpp`` assigns them onto the
bound DSpine weights), so the host forward must materialise the *target's* objects.
The head is the one linear the forward cannot run on BF16 alone: the V3 tower ships it
as ``q8_g32_fp16 / row_split_k128_v1`` (1.26 GiB at V=248,320 x H=5,120) and the
DFlash2 towers as ``fp8_e4m3fn_row_bf16 / row_scale_v1``; the host twin dequantizes
either to FP32.

The row-split decoder is NumPy, a byte-for-byte companion of the registered layout
``tools.artifact.layouts.row_split_geometry`` +
``tools.artifact.codecs.row_split._unpack_codes`` (torch-free on purpose: the CMake
discovered interpreter has no torch, which is exactly why the P1 suite's 8 python-dep
tests skip).  ``--self-check`` proves the two decoders agree on a real artifact.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np

SCRIPTS = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS))

from tools.artifact.layouts import row_split_geometry  # noqa: E402

ROW_SPLIT = "row_split_k128_v1"
ROW_SCALE = "row_scale_v1"
CONTIGUOUS = "contiguous_le_v1"
from tools.artifact.layouts import PLANE_ALIGNMENT  # noqa: E402

K_ALIGNMENT = 128


def _read_at(path: str, begin: int, length: int) -> bytes:
    with open(path, "rb") as handle:
        handle.seek(begin)
        data = handle.read(length)
    if len(data) != length:
        raise SystemExit(f"{path}: short read at {begin}")
    return data


def _bf16(payload: bytes, shape: tuple[int, ...]) -> np.ndarray:
    bits = np.frombuffer(bytearray(payload), dtype=np.uint16)
    if bits.size != int(np.prod(shape)):
        raise SystemExit(f"bf16 payload has {bits.size} elements, expected {shape}")
    return (bits.astype(np.uint32) << 16).view(np.float32).reshape(shape)


def _bits_of(fmt: str) -> int:
    for bits in (8, 6, 5, 4):
        if fmt.startswith(f"q{bits}"):
            return bits
    raise SystemExit(f"{fmt}: not a qN grouped format")


def dequantize_row_split(payload: bytes, fmt: str, shape: tuple[int, ...],
                         group_size: int) -> np.ndarray:
    """``row_split_k128_v1`` -> FP32 [N, K]: ``code * fp16_group_scale``."""

    spec_bits = _bits_of(fmt)
    geometry = row_split_geometry(fmt, (*shape,))
    if len(payload) != geometry.payload_bytes:
        raise SystemExit(
            f"row-split payload has {len(payload)} bytes, "
            f"expected {geometry.payload_bytes}"
        )
    raw = np.frombuffer(bytearray(payload), dtype=np.uint8)
    base = raw[geometry.base_offset : geometry.base_offset + geometry.base_bytes]
    scale = raw[geometry.scale_offset : geometry.scale_offset + geometry.scale_bytes]
    scales = (
        scale.view(np.float16).reshape(geometry.n, geometry.groups_per_row).astype(np.float32)
    )
    groups = geometry.n * geometry.groups_per_row
    if spec_bits == 8:
        codes = base.view(np.int8).reshape(
            geometry.n, geometry.groups_per_row, group_size
        )
    else:
        packed = base.reshape(groups, geometry.base_bytes_per_group).astype(np.int32)
        low = np.empty((groups, group_size), dtype=np.int32)
        low[:, 0::2] = packed & 0x0F
        low[:, 1::2] = (packed >> 4) & 0x0F
        if spec_bits == 4:
            unsigned = low
        else:
            high_plane = raw[
                geometry.high_offset : geometry.high_offset + geometry.high_bytes
            ].reshape(groups, geometry.high_bytes_per_group).astype(np.int32)
            byte_index = np.arange(group_size) * (spec_bits - 4) // 8
            shift = (np.arange(group_size) * (spec_bits - 4)) % 8
            high = (
                high_plane[:, byte_index] >> shift[None, :]
            ) & ((1 << (spec_bits - 4)) - 1)
            unsigned = low | (high << 4)
        sign = 1 << (spec_bits - 1)
        codes = np.where((unsigned & sign) != 0, unsigned - (1 << spec_bits), unsigned)
        codes = codes.astype(np.int8).reshape(
            geometry.n, geometry.groups_per_row, group_size
        )
    values = codes.astype(np.float32) * scales[:, :, None]
    return values.reshape(geometry.n, geometry.k_pad)[:, : geometry.k].astype(np.float32)


def dequantize_row_scaled(payload: bytes, shape: tuple[int, ...]) -> np.ndarray:
    """``fp8_e4m3fn_row_bf16 / row_scale_v1``: E4M3FN codes x a BF16 row scale."""

    rows, columns = shape
    codes = np.frombuffer(bytearray(payload), dtype=np.uint8).reshape(rows, columns)
    exponent = ((codes.astype(np.uint32) >> 3) & 0xF).astype(np.int32)
    mantissa = (codes & 0x7).astype(np.float32)
    sign = np.where(((codes.astype(np.uint32) >> 7) & 1) != 0, -1.0, 1.0)
    normal = (1.0 + mantissa / 8.0) * np.exp2((exponent - 7).astype(np.float64))
    subnormal = (mantissa / 8.0) * (2.0**-6)
    decoded = (sign * np.where(exponent == 0, subnormal, normal)).astype(np.float32)
    scale_offset = (rows * columns + 255) // 256 * 256
    scales = _bf16(payload[scale_offset : scale_offset + 2 * rows], (rows,))
    return decoded * scales.astype(np.float32)[:, None]


def _head_entry(path: str, binding: str) -> tuple[dict, dict, int]:
    from tools.dspine.artifact import read_directory

    directory, base = read_directory(path)
    objects = {item["id"]: item for item in directory["objects"]}
    name = directory["bindings"][binding]["object"]
    item = objects[name]
    return directory, item, base


def head_from_artifact(path: str, binding: str = "text/output_head",
                       *, quantize: bool = True) -> dict:
    """Dequantize ``text/output_head`` to FP32 [V, H] (+ its A8 group geometry)."""

    _, item, base = _head_entry(path, binding)
    payload = _read_at(path, base + item["offset"], item["bytes"])
    shape = tuple(int(v) for v in item["shape"])
    fmt, layout = item["format"], item["layout"]
    group = _group_of(fmt)
    if layout == ROW_SPLIT:
        dequant = dequantize_row_split(payload, fmt, shape, group)
    elif layout == ROW_SCALE:
        dequant = dequantize_row_scaled(payload, shape)
    elif fmt == "bf16" and layout == CONTIGUOUS:
        dequant = _bf16(payload, shape)
    else:
        raise SystemExit(f"{binding}: unsupported head {fmt}/{layout}")
    out = {
        "dequant": dequant,
        "format": fmt,
        "layout": layout,
        "shape": shape,
        "bytes": item["bytes"],
        "group_size": group,
    }
    if quantize:
        from tests.dspine_forward import numeric

        out["codes"] = numeric.quantize_rows(dequant, bits=8, groups=group)
        out["scales"] = _group_scales(dequant, group)
    return out


def embedding_from_artifact(path: str, binding: str = "text/token_embedding") -> dict:
    """Dequantize ``text/token_embedding`` to FP32 [V, H] (the draft's input embed)."""

    _, item, base = _head_entry(path, binding)
    payload = _read_at(path, base + item["offset"], item["bytes"])
    shape = tuple(int(v) for v in item["shape"])
    fmt, layout = item["format"], item["layout"]
    if fmt == "bf16" and layout == CONTIGUOUS:
        values = _bf16(payload, shape)
    elif layout == ROW_SPLIT:
        values = dequantize_row_split(payload, fmt, shape, _group_of(fmt))
    elif fmt == "nvfp4":
        from tools.dspine.artifact import dequant_nvfp4

        values = dequant_nvfp4(payload, shape)
    else:
        raise SystemExit(f"{binding}: unsupported embedding {fmt}/{layout}")
    return {"embed": values, "shape": shape, "format": fmt, "layout": layout,
            "bytes": item["bytes"]}


def _group_of(fmt: str) -> int:
    for size in (128, 64, 32, 16):
        if f"_g{size}_" in f"_{fmt}_":
            return size
    raise SystemExit(f"{fmt}: no group size in the format name")


def _group_scales(values: np.ndarray, group: int) -> np.ndarray:
    """The FP16 per-group scales of the A8 geometry, in the power-of-two form the
    ``q8_g32_fp16`` contract stores (``numeric.quantize_rows`` uses the same rule)."""

    source = np.ascontiguousarray(values, dtype=np.float32)
    rows, columns = source.shape
    grouped = source.reshape(rows, columns // group, group)
    amax = np.max(np.abs(grouped), axis=-1)
    high = np.float32((1 << 7) - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(amax > 0, amax / high, np.float32(1.0))
    return np.exp2(np.floor(np.log2(scale))).astype(np.float32)
