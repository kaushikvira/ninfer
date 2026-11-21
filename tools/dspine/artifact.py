#!/usr/bin/env python3
"""Load the DSpine draft region of a NInfer v3 artifact on the host.

    python3 -m tools.dspine.artifact ARTIFACT.ninfer [--draft-config DIR]
                                                   [--json] [--verify-bytes]

This is the **host twin of the engine's load side**
(``src/models/common/dspine/{config,weights}.*``): it reads the ``dspine``
component's declared config, walks the same object set the engine binds, checks
every declared shape/format/layout, classifies each object into the D-20 regions
(backbone / code bridge / norms / serial tables), and reports the draft footprint.

It also materialises the tensors the draft forward needs, which is where the
**NVFP4 table dequant path** lives (the path P0b recorded as UNMEASURED and D-20
made load-bearing for the 1.3157 GiB sizing): ``dequant_nvfp4`` decodes the
registered NVFP4 contract -- E2M1 codes x E4M3FN K16 block scales / a per-matrix
FP32 divisor -- so the forward can run on the *real* artifact bytes.

``--reference-config`` compares the artifact's recorded draft config against the
release ``config.json`` it was built from (the D-20 contract check: block_size 16,
code_size 1024, rank 512, taps [1,16,31,46,61], ...).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

import numpy as np

SCRIPTS = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS))

from tools.artifact.framing import HEADER, MAGIC, PAYLOAD_ALIGNMENT  # noqa: E402
from tools.artifact.layouts import block_scale_geometry  # noqa: E402

PREFIX = "dspine"
ALIGN = PAYLOAD_ALIGNMENT

REGIONS = ("backbone", "code_bridge", "norms", "serial_tables", "codebook", "other")

# D-20 / D-13: the contract the V3-tailored artifacts are built to.
CONTRACT = {
    "block_size": 16,
    "code_size": 1024,
    "rank": 512,
    "eps": 1e-6,
    "kappa": 10.0,
    "decode_mode": "serial",
    "target_layer_ids": [1, 16, 31, 46, 61],
    "mask_token_id": 248070,
    "first_exact": 4,
    "head_topk": 16,
    "num_target_layers": 64,
    "num_hidden_layers": 5,
}

BF16_CODE = "bf16"
NVFP4_CODE = "nvfp4"


# --------------------------------------------------------------------------- io


def _align(value: int, amount: int = ALIGN) -> int:
    return (value + amount - 1) // amount * amount


def read_directory(path: str) -> tuple[dict, int]:
    """Return the artifact directory and the payload base offset."""

    with open(path, "rb") as handle:
        magic, json_bytes, _ = HEADER.unpack(handle.read(HEADER.size))
        if magic != MAGIC:
            raise SystemExit(f"{path}: not a NInfer v3 artifact (magic {magic!r})")
        raw = handle.read(json_bytes)
    directory = json.loads(raw)
    return directory, HEADER.size + json_bytes


def _read_at(path: str, begin: int, length: int) -> bytes:
    with open(path, "rb") as handle:
        handle.seek(begin)
        data = handle.read(length)
    if len(data) != length:
        raise SystemExit(f"{path}: short read at {begin} ({len(data)} != {length})")
    return data


# ------------------------------------------------------------------ nvfp4 decode

_E2M1 = np.array(
    [float(v) for v in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)], dtype=np.float64
)


def _e2m1_table() -> np.ndarray:
    magnitudes = []
    for code in range(16):
        magnitude = float(_E2M1[code & 7])
        magnitudes.append(-magnitude if code & 8 else magnitude)
    return np.array(magnitudes, dtype=np.float64)


def _e4m3fn_table() -> np.ndarray:
    """E4M3FN: 4-bit exponent (bias 7), 3-bit mantissa, no sign bit (values >= 0).

    Same decode the engine's ``common::validate_nvfp4_weight`` path and the
    ``tools.artifact.codecs.nvfp4`` contract assume; scales with the sign bit set
    or the 0x7F (NaN) word are rejected by the codec, so this table is total.
    """
    values = np.zeros(256, dtype=np.float64)
    for code in range(256):
        exponent = (code >> 3) & 0xF
        mantissa = code & 0x7
        if exponent == 0:
            values[code] = (mantissa / 8.0) * (2.0**-6)
        else:
            values[code] = (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))
    return values


def unswizzle_nvfp4_scales(stored: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Registered swizzled K16 scale plane -> natural ``[…, K/16]`` E4M3FN words.

    The byte-for-byte inverse of ``tools.artifact.codecs.nvfp4.swizzle_nvfp4_scales``
    (``reshape(m*n/128, 4, 32, k/64, 4).permute(0,3,2,1,4)``), expressed on the
    inverse permutation.  NumPy-only on purpose: the torch-free host twin is what
    lets the forward tests run in the CMake-discovered interpreter (which has no
    torch — the P1 suite's 8 python-dep failures are exactly that gap).
    """

    geometry = block_scale_geometry("nvfp4", shape)
    expect = geometry.scale_plane_bytes
    if stored.size != expect:
        raise SystemExit(
            f"NVFP4 scale plane has {stored.size} bytes, expected {expect} for {shape}"
        )
    # The byte-for-byte inverse of the encoder's swizzle: the stored stream is
    # ``reshape(matrices * n/128, 4, 32, k_tiles, 4).permute(0, 3, 2, 1, 4)``, so the
    # stored order reads (matrix-tile, k_tile, 32, 4, 4) and the natural order is
    # (matrix-tile, 4, 32, k_tile, 4) = (n, k/16) once flattened.
    if geometry.matrices == 1:
        # ``[n, k/16]`` from ``(n/128, k_tiles, 32, 4, 4)`` -> ``(n/128, 4, 32, k, 4)``.
        return (
            stored.reshape(geometry.n // 128, geometry.k_tiles, 32, 4, 4)
            .transpose(0, 3, 2, 1, 4)
            .reshape(geometry.n, geometry.groups_per_row)
            .copy()
        )
    return (
        stored.reshape(
            geometry.matrices,
            geometry.n // 128,
            geometry.k_tiles,
            32,
            4,
            4,
        )
        .transpose(0, 2, 1, 4, 3, 5)
        .reshape(geometry.matrices, geometry.n, geometry.groups_per_row)
        .copy()
    )


def dequant_nvfp4(payload: bytes, shape: tuple[int, ...]) -> np.ndarray:
    """Decode a registered NVFP4 object to FP32: ``decode(q) * decode(s) / d_w``.

    Planes: packed E2M1 codes, the swizzled E4M3FN K16 block-scale plane, then one
    FP32 divisor per 128-row x 64-col matrix (``block_scale_k16_m128x4_v1``).  The same
    decode the ``NVFP4_MAXABS_DIVISOR_RNE_V1`` encoder profile round-trips against.

    The 16-code group expansion is a loop over the group axis, and every shape is
    stated in full (no ``-1``): this environment's NumPy (Ubuntu ``2.3.5+ds``) folds a
    singleton appended by ``None`` against a *sibling* singleton — two operands that
    standard broadcasting would resolve to ``[…, n, k/16, 16]`` here fail as
    ``[…, n, k/16]`` vs ``[…, n, k/16, 1]`` — and resolves ``reshape(-1)`` against a
    non-element-count "size".  The loop form is the one that is provably equivalent in
    both NumPy semantics.
    """

    shape = tuple(int(v) for v in shape)
    geometry = block_scale_geometry("nvfp4", shape)
    if len(payload) != geometry.payload_bytes:
        raise SystemExit(
            f"NVFP4 payload has {len(payload)} bytes, expected {geometry.payload_bytes}"
        )
    raw = np.frombuffer(bytearray(payload), dtype=np.uint8)
    rows = geometry.matrices * geometry.n
    codes = raw[: geometry.code_plane_bytes].reshape(rows, geometry.k // 2)
    scales = unswizzle_nvfp4_scales(
        raw[geometry.scale_plane_offset : geometry.scale_plane_offset
            + geometry.scale_plane_bytes],
        shape,
    ).reshape(rows, geometry.groups_per_row)
    divisor = raw[geometry.matrix_scale_offset :].view(np.float32).astype(np.float32)
    if ((scales & 0x80) != 0).any() or (scales == 0x7F).any():
        raise SystemExit("NVFP4 scale word is signed/NaN")
    if not np.isfinite(divisor).all() or (divisor <= 0).any():
        raise SystemExit("NVFP4 matrix divisor must be finite and positive")
    # E2M1 nibble: bit3 sign, bits0-2 magnitude — the P0b reference's own decode
    # (tests/dspine_parity/reference.py).  Signed-view indexing keeps the table small.
    low = np.take(_e2m1_table(), (codes & 0x07).astype(np.uint8))
    np.copyto(low, -low, where=(codes & 0x08).astype(bool))
    high = np.take(_e2m1_table(), ((codes >> 4) & 0x07).astype(np.uint8))
    np.copyto(high, -high, where=(codes & 0x80).astype(bool))
    group = np.take(_e4m3fn_table(), scales)
    values = np.empty((rows, geometry.k), dtype=np.float32)
    for column in range(geometry.groups_per_row):
        # One scale per 16 codes = 8 bytes; the even column is the low nibble, the odd
        # one the high nibble of the same byte (``_dequant_nvfp4`` in
        # ``tests/dspine_parity/fixture``).  Column slicing throughout — this
        # environment's ``reshape`` cannot be trusted at these element counts.
        scale = np.ascontiguousarray(group[:, column][:, None], dtype=np.float32)
        block = values[:, column * 16 : (column + 1) * 16]
        block[:, 0::2] = low[:, column * 8 : (column + 1) * 8] * scale
        block[:, 1::2] = high[:, column * 8 : (column + 1) * 8] * scale
    values = values.reshape(geometry.matrices, geometry.n, geometry.k)
    for matrix in range(geometry.matrices):
        # One FP32 divisor per matrix: the encoder stores ``amax / 6`` and the
        # dequant divides by it (tools/convert/quantization/nvfp4_maxabs.py,
        # ``NVFP4_MAXABS_DIVISOR_RNE_V1`` — the profile P0b round-tripped).
        values[matrix] = values[matrix] / divisor[matrix]
    return np.ascontiguousarray(
        values.reshape(*(
            (geometry.matrices, geometry.n, geometry.k)
            if len(shape) == 3
            else (geometry.n, geometry.k)
        )),
        dtype=np.float32,
    )


# ------------------------------------------------------------------ bf16 decode


def decode_bf16(payload: bytes, shape: tuple[int, ...]) -> np.ndarray:
    bits = np.frombuffer(bytearray(payload), dtype=np.uint16)
    expected = int(np.prod(shape)) if shape else 0
    if bits.size != expected:
        raise SystemExit(f"bf16 object has {bits.size} elements, expected {expected}")
    widened = (bits.astype(np.uint32) << 16).view(np.float32)
    return widened.reshape(shape)


# --------------------------------------------------------------------- the load


@dataclass
class DraftObject:
    name: str
    format: str
    layout: str
    shape: tuple[int, ...]
    offset: int
    bytes: int
    region: str
    cache: np.ndarray | None = None

    def elements(self) -> int:
        return int(np.prod(self.shape)) if self.shape else 0


@dataclass
class DspineLoad:
    """One DSpine draft region of one artifact, bound and (on demand) materialised."""

    path: str
    directory: dict
    base: int
    config: dict
    bridge: dict
    second_pass: dict
    objects: dict[str, DraftObject] = field(default_factory=dict)
    bytes_by_region: dict[str, int] = field(default_factory=dict)
    referenced: set[str] = field(default_factory=set)
    parts: dict[str, list[tuple[str, int, int]]] = field(default_factory=dict)
    bind_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    dequant_events: list[str] = field(default_factory=list)

    # geometry (the fields the forward needs)
    hidden: int = 0
    vocab: int = 0
    layers: int = 0
    code_size: int = 0
    rank: int = 0
    eps: float = 0.0
    block_size: int = 0
    head_topk: int = 0
    mask_token_id: int = 0
    taps: tuple[int, ...] = ()
    rms_eps: float = 0.0
    intermediate: int = 0
    heads: int = 0
    kv_heads: int = 0
    head_dim: int = 0

    def query_width(self) -> int:
        return self.heads * self.head_dim

    def key_width(self) -> int:
        return self.kv_heads * self.head_dim

    def qkv_row_stride(self) -> int:
        """Total rows of a packed ``attention/query_key_value`` object."""

        return self.query_width() + 2 * self.key_width()

    def qkv_bands(self) -> list[tuple[int, int, int]]:
        """``(query, key, value)`` row counts **per head**, the ``linear_split`` order.

        ``linear_split`` emits the QKV projections head-interleaved — one head's
        query, key and value rows adjacent — and splits heads across the GQA groups
        (``src/ops/linear/linear_split.h:100-112``).  A ``query_key_value`` object of
        ``query_width + 2 * key_width`` rows is therefore *not* three contiguous
        blocks when ``heads != kv_heads``: the per-head key/value slice is
        ``key_width // kv_heads`` rows, not the full ``key_width``.
        """
        if self.heads % self.kv_heads:
            raise ValueError(
                f"heads {self.heads} is not a multiple of kv heads {self.kv_heads}"
            )
        group = self.heads // self.kv_heads
        if self.head_dim % group:
            raise ValueError(
                f"head_dim {self.head_dim} is not a multiple of the GQA group {group}"
            )
        return [(self.head_dim, self.head_dim // group, self.head_dim // group)
                for _ in range(self.heads)]

    table_encoding: str = ""       # "nvfp4" | "bf16" (D-20: the runtime picks by the artifact)
    backbone_encoding: str = ""

    def total_bytes(self) -> int:
        return sum(item.bytes for item in self.objects.values())

    def names(self) -> list[str]:
        return sorted(self.objects)

    # -- materialisation -----------------------------------------------------

    def tensor(self, name: str) -> np.ndarray:
        """Materialise one draft tensor in FP32 (BF16 values widened exactly)."""

        item = self.objects.get(name)
        if item is None:
            raise KeyError(f"{self.path}: draft object {name!r} is absent")
        if item.cache is not None:
            return item.cache
        payload = _read_at(self.path, self.base + item.offset, item.bytes)
        if item.format == BF16_CODE:
            values = decode_bf16(payload, item.shape)
        elif item.format == NVFP4_CODE:
            values = dequant_nvfp4(payload, item.shape)
            self.dequant_events.append(f"{name}:{item.shape}")
        else:
            raise SystemExit(f"{name}: draft object format {item.format!r} is not loadable")
        item.cache = values
        return values

    def sha256(self, name: str) -> str:
        item = self.objects[name]
        digest = hashlib.sha256()
        remaining = item.bytes
        handle = open(self.path, "rb")
        with handle:
            handle.seek(self.base + item.offset)
            while remaining:
                block = handle.read(min(1 << 26, remaining))
                if not block:
                    raise SystemExit(f"{name}: payload truncated")
                digest.update(block)
                remaining -= len(block)
        return digest.hexdigest()

    def free(self, *names: str) -> None:
        for name in names:
            if name in self.objects:
                self.objects[name].cache = None


def _region(name: str, config: dict) -> str:
    rest = name[len(PREFIX) + 1 :]
    if rest.startswith("bridge/"):
        tail = rest[len("bridge/") :]
        if tail in ("tg", "tsv"):
            return "serial_tables"
        if tail == "codebook":
            return "codebook"
        return "code_bridge"
    tail = rest.rsplit("/", 1)[-1]
    if tail.endswith("_norm"):
        return "norms"
    return "backbone"


def load_draft(path: str, *, verbose: bool = True) -> DspineLoad:
    """Bind the ``dspine`` component of ``path`` (shapes/formats validated)."""

    directory, base = read_directory(path)
    components = directory.get("components", {})
    if "dspine" not in components:
        raise SystemExit(f"{path}: artifact carries no dspine component")
    component = components["dspine"]
    config = component["config"]
    draft = config["dflash_config"]
    bridge = dict(draft["selected_prefix_code_bridge"])
    second_pass = dict(config.get("code_bridge_second_pass", {}))

    load = DspineLoad(
        path=path,
        directory=directory,
        base=base,
        config=config,
        bridge=bridge,
        second_pass=second_pass,
        hidden=int(config["hidden_size"]),
        vocab=int(config["vocab_size"]),
        layers=int(config["num_hidden_layers"]),
        code_size=int(bridge["code_size"]),
        rank=int(bridge["rank"]),
        eps=float(bridge["eps"]),
        block_size=int(config["block_size"]),
        head_topk=int(draft["selected_prefix_last_write_refine"]["head_topk"]),
        mask_token_id=int(draft["mask_token_id"]),
        taps=tuple(int(v) for v in draft["target_layer_ids"]),
        rms_eps=float(config["rms_norm_eps"]),
        intermediate=int(config["intermediate_size"]),
        heads=int(config["num_attention_heads"]),
        kv_heads=int(config["num_key_value_heads"]),
        head_dim=int(config["head_dim"]),
    )

    objects = {item["id"]: item for item in directory["objects"]}
    bindings = directory.get("bindings", {})
    # The engine binds by *parameter* name (dspine::bind_weights): a packed parent
    # (query_key_value, gate_up) is a tensor object with NO binding of its own and is
    # reached through the parts ranges of its children.  The inventory is therefore
    # the union of the draft bindings and every draft-named object they reference.
    for name in sorted(n for n in bindings if n.startswith(f"{PREFIX}/")):
        parts = bindings[name].get("parts") or [bindings[name]]
        for part in parts:
            item = objects.get(part["object"])
            if item is None:
                raise SystemExit(f"{name}: binding names an unknown object")
            if part["object"].startswith(f"{PREFIX}/") and len(parts) > 1:
                raise SystemExit(f"{name}: draft packed parent spans several objects")
            load.referenced.add(part["object"])
            span = part.get("range")
            load.parts.setdefault(name, []).append(
                (part["object"], None if span is None else tuple(span))
            )
    for name in sorted(load.referenced):
        if not name.startswith(f"{PREFIX}/"):
            continue
        item = objects[name]
        shape = tuple(int(v) for v in item["shape"])
        want = _object_shapes(load).get(name)
        if want is None:
            raise SystemExit(f"{name}: draft object is outside the bind geometry")
        if shape != tuple(want):
            raise SystemExit(f"{name}: declared shape {shape} != bind geometry {tuple(want)}")
        if item["format"] not in (BF16_CODE, NVFP4_CODE):
            raise SystemExit(f"{name}: unexpected draft format {item['format']!r}")
        load.objects[name] = DraftObject(
            name=name,
            format=item["format"],
            layout=item["layout"],
            shape=shape,
            offset=int(item["offset"]),
            bytes=int(item["bytes"]),
            region=_region(name, config),
        )
    missing = sorted(set(_object_shapes(load)) - set(load.objects))
    if missing:
        raise SystemExit(f"{path}: draft objects absent from the artifact: {missing}")
    load.bind_shapes = _bind_shapes(load)

    for region in REGIONS:
        load.bytes_by_region[region] = sum(
            item.bytes for item in load.objects.values() if item.region == region
        )
    encodings = {
        name[len(PREFIX) + 1 :]: item.format
        for name, item in load.objects.items()
        if name[len(PREFIX) + 1 :] in ("bridge/codebook", "bridge/tg", "bridge/tsv")
    }
    table_set = set(encodings.values())
    if len(table_set) != 1:
        raise SystemExit(f"{path}: draft table encodings disagree: {encodings}")
    load.table_encoding = table_set.pop()
    backbone = {
        item.format
        for name, item in load.objects.items()
        if item.region == "backbone" and item.elements() >= 1 << 20
    }
    load.backbone_encoding = "+".join(sorted(backbone))
    if verbose:
        print_report(load)
    return load


def _bind_shapes(load: DspineLoad) -> dict[str, tuple[int, ...]]:
    """The parameter-name geometry ``dspine::bind_weights`` binds by.

    A packed parent is bound as one parameter with its own ``[rows, hidden]`` geometry
    (the engine's ``Bindings::parameter`` validates that shape against the binding's
    parts), and the attention/MLP children address slices of it.
    """

    shapes = dict(_object_shapes(load))
    for layer in range(load.layers):
        p = f"{PREFIX}/layers/{layer}/"
        shapes[p + "attention/query_key_value"] = (load.query_width() + 2 * load.key_width(), load.hidden)
        shapes[p + "attention/query"] = (load.query_width(), load.hidden)
        shapes[p + "attention/key"] = (load.key_width(), load.hidden)
        shapes[p + "attention/value"] = (load.key_width(), load.hidden)
        shapes[p + "mlp/gate_up"] = (2 * load.intermediate, load.hidden)
        shapes[p + "mlp/gate"] = (load.intermediate, load.hidden)
        shapes[p + "mlp/up"] = (load.intermediate, load.hidden)
    return shapes


def _object_shapes(load: DspineLoad) -> dict[str, tuple[int, ...]]:
    h, i = load.hidden, load.intermediate
    q, k, d = load.heads * load.head_dim, load.kv_heads * load.head_dim, load.head_dim
    c, r, v = load.code_size, load.rank, load.vocab
    n = load.layers
    out: dict[str, tuple[int, ...]] = {
        f"{PREFIX}/feature_projection": (h, len(load.taps) * h),
        f"{PREFIX}/context_norm": (h,),
        f"{PREFIX}/final_norm": (h,),
        f"{PREFIX}/bridge/codebook": (v, c),
        f"{PREFIX}/bridge/early_read_weights": (n - 1, c, h),
        f"{PREFIX}/bridge/read_weight": (c, h),
        f"{PREFIX}/bridge/feature_write_weights": (n, h, r),
        f"{PREFIX}/bridge/gate": (r, c),
        f"{PREFIX}/bridge/gate_bias": (r,),
        f"{PREFIX}/bridge/predecessor_gate": (r, c),
        f"{PREFIX}/bridge/prev": (r, c),
        f"{PREFIX}/bridge/tg": (v, r),
        f"{PREFIX}/bridge/tsv": (v, 2 * r),
    }
    for layer in range(n):
        p = f"{PREFIX}/layers/{layer}/"
        out[p + "input_norm"] = (h,)
        out[p + "post_attention_norm"] = (h,)
        out[p + "attention/query_key_value"] = (q + 2 * k, h)
        out[p + "attention/query_norm"] = (d,)
        out[p + "attention/key_norm"] = (d,)
        out[p + "attention/output"] = (h, q)
        out[p + "mlp/gate_up"] = (2 * i, h)
        out[p + "mlp/down"] = (h, i)
    return out


def print_report(load: DspineLoad) -> None:
    print(f"artifact        : {load.path}")
    print(
        f"draft config    : hidden={load.hidden} vocab={load.vocab} layers={load.layers} "
        f"block_size={load.block_size} head_topk={load.head_topk} mask={load.mask_token_id}"
    )
    print(
        f"code bridge     : code_size={load.code_size} rank={load.rank} eps={load.eps} "
        f"kappa={load.bridge.get('kappa')} decode_mode="
        f"{load.config['dflash_config'].get('selected_prefix_decode_mode')}"
    )
    print(f"target taps     : {list(load.taps)}  num_target_layers={load.config['num_target_layers']}")
    print(f"second pass     : {load.second_pass}")
    print(f"table encoding  : {load.table_encoding}   backbone: {load.backbone_encoding}")
    print(f"objects bound   : {len(load.objects)}")
    for region in REGIONS:
        count = sum(1 for item in load.objects.values() if item.region == region)
        if count:
            print(f"  {region:<14} {count:>3} objects  {load.bytes_by_region[region]:>13,} B")
    print(f"  {'TOTAL':<14} {len(load.objects):>3} objects  {load.total_bytes():>13,} B")


def contract_check(load: DspineLoad, reference: dict | None = None) -> list[str]:
    """Compare the recorded draft config against the D-20 contract."""

    draft = load.config["dflash_config"]
    found = {
        "block_size": load.block_size,
        "code_size": load.code_size,
        "rank": load.rank,
        "eps": load.eps,
        "kappa": load.bridge.get("kappa"),
        "decode_mode": draft.get("selected_prefix_decode_mode"),
        "target_layer_ids": list(draft["target_layer_ids"]),
        "mask_token_id": load.mask_token_id,
        "first_exact": load.second_pass.get("first_exact"),
        "head_topk": load.head_topk,
        "num_target_layers": int(load.config["num_target_layers"]),
        "num_hidden_layers": load.layers,
    }
    want = dict(CONTRACT)
    if reference is not None:
        ref_draft = reference["dflash_config"]
        want.update(
            {
                "block_size": int(reference["block_size"]),
                "code_size": int(ref_draft["selected_prefix_code_bridge"]["code_size"]),
                "rank": int(ref_draft["selected_prefix_code_bridge"]["rank"]),
                "eps": float(ref_draft["selected_prefix_code_bridge"]["eps"]),
                "kappa": ref_draft["selected_prefix_code_bridge"].get("kappa"),
                "decode_mode": ref_draft.get("selected_prefix_decode_mode"),
                "target_layer_ids": list(ref_draft["target_layer_ids"]),
                "mask_token_id": int(ref_draft["mask_token_id"]),
                "first_exact": int(reference["code_bridge_second_pass"]["first_exact"]),
                "head_topk": int(
                    ref_draft["selected_prefix_last_write_refine"]["head_topk"]
                ),
                "num_target_layers": int(reference["num_target_layers"]),
                "num_hidden_layers": int(reference["num_hidden_layers"]),
            }
        )
    return [
        f"{key}: {found[key]!r} != {want[key]!r}"
        for key in sorted(want)
        if found[key] != want[key]
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="tools.dspine.artifact", description=__doc__)
    parser.add_argument("artifact")
    parser.add_argument(
        "--reference-config",
        help="release config.json of the draft the artifact was built from (D-20 check)",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--objects", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load = load_draft(args.artifact, verbose=not args.json)
    problems: list[str] = []
    if args.reference_config:
        reference = json.loads(Path(args.reference_config).read_text())
        problems = contract_check(load, reference)
    if args.objects:
        for name in load.names():
            item = load.objects[name]
            print(
                f"{item.bytes:>13,}  {item.format:<5} {item.layout:<24} "
                f"{str(list(item.shape)):<24} {item.region:<13} {name}"
            )
    if args.json:
        print(
            json.dumps(
                {
                    "artifact": load.path,
                    "objects": len(load.objects),
                    "bytes": load.total_bytes(),
                    "bytes_by_region": load.bytes_by_region,
                    "table_encoding": load.table_encoding,
                    "backbone_encoding": load.backbone_encoding,
                    "hidden": load.hidden,
                    "vocab": load.vocab,
                    "layers": load.layers,
                    "code_size": load.code_size,
                    "rank": load.rank,
                    "block_size": load.block_size,
                    "head_topk": load.head_topk,
                    "taps": list(load.taps),
                    "contract_problems": problems,
                },
                indent=2,
            )
        )
    if problems:
        for problem in problems:
            print(f"CONTRACT MISMATCH: {problem}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
