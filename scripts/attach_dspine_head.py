#!/usr/bin/env python3
"""Attach a DSpine draft head to an existing NInfer v3 target artifact.

    python3 scripts/attach_dspine_head.py TARGET.ninfer DSPINE_DIR OUTPUT.ninfer

``DSPINE_DIR`` is the ``tencent/Qwen3.8-27B-DSpine-b16`` checkpoint directory
(``config.json`` + ``model.safetensors``).  Backbone linears are quantized to
NVFP4 (weight-only, ``A16Only``); the code bridge, norms and codebook stay BF16.
Weight bytes of the target are copied verbatim and the DSpine payload region is
appended after the target payload.

The two vocabulary-wide refinement tables (``dspine/bridge/tg`` and
``dspine/bridge/tsv``) are **tower-specific**: they are computed here from the
``text/output_head`` of the artifact named by ``TARGET.ninfer``, never copied
from the release.  A draft attached to the wrong tower is a weaker A/B, so the
head and table checksums are recorded in the artifact ``provenance``.

``--tables-nvfp4`` stores the codebook and both refinement tables as NVFP4
instead of BF16; the DSpine gather kernels dequantize rows on access.  It trades
~870 MiB of draft bytes for quantization error in the serial tail.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.artifact.codecs.direct import encode_direct  # noqa: E402
from tools.artifact.codecs.fp8_row import dequantize_fp8_row_scaled  # noqa: E402
from tools.artifact.codecs.row_split import dequantize_row_split  # noqa: E402
from tools.artifact.framing import (  # noqa: E402
    HEADER,
    MAGIC,
    PAYLOAD_ALIGNMENT,
)
from tools.artifact.schema import parse_directory  # noqa: E402
from tools.convert.qwen3_8_27b import dspine_recipe  # noqa: E402
from tools.convert.qwen3_8_27b.recipe import ShardReader  # noqa: E402
from tools.convert.quantization.groupwise import pick_device  # noqa: E402
from tools.convert.quantization.nvfp4_maxabs import (  # noqa: E402
    ENCODER_PROFILE,
    encode_nvfp4_parent,
    quantize_nvfp4,
    relative_frobenius_error,
    self_test,
)
from tools.convert.quantization.nvfp4_maxabs import (  # noqa: E402
    dequantize_nvfp4 as _dequantize_nvfp4,
)

ALIGN = PAYLOAD_ALIGNMENT
OBJ_ALIGN = 256
CHUNK = 64 * 1024 * 1024
HEAD_ROW_CHUNK = 8192
PREFIX = dspine_recipe.PREFIX
HEAD_ROW_SCALE = "row_scale_v1"
HEAD_ROW_SPLIT = "row_split_k128_v1"
HEAD_FORMATS = frozenset(("fp8_e4m3fn_row_bf16", "q8_g32_fp16"))


def dequantize_head(payload: bytes, head: dict) -> torch.Tensor:
    """Reconstruct a deployed ``text/output_head`` in dense FP32.

    The public DSpine converter accepts only the row-scaled FP8 head NVIDIA
    publishes.  This repository quantizes a Qwen3.8-27B vocabulary head with
    grouped codes (``q8_g32_fp16`` / ``row_split_k128_v1``) — as the Swift-1.5 V3
    tower does — and that is exactly the head the Engine multiplies
    ``dspine/final_hidden`` by, so the refinement tables are derived from the
    representation the serial tail shares with the target.
    """

    fmt, layout = head["format"], head["layout"]
    shape = tuple(head["shape"])
    if fmt not in HEAD_FORMATS:
        raise SystemExit(f"text/output_head: unsupported format {fmt!r}")
    if layout == HEAD_ROW_SCALE:
        return dequantize_fp8_row_scaled(payload, shape, torch.float32)
    if layout == HEAD_ROW_SPLIT:
        return dequantize_row_split(payload, fmt, shape, "cpu", dtype=torch.float32)
    raise SystemExit(f"text/output_head: unsupported layout {layout!r}")


@dataclass(slots=True)
class Evidence:
    """Checksums proving which tower and which release produced each region."""

    head_format: str = ""
    head_layout: str = ""
    head_shape: tuple[int, ...] = ()
    head_sha256: str = ""
    head_rows: int = 0
    table_sha256: dict[str, str] = field(default_factory=dict)
    quantization_error: dict[str, float] = field(default_factory=dict)


def align(value: int, amount: int = ALIGN) -> int:
    return (value + amount - 1) // amount * amount


def sha256_file_range(path: str, begin: int, length: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        handle.seek(begin)
        remaining = length
        while remaining:
            block = handle.read(min(CHUNK, remaining))
            if not block:
                raise SystemExit(f"{path}: payload truncated")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    host = tensor.detach().cpu().contiguous()
    return hashlib.sha256(
        host.view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def read_directory(path: str) -> tuple[dict, int]:
    with open(path, "rb") as handle:
        magic, json_bytes, _ = HEADER.unpack(handle.read(HEADER.size))
        if magic != MAGIC:
            raise SystemExit(f"{path}: not a NInfer v3 artifact (magic {magic!r})")
        raw = handle.read(json_bytes)
    directory = json.loads(raw)
    payload = directory["files"][0]["payload_bytes"]
    if align(HEADER.size + json_bytes) + payload != os.path.getsize(path):
        raise SystemExit(f"{path}: entry length differs from directory")
    # Structural admission control: a target that is not a valid v3 directory
    # must not be copied forward.
    parse_directory(directory, entry_name=Path(path).name)
    return directory, HEADER.size + json_bytes


def base_summary(target: dict) -> dict:
    text = target["components"]["text"]["config"]
    return {
        "text": {
            name: text[name]
            for name in (
                "hidden_size",
                "vocab_size",
                "num_hidden_layers",
                "max_position_embeddings",
            )
        },
        "rope": {"rope_theta": text["rope_parameters"]["rope_theta"]},
    }


def serialize(directory: dict) -> tuple[bytes, int]:
    reserve = ALIGN - HEADER.size
    while True:
        data = json.dumps(
            directory, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
        if len(data) <= reserve:
            return data + b" " * (reserve - len(data)), reserve
        reserve = align(HEADER.size + len(data)) - HEADER.size


def encoded_object(
    name: str,
    tensor: torch.Tensor,
    shape: tuple[int, ...],
    device: torch.device,
    gather_tables_nvfp4: bool,
    evidence: Evidence,
) -> tuple[dict, bytes]:
    fmt = dspine_recipe.object_format(name, gather_tables_nvfp4)
    if tuple(tensor.shape) != shape:
        raise SystemExit(f"{name}: shape {tuple(tensor.shape)} != {shape}")
    if fmt == dspine_recipe.NVFP4:
        words = quantize_nvfp4(tensor, device=device)
        payload = encode_nvfp4_parent(tensor, device=device)
        error = relative_frobenius_error(
            tensor, _dequantize_nvfp4(words, device=device)
        )
        evidence.quantization_error[name] = error
        print(f"  {name}  NVFP4 {shape}  rel_err={error:.5f}", flush=True)
        del words
        layout = dspine_recipe.NVFP4_LAYOUT
    else:
        payload = encode_direct(tensor, fmt)
        layout = dspine_recipe.CONTIGUOUS_LAYOUT
    return (
        {
            "id": name,
            "kind": "tensor",
            "shape": list(shape),
            "format": fmt,
            "layout": layout,
            "bytes": len(payload),
        },
        payload,
    )


def build_payloads(
    dspine_dir: Path,
    device: torch.device,
    gather_tables_nvfp4: bool,
    evidence: Evidence,
) -> list[tuple[dict, bytes]]:
    entries: list[tuple[dict, bytes]] = []
    with ShardReader.from_file(dspine_dir / "model.safetensors") as reader:
        for recipe in dspine_recipe.DSPINE_RECIPE_SPECS:
            name = recipe.object_name
            tensor = dspine_recipe.materialize_tensor(name, reader)
            shape = dspine_recipe.object_shape(name)
            entry, payload = encoded_object(
                name,
                tensor,
                shape,
                device,
                gather_tables_nvfp4,
                evidence,
            )
            entries.append((entry, payload))
            del tensor, payload
    return entries


def read_target_head(
    target_path: str, target: dict, payload_start: int
) -> tuple[torch.Tensor, str, dict]:
    """Dequantize the target's row-scaled FP8 output head into dense FP32."""

    head_binding = target["bindings"]["text/output_head"]
    if set(head_binding) != {"object"}:
        raise SystemExit("text/output_head must be a single-object binding")
    objects = {obj["id"]: obj for obj in target["objects"]}
    head_obj = objects[head_binding["object"]]
    shape = tuple(head_obj["shape"])
    if len(shape) != 2:
        raise SystemExit("text/output_head must be rank 2")
    head_sha256 = sha256_file_range(
        target_path, payload_start + head_obj["offset"], head_obj["bytes"]
    )
    with open(target_path, "rb") as handle:
        handle.seek(payload_start + head_obj["offset"])
        head_payload = handle.read(head_obj["bytes"])
    if len(head_payload) != head_obj["bytes"]:
        raise SystemExit("text/output_head payload truncated")
    head = dequantize_head(head_payload, head_obj)
    del head_payload
    return head, head_sha256, head_obj


def _json_bytes(path: str) -> int:
    with open(path, "rb") as handle:
        return HEADER.unpack(handle.read(HEADER.size))[1]


def compute_serial_tables(
    target_path: str,
    target: dict,
    dspine_dir: Path,
    device: torch.device,
    gather_tables_nvfp4: bool,
    evidence: Evidence,
) -> list[tuple[dict, bytes]]:
    """Precompute the vocabulary-wide greedy refinement tables for this tower.

    Mirrors ``dspine_mlx.bridge.build_serial_tables`` rounding point by rounding
    point: ``TG`` rows are ``bf16(fp32(lm_head) * final_norm) @ write_w[-1]`` and
    ``TSV`` rows are ``bf16(sqrt(C) * codebook) @ [predecessor_gate; prev].T``.
    The target head is stored row-scaled FP8, so it is dequantized to dense
    FP32 first; the checksum of those stored bytes is the tailoring evidence.
    """

    import math

    head, head_sha256, head_obj = read_target_head(
        target_path, target, align(HEADER.size + _json_bytes(target_path))
    )
    head = head.to(device)
    evidence.head_format = head_obj["format"]
    evidence.head_layout = head_obj["layout"]
    evidence.head_shape = tuple(head_obj["shape"])
    evidence.head_sha256 = head_sha256
    evidence.head_rows = head_obj["shape"][0]

    with ShardReader.from_file(dspine_dir / "model.safetensors") as reader:
        norm_w = dspine_recipe.materialize_tensor("dspine/final_norm", reader)
        write_w = dspine_recipe.materialize_tensor(
            "dspine/bridge/feature_write_weights", reader
        )[-1]
        pred = dspine_recipe.materialize_tensor(
            "dspine/bridge/predecessor_gate", reader
        )
        prev = dspine_recipe.materialize_tensor("dspine/bridge/prev", reader)
        codebook = dspine_recipe.materialize_tensor(
            "dspine/bridge/codebook", reader
        )
    norm_w, write_w, pred, prev, codebook = (
        value.to(device) for value in (norm_w, write_w, pred, prev, codebook)
    )
    code_size = codebook.shape[1]
    pv = torch.cat([pred, prev], dim=0)
    sqrt_c = math.sqrt(code_size)
    norm_f = norm_w.float()

    vocab = head_obj["shape"][0]
    tg_parts: list[torch.Tensor] = []
    tsv_parts: list[torch.Tensor] = []
    for start in range(0, vocab, HEAD_ROW_CHUNK):
        end = min(start + HEAD_ROW_CHUNK, vocab)
        tg_parts.append(
            (head[start:end].float() * norm_f).to(torch.bfloat16) @ write_w
        )
        tsv_parts.append(
            (sqrt_c * codebook[start:end].float()).to(torch.bfloat16) @ pv.T
        )
    tg = torch.cat(tg_parts, dim=0).contiguous()
    tsv = torch.cat(tsv_parts, dim=0).contiguous()
    del head, tg_parts, tsv_parts

    entries = []
    for name, tensor in (("dspine/bridge/tg", tg), ("dspine/bridge/tsv", tsv)):
        shape = dspine_recipe.SERIAL_TABLE_OBJECTS[name]
        evidence.table_sha256[name] = sha256_tensor(tensor)
        entry, payload = encoded_object(
            name, tensor, shape, device, gather_tables_nvfp4, evidence
        )
        entries.append((entry, payload))
    print(
        f"  serial tables from {head_obj['id']} ({head_obj['format']})  "
        f"TG {tuple(tg.shape)}  TSV {tuple(tsv.shape)}  "
        f"head sha256 {head_sha256[:16]}…",
        flush=True,
    )
    del tg, tsv
    return entries


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attach a DSpine draft head to a NInfer v3 target artifact."
    )
    parser.add_argument("target", help="v3 target artifact without a DSpine head")
    parser.add_argument("dspine_dir", help="DSpine checkpoint directory")
    parser.add_argument("output", help="fused v3 artifact to create")
    parser.add_argument(
        "--tables-nvfp4",
        action="store_true",
        default=os.environ.get("NINFER_DSPINE_TABLES_NVFP4") == "1",
        help="store codebook/TG/TSV as NVFP4 instead of BF16 (saves ~870 MiB)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="conversion device; CUDA is used when available (default: cuda)",
    )
    parser.add_argument(
        "--report",
        help="write the provenance/evidence JSON here",
    )
    parser.add_argument(
        "--skip-self-test",
        action="store_true",
        help="do not verify the NVFP4 encoder words before converting",
    )
    parser.add_argument(
        "--artifact-id",
        help=(
            "32 hex digits for the 16-byte v3 artifact id; the default is random, "
            "so pass a fixed id to make the whole file byte-reproducible"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    target_path, dspine_dir, output_path = args.target, Path(args.dspine_dir), args.output
    if os.path.exists(output_path):
        raise SystemExit(f"refusing to overwrite {output_path}")
    for name in ("config.json", "model.safetensors"):
        if not (dspine_dir / name).is_file():
            raise SystemExit(f"{dspine_dir}: missing {name}")
    device = pick_device(args.device)
    if not args.skip_self_test:
        self_test(device)
        print(f"nvfp4 encoder self-test ok on {device}", flush=True)

    target, _ = read_directory(target_path)
    if any(obj["id"].startswith(PREFIX) for obj in target["objects"]):
        raise SystemExit("target artifact already carries a DSpine head")

    config = json.loads((dspine_dir / "config.json").read_text(encoding="utf-8"))
    summary = dspine_recipe.validate_config(config)
    dspine_recipe.validate_base_compatibility(base_summary(target), summary)
    preflight = dspine_recipe.preflight_sources(dspine_dir)
    print(
        f"source: {dspine_recipe.REPOSITORY}@{dspine_recipe.REVISION[:12]}\n"
        f"preflight: {preflight.recipe_count} recipes, "
        f"{preflight.source_tensor_count} source tensors, "
        f"{preflight.source_dtype_counts}",
        flush=True,
    )

    evidence = Evidence(head_format="", head_layout="", head_shape=(), head_sha256="", head_rows=0)
    entries = build_payloads(
        dspine_dir, device, args.tables_nvfp4, evidence
    )
    entries += compute_serial_tables(
        target_path, target, dspine_dir, device, args.tables_nvfp4, evidence
    )

    target_payload = target["files"][0]["payload_bytes"]
    last_end = max(obj["offset"] + obj["bytes"] for obj in target["objects"])
    if last_end > target_payload:
        raise SystemExit("target payload does not cover its own objects")
    new_base = align(max(target_payload, last_end))

    cursor = new_base
    attached = []
    for obj, payload in entries:
        cursor = align(cursor, OBJ_ALIGN)
        attached.append(dict(obj, offset=cursor))
        cursor += len(payload)
    payload_total = cursor

    merged = dict(target)
    merged["components"] = dict(target["components"])
    merged["components"]["dspine"] = dspine_recipe.build_component(summary)
    merged["objects"] = list(target["objects"]) + attached
    merged["bindings"] = dict(target["bindings"])
    merged["bindings"].update(dspine_recipe.build_bindings())
    merged["uses"] = list(target["uses"]) + dspine_recipe.build_uses()
    merged["files"] = [{"path": None, "payload_bytes": payload_total}]
    merged["provenance"] = dict(target.get("provenance") or {})
    merged["provenance"]["dspine"] = {
        "repository": dspine_recipe.REPOSITORY,
        "revision": dspine_recipe.REVISION,
        "encoder_profile": ENCODER_PROFILE,
        "gather_tables_format": (
            "nvfp4" if args.tables_nvfp4 else "bf16"
        ),
        "target_artifact_id": _artifact_id(target_path),
        "serial_tables": {
            "computed_from": "text/output_head of " + os.path.basename(target_path),
            "head_object": target["bindings"]["text/output_head"]["object"],
            "head_format": evidence.head_format,
            "head_layout": evidence.head_layout,
            "head_shape": list(evidence.head_shape),
            "head_sha256": evidence.head_sha256,
            "table_sha256": dict(evidence.table_sha256),
        },
        "quantization_relative_frobenius": dict(evidence.quantization_error),
    }
    # Structural admission control before a single byte is written.
    parse_directory(merged, entry_name=Path(output_path).name)

    data, reserve = serialize(merged)
    region = payload_total - new_base
    print(
        f"target payload {target_payload:,} | dspine region {region:,} in "
        f"{len(attached)} objects | json reserve {reserve:,} | "
        f"entry bytes {HEADER.size + reserve + payload_total:,}",
        flush=True,
    )
    if args.report:
        Path(args.report).write_text(
            json.dumps(merged["provenance"]["dspine"], indent=2) + "\n",
            encoding="utf-8",
        )

    with open(target_path, "rb") as source, open(output_path, "wb") as out:
        out.write(HEADER.pack(MAGIC, reserve, _artifact_id_bytes(args.artifact_id)))
        out.write(data)
        remaining = target_payload
        source.seek(align(HEADER.size + _json_bytes(target_path)))
        while remaining:
            block = source.read(min(CHUNK, remaining))
            if not block:
                raise SystemExit("target payload truncated")
            out.write(block)
            remaining -= len(block)
        payload_start = HEADER.size + reserve
        for obj, (_, payload) in zip(attached, entries):
            gap = payload_start + obj["offset"] - out.tell()
            if gap < 0:
                raise SystemExit(f"{obj['id']}: overlapping payload layout")
            out.write(b"\0" * gap)
            out.write(payload)
        if out.tell() != HEADER.size + reserve + payload_total:
            raise SystemExit("output layout does not match directory")
        out.flush()
        os.fsync(out.fileno())

    if os.path.getsize(output_path) != HEADER.size + reserve + payload_total:
        raise SystemExit("output size mismatch")
    check, _ = read_directory(output_path)
    if len(check["objects"]) != len(merged["objects"]):
        raise SystemExit("output directory object count mismatch")
    print(f"wrote {output_path}")


def _artifact_id(path: str) -> str:
    with open(path, "rb") as handle:
        return HEADER.unpack(handle.read(HEADER.size))[2].hex()


def _artifact_id_bytes(value: str | None) -> bytes:
    if value is None:
        return uuid.uuid4().bytes
    try:
        raw = bytes.fromhex(value)
    except ValueError as error:
        raise SystemExit(f"--artifact-id: {error}") from error
    if len(raw) != 16:
        raise SystemExit("--artifact-id: expected 32 hex digits (16 bytes)")
    return raw


if __name__ == "__main__":
    main()
