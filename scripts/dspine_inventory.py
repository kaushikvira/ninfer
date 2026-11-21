#!/usr/bin/env python3
"""Print and verify the DSpine region of a fused NInfer v3 artifact.

    python3 scripts/dspine_inventory.py FUSED.ninfer [options]

Sections

* ``--objects``   every ``dspine/*`` object: id, format/layout, shape, bytes.
* ``--evidence``  the recorded provenance: source release, encoder profile, and
  the checksum of the ``text/output_head`` bytes the TG/TSV tables were computed
  from, plus the table checksums themselves.
* ``--head-artifact A.ninfer``  checksum of a *different* artifact's output head.
  The tower-tailoring claim is that the fused tables name one specific head, so
  any head whose checksum differs is a tower the tables were not built for.
* ``--derive-tables A.ninfer --draft-dir DIR``  recompute TG/TSV from that
  artifact's head with the same code path the attacher uses, and report byte
  identity, differing rows, and the relative Frobenius delta against the tables
  stored in ``FUSED.ninfer``.  Byte identity against the fused tower and a large
  delta against a stock tower is the D-06 evidence.
* ``--payload-sha256``  checksum the bytes an artifact run must reproduce
  (directory JSON plus payload, excluding the random artifact id in the header).
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS.parent))
sys.path.insert(0, str(SCRIPTS))

from attach_dspine_head import (  # noqa: E402
    CHUNK,
    HEAD_ROW_CHUNK,
    align,
    dequantize_head,
    read_directory,
    sha256_tensor,
)
from tools.artifact.codecs.direct import decode_direct  # noqa: E402
from tools.artifact.codecs.nvfp4 import decode_nvfp4_words  # noqa: E402
from tools.artifact.framing import HEADER  # noqa: E402
from tools.artifact.schema import parse_directory  # noqa: E402
from tools.convert.qwen3_8_27b import dspine_recipe  # noqa: E402
from tools.convert.quantization.nvfp4_maxabs import (  # noqa: E402
    Nvfp4Words,
    dequantize_nvfp4,
    relative_frobenius_error,
)

PREFIX = dspine_recipe.PREFIX


def head_checksum(path: str) -> tuple[str, dict]:
    directory, start = read_directory(path)
    binding = directory["bindings"]["text/output_head"]
    obj = {item["id"]: item for item in directory["objects"]}[binding["object"]]
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        handle.seek(start + obj["offset"])
        remaining = obj["bytes"]
        while remaining:
            block = handle.read(min(CHUNK, remaining))
            if not block:
                raise SystemExit(f"{path}: output head truncated")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest(), obj


def payload_checksum(path: str) -> str:
    directory, start = read_directory(path)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        handle.seek(HEADER.size)
        digest.update(handle.read(start - HEADER.size))
        handle.seek(start)
        remaining = directory["files"][0]["payload_bytes"]
        while remaining:
            block = handle.read(min(CHUNK, remaining))
            if not block:
                raise SystemExit(f"{path}: payload truncated")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def read_object_tensor(
    path: str, directory: dict, objects: dict[str, dict], name: str, device: str
) -> torch.Tensor:
    obj = objects[name]
    start = align(HEADER.size + _json_bytes(path))
    with open(path, "rb") as handle:
        handle.seek(start + obj["offset"])
        raw = handle.read(obj["bytes"])
    if len(raw) != obj["bytes"]:
        raise SystemExit(f"{name}: payload truncated")
    shape = tuple(obj["shape"])
    if obj["format"] == "bf16":
        return decode_direct(raw, "bf16", shape, device).float()
    if obj["format"] == "nvfp4":
        codes, scales, scale = decode_nvfp4_words(raw, shape)
        words = Nvfp4Words(
            packed_codes=codes,
            natural_scales=scales,
            weight_scale=float(scale.reshape(-1)[0].item()),
        )
        return dequantize_nvfp4(words, device=device).to(device).float()
    raise SystemExit(f"{name}: unsupported table format {obj['format']!r}")


def _json_bytes(path: str) -> int:
    with open(path, "rb") as handle:
        return HEADER.unpack(handle.read(HEADER.size))[1]


def derive_tables(
    source_artifact: str, draft_dir: Path, device: str
) -> dict[str, torch.Tensor]:
    """Recompute TG/TSV from one artifact's head, with the attacher's math."""

    directory, start = read_directory(source_artifact)
    binding = directory["bindings"]["text/output_head"]
    head_obj = {item["id"]: item for item in directory["objects"]}[binding["object"]]
    print(
        f"  head {head_obj['id']} {head_obj['format']}/{head_obj['layout']} "
        f"{head_obj['shape']}",
        flush=True,
    )
    head = dequantize_head(
        _read_at(source_artifact, start + head_obj["offset"], head_obj["bytes"]),
        head_obj,
    ).to(device)

    reader = dspine_recipe.ShardReader.from_file(draft_dir / "model.safetensors")
    with reader:
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
    norm_f = norm_w.float()
    pv = torch.cat([pred, prev], dim=0)
    sqrt_c = math.sqrt(codebook.shape[1])

    parts: dict[str, list[torch.Tensor]] = {"dspine/bridge/tg": [], "dspine/bridge/tsv": []}
    vocab = head_obj["shape"][0]
    for begin in range(0, vocab, HEAD_ROW_CHUNK):
        end = min(begin + HEAD_ROW_CHUNK, vocab)
        parts["dspine/bridge/tg"].append(
            (head[begin:end].float() * norm_f).to(torch.bfloat16) @ write_w
        )
        parts["dspine/bridge/tsv"].append(
            (sqrt_c * codebook[begin:end].float()).to(torch.bfloat16) @ pv.T
        )
    del head, codebook
    return {
        name: torch.cat(values, dim=0).contiguous() for name, values in parts.items()
    }


def _read_at(path: str, begin: int, length: int) -> bytes:
    with open(path, "rb") as handle:
        handle.seek(begin)
        data = handle.read(length)
    if len(data) != length:
        raise SystemExit(f"{path}: short read at {begin}")
    return data


def compare(name: str, stored: torch.Tensor, derived: torch.Tensor) -> None:
    if tuple(stored.shape) != tuple(derived.shape):
        raise SystemExit(
            f"{name}: stored {tuple(stored.shape)} != derived {tuple(derived.shape)}"
        )
    # Both sides are compared in the BF16 domain the tables are stored in, so
    # the checksums are directly comparable with the recorded provenance.
    left = stored.to(torch.bfloat16).contiguous()
    right = derived.to(torch.bfloat16).contiguous()
    same = left == right
    differing_rows = int((~same.all(dim=1)).sum().item())
    error = relative_frobenius_error(stored, derived)
    print(
        f"  {name}: stored bf16 sha256 {sha256_tensor(left)[:16]}…  "
        f"derived bf16 sha256 {sha256_tensor(right)[:16]}…  "
        f"identical={bool(same.all().item())}  "
        f"rows_differing={differing_rows:,}/{stored.shape[0]:,}  "
        f"rel_frobenius={error:.6f}",
        flush=True,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("artifact", help="fused v3 artifact with a DSpine head")
    parser.add_argument("--objects", action="store_true")
    parser.add_argument("--evidence", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--payload-sha256", action="store_true")
    parser.add_argument(
        "--head-artifact",
        action="append",
        default=[],
        metavar="PATH",
        help="checksum this artifact's output head and compare (repeatable)",
    )
    parser.add_argument(
        "--derive-tables",
        action="append",
        default=[],
        metavar="PATH",
        help="recompute TG/TSV from this artifact's head and compare (repeatable)",
    )
    parser.add_argument("--draft-dir", help="DSpine checkpoint directory")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    path = args.artifact
    directory, start = read_directory(path)
    parse_directory(directory, entry_name=Path(path).name)
    objects = {obj["id"]: obj for obj in directory["objects"]}
    if "dspine" not in directory["components"]:
        raise SystemExit(f"{path}: artifact has no dspine component")

    tables = [name for name in dspine_recipe.SERIAL_TABLE_OBJECTS if name in objects]
    dspine_objects = [obj for obj in directory["objects"] if obj["id"].startswith(PREFIX)]
    if not dspine_objects:
        raise SystemExit(f"{path}: artifact has no dspine objects")

    def section(obj: dict) -> str:
        name = obj["id"]
        if name in dspine_recipe.SERIAL_TABLE_OBJECTS:
            return "serial_tables"
        if name.startswith("dspine/bridge/"):
            return "code_bridge"
        if name.endswith("_norm"):
            return "norms"
        return "backbone"

    bytes_by_section: Counter[str] = Counter()
    bytes_by_format: Counter[str] = Counter()
    for obj in dspine_objects:
        bytes_by_section[section(obj)] += obj["bytes"]
        bytes_by_format[f"{obj['format']}/{obj['layout']}"] += obj["bytes"]
    total = sum(obj["bytes"] for obj in dspine_objects)
    declared = {
        "path": path,
        "target_payload_bytes": directory["files"][0]["payload_bytes"],
        "objects": len(dspine_objects),
        "uses": sum(
            1
            for use in directory["uses"]
            if use.get("parameter", "").startswith(PREFIX)
            or use.get("input", "").startswith(PREFIX)
        ),
        "bindings": sum(1 for name in directory["bindings"] if name.startswith(PREFIX)),
        "dspine_bytes": total,
        "dspine_gib": round(total / 2**30, 4),
        "bytes_by_section": dict(sorted(bytes_by_section.items())),
        "bytes_by_format": dict(sorted(bytes_by_format.items())),
        "serial_table_objects": tables,
        "component_config": directory["components"]["dspine"]["config"],
    }
    provenance = (directory.get("provenance") or {}).get("dspine")
    if args.evidence and provenance is None:
        raise SystemExit(f"{path}: no provenance.dspine record")
    if provenance is not None:
        declared["provenance"] = provenance

    print(f"artifact: {path}")
    for key in (
        "objects",
        "bindings",
        "uses",
        "dspine_bytes",
        "dspine_gib",
        "bytes_by_section",
        "bytes_by_format",
        "serial_table_objects",
    ):
        print(f"{key}: {declared[key]}")
    if args.objects:
        for obj in dspine_objects:
            print(
                f"{obj['offset']:>14} {obj['bytes']:>14,} "
                f"{obj['format'] + '/' + obj['layout']:<40} "
                f"{str(obj['shape']):<22} {obj['id']}"
            )
    if args.evidence:
        print(json.dumps(provenance, indent=2))
    if args.payload_sha256:
        print(f"payload_sha256: {payload_checksum(path)}")

    for other in args.head_artifact:
        digest, obj = head_checksum(other)
        recorded = ((provenance or {}).get("serial_tables") or {}).get("head_sha256")
        print(
            f"head {other}: {obj['id']} {obj['format']} {obj['shape']}  "
            f"sha256 {digest[:16]}…  matches_fused_tables={digest == recorded}",
            flush=True,
        )

    if args.derive_tables:
        if not args.draft_dir:
            raise SystemExit("--derive-tables requires --draft-dir")
        draft_dir = Path(args.draft_dir)
        for other in args.derive_tables:
            print(f"derived tables from {other}:", flush=True)
            derived = derive_tables(other, draft_dir, args.device)
            for name, tensor in derived.items():
                compare(name, read_object_tensor(path, directory, objects, name, args.device), tensor)
                del tensor
            del derived

    if args.json:
        print(json.dumps(declared, indent=2))


if __name__ == "__main__":
    main()
