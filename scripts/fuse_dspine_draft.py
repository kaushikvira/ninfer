#!/usr/bin/env python3
"""Fuse a standalone DSpine draft suffix onto a target tower NInfer v3 artifact.

    python3 scripts/fuse_dspine_draft.py TARGET.ninfer DRAFT_SUFFIX.ninfer OUTPUT.ninfer

``TARGET`` is a tower artifact without a DSpine head.  ``DRAFT_SUFFIX`` is
produced by ``scripts/extract_dspine_draft.py`` and already carries the
tower-specific TG/TSV refinement tables, so it must match the tower it was
extracted for: the geometry and RoPE of the two components are checked, and the
head checksum recorded in the suffix provenance is checked against the head
bytes of this target.  The target payload is copied verbatim and the draft
region is appended after it.  This is the layout ``attach_dspine_head.py``
produces, reusing prebuilt draft bytes instead of requantizing the checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import hashlib
import json
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.artifact.framing import (  # noqa: E402
    HEADER,
    MAGIC,
    PAYLOAD_ALIGNMENT,
)
from tools.artifact.schema import parse_directory  # noqa: E402

ALIGN = PAYLOAD_ALIGNMENT
CHUNK = 64 * 1024 * 1024
PREFIX = "dspine/"
HEAD_FORMAT = "fp8_e4m3fn_row_bf16"


def align(value: int, amount: int = ALIGN) -> int:
    return (value + amount - 1) // amount * amount


def read_directory(path: str) -> tuple[dict, int]:
    with open(path, "rb") as handle:
        magic, json_bytes, _ = HEADER.unpack(handle.read(HEADER.size))
        if magic != MAGIC:
            raise SystemExit(f"{path}: not a NInfer v3 artifact (magic {magic!r})")
        directory = json.loads(handle.read(json_bytes))
    payload = directory["files"][0]["payload_bytes"]
    if align(HEADER.size + json_bytes) + payload != os.path.getsize(path):
        raise SystemExit(f"{path}: entry length differs from directory")
    return directory, HEADER.size + json_bytes


def serialize(directory: dict) -> tuple[bytes, int]:
    reserve = ALIGN - HEADER.size
    while True:
        data = json.dumps(
            directory, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
        if len(data) <= reserve:
            return data + b" " * (reserve - len(data)), reserve
        reserve = align(HEADER.size + len(data)) - HEADER.size


def head_record(target: dict, payload_start: int, path: str) -> tuple[str, dict]:
    binding = target["bindings"]["text/output_head"]
    if set(binding) != {"object"}:
        raise SystemExit("text/output_head must be a single-object binding")
    obj = {item["id"]: item for item in target["objects"]}[binding["object"]]
    if obj["format"] != HEAD_FORMAT:
        raise SystemExit(f"text/output_head: unexpected format {obj['format']!r}")
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        handle.seek(payload_start + obj["offset"])
        remaining = obj["bytes"]
        while remaining:
            block = handle.read(min(CHUNK, remaining))
            if not block:
                raise SystemExit("text/output_head payload truncated")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest(), obj


def check_compat(target: dict, draft: dict, target_head_sha256: str) -> None:
    text = target["components"]["text"]["config"]
    config = draft["components"]["dspine"]["config"]
    for key in ("hidden_size", "vocab_size"):
        if text.get(key) != config.get(key):
            raise SystemExit(
                f"draft/target {key} mismatch: {config.get(key)} vs {text.get(key)}"
            )
    if (
        text["rope_parameters"]["rope_theta"]
        != config["rope_parameters"]["rope_theta"]
    ):
        raise SystemExit("draft/target rope_theta mismatch")
    tables = ((draft.get("provenance") or {}).get("dspine") or {}).get(
        "serial_tables"
    )
    recorded = (tables or {}).get("head_sha256")
    if recorded is None:
        print(
            "warning: draft suffix records no head checksum; tower match is "
            "unchecked",
            file=sys.stderr,
            flush=True,
        )
    elif recorded != target_head_sha256:
        raise SystemExit(
            "draft suffix TG/TSV were computed from a different tower: recorded "
            f"head sha256 {recorded[:16]}… but this target has "
            f"{target_head_sha256[:16]}…"
        )


def copy_payload(
    src_path: str, src_start: int, out, begin: int, nbytes: int
) -> None:
    with open(src_path, "rb") as src:
        src.seek(src_start + begin)
        remaining = nbytes
        while remaining:
            block = src.read(min(CHUNK, remaining))
            if not block:
                raise SystemExit("payload truncated")
            out.write(block)
            remaining -= len(block)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse a DSpine draft suffix onto a target tower artifact."
    )
    parser.add_argument("target", help="tower artifact without a DSpine head")
    parser.add_argument("draft", help="draft suffix from extract_dspine_draft.py")
    parser.add_argument("output", help="fused v3 artifact to create")
    parser.add_argument(
        "--allow-unknown-tower",
        action="store_true",
        help="fuse even when the suffix records no head checksum",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    target_path, draft_path, out_path = args.target, args.draft, args.output
    if os.path.exists(out_path):
        raise SystemExit(f"refusing to overwrite {out_path}")
    target, target_start = read_directory(target_path)
    parse_directory(target, entry_name=Path(target_path).name)
    draft, draft_start = read_directory(draft_path)
    if any(obj["id"].startswith(PREFIX) for obj in target["objects"]):
        raise SystemExit("target artifact already carries a DSpine head")
    if not all(obj["id"].startswith(PREFIX) for obj in draft["objects"]):
        raise SystemExit("draft suffix contains non-dspine objects")
    head_sha256, head_obj = head_record(target, target_start, target_path)
    recorded = (
        ((draft.get("provenance") or {}).get("dspine") or {}).get("serial_tables")
        or {}
    ).get("head_sha256")
    if recorded is None and not args.allow_unknown_tower:
        raise SystemExit(
            "draft suffix records no head checksum; pass --allow-unknown-tower"
        )
    check_compat(target, draft, head_sha256)
    print(
        f"target head {head_obj['id']} sha256 {head_sha256[:16]}… "
        f"{'matches the draft suffix' if recorded == head_sha256 else '(unchecked)'}",
        flush=True,
    )

    target_payload = target["files"][0]["payload_bytes"]
    last_end = max(obj["offset"] + obj["bytes"] for obj in target["objects"])
    if last_end > target_payload:
        raise SystemExit("target payload does not cover its own objects")
    new_base = align(max(target_payload, last_end))

    draft_payload = draft["files"][0]["payload_bytes"]
    attached = []
    for obj in draft["objects"]:
        copy = dict(obj)
        copy["offset"] = new_base + obj["offset"]
        attached.append(copy)
    payload_total = new_base + draft_payload

    merged = dict(target)
    merged["components"] = dict(target["components"])
    merged["components"]["dspine"] = draft["components"]["dspine"]
    merged["objects"] = list(target["objects"]) + attached
    merged["bindings"] = dict(target["bindings"])
    merged["bindings"].update(draft["bindings"])
    merged["uses"] = list(target["uses"]) + draft["uses"]
    merged["files"] = [{"path": None, "payload_bytes": payload_total}]
    merged["provenance"] = dict(target.get("provenance") or {})
    for key, value in (draft.get("provenance") or {}).items():
        merged["provenance"][key] = value
    parse_directory(merged, entry_name=Path(out_path).name)

    data, reserve = serialize(merged)
    print(
        f"target payload {target_payload:,} | dspine region {draft_payload:,} in "
        f"{len(attached)} objects | json reserve {reserve:,} | "
        f"entry bytes {HEADER.size + reserve + payload_total:,}",
        flush=True,
    )

    with open(out_path, "wb") as out:
        out.write(HEADER.pack(MAGIC, reserve, _artifact_id(target_path)))
        out.write(data)
        copy_payload(target_path, target_start, out, 0, target_payload)
        gap = HEADER.size + reserve + new_base - out.tell()
        if gap < 0:
            raise SystemExit("overlapping payload layout")
        out.write(b"\0" * gap)
        copy_payload(draft_path, draft_start, out, 0, draft_payload)
        if out.tell() != HEADER.size + reserve + payload_total:
            raise SystemExit("output layout does not match directory")
        out.flush()
        os.fsync(out.fileno())
    if os.path.getsize(out_path) != HEADER.size + reserve + payload_total:
        raise SystemExit("output size mismatch")
    check, _ = read_directory(out_path)
    if len(check["objects"]) != len(merged["objects"]):
        raise SystemExit("output directory object count mismatch")
    print(f"wrote {out_path}")


def _artifact_id(path: str) -> bytes:
    with open(path, "rb") as handle:
        return HEADER.unpack(handle.read(HEADER.size))[2]


if __name__ == "__main__":
    main()
