#!/usr/bin/env python3
"""Extract the DSpine draft suffix of a fused NInfer v3 artifact.

    python3 scripts/extract_dspine_draft.py FULL.ninfer DRAFT_SUFFIX.ninfer

The result holds only the ``dspine/*`` objects (backbone, code bridge, and the
tower-specific TG/TSV refinement tables) plus the ``dspine`` component, with
offsets rebased to zero.  It is not loadable on its own: a v3 directory requires
a ``text`` component, so this file is a portable draft payload that
``scripts/fuse_dspine_draft.py`` mounts onto a compatible target tower.  Pure
byte copy: no requantization, the draft payload is preserved verbatim from the
validated full artifact, so the tables keep the tower they were computed for.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.artifact.framing import (  # noqa: E402
    HEADER,
    MAGIC,
    PAYLOAD_ALIGNMENT,
)
from tools.artifact.schema import _parse_object, validate_encoding  # noqa: E402

ALIGN = PAYLOAD_ALIGNMENT
OBJ_ALIGN = 256
CHUNK = 64 * 1024 * 1024
PREFIX = "dspine/"


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the DSpine draft suffix of a fused v3 artifact."
    )
    parser.add_argument("full", help="fused v3 artifact carrying a DSpine head")
    parser.add_argument("output", help="draft-suffix file to create")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    full_path, out_path = args.full, args.output
    if os.path.exists(out_path):
        raise SystemExit(f"refusing to overwrite {out_path}")
    full, full_start = read_directory(full_path)

    objects = [obj for obj in full["objects"] if obj["id"].startswith(PREFIX)]
    if not objects:
        raise SystemExit("source artifact carries no DSpine head")
    region_begin = min(obj["offset"] for obj in objects)
    region_end = max(obj["offset"] + obj["bytes"] for obj in objects)
    if region_begin % OBJ_ALIGN:
        raise SystemExit("dspine region is not object-aligned")
    ids = [obj["id"] for obj in objects]
    if len(set(ids)) != len(ids):
        raise SystemExit("dspine region contains a duplicate object id")
    if sorted(obj["offset"] for obj in objects) != [
        obj["offset"] for obj in objects
    ]:
        raise SystemExit("dspine objects are not in offset order")

    rebased = []
    for obj in objects:
        copy = dict(obj)
        copy["offset"] = obj["offset"] - region_begin
        validate_encoding(_parse_object(copy))
        rebased.append(copy)
    payload_bytes = region_end - region_begin

    bindings = {
        name: binding
        for name, binding in full["bindings"].items()
        if name.startswith(PREFIX)
    }
    uses = [
        use
        for use in full["uses"]
        if use.get("parameter", "").startswith(PREFIX)
        or use.get("input", "").startswith(PREFIX)
    ]
    if "dspine" not in full["components"]:
        raise SystemExit("source artifact has no dspine component")
    suffix = {
        "components": {"dspine": full["components"]["dspine"]},
        "objects": rebased,
        "bindings": bindings,
        "uses": uses,
        "files": [{"path": None, "payload_bytes": payload_bytes}],
        "metadata": {"name": os.path.basename(out_path)},
        "provenance": dict(full.get("provenance") or {}),
    }
    data, reserve = serialize(suffix)
    print(
        f"dspine region {payload_bytes:,} bytes in {len(rebased)} objects | "
        f"json reserve {reserve:,}",
        flush=True,
    )

    with open(full_path, "rb") as source, open(out_path, "wb") as out:
        out.write(HEADER.pack(MAGIC, reserve, _artifact_id(full_path)))
        out.write(data)
        source.seek(full_start + region_begin)
        remaining = payload_bytes
        while remaining:
            block = source.read(min(CHUNK, remaining))
            if not block:
                raise SystemExit("dspine payload truncated")
            out.write(block)
            remaining -= len(block)
        out.flush()
        os.fsync(out.fileno())
    if os.path.getsize(out_path) != HEADER.size + reserve + payload_bytes:
        raise SystemExit("output size mismatch")
    print(f"wrote {out_path}")


def _artifact_id(path: str) -> bytes:
    with open(path, "rb") as handle:
        return HEADER.unpack(handle.read(HEADER.size))[2]


if __name__ == "__main__":
    main()
