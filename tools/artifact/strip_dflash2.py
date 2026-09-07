"""Strip the DFlash2 module from a Qwen3.8-27B ninfer artifact.

Removes every object whose name starts with ``dflash2/`` and writes the rest
verbatim (payloads copied byte-for-byte) under the same identity. Useful to
produce a MTP-only, no-patch artifact from an official nvfp4/groupwise image
that shipped the DFlash2 module.

The source artifact is opened read-only; output goes to a fresh file.

    python3 -m tools.artifact.strip_dflash2 \\
      --artifact models/qwen3_8_27b_nvfp4.ninfer \\
      --out out/qwen3_8_27b_nvfp4-mtp-only.ninfer
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

from tools.artifact.container import (
    Artifact,
    ArtifactIdentity,
    ArtifactWriter,
    ResourceObject,
    TensorObject,
)
from tools.convert.qwen3_6.common import inventory as family_inventory

COPY_CHUNK_BYTES = 64 << 20


def _copy_chunks(payload: memoryview) -> Iterator[memoryview]:
    for begin in range(0, len(payload), COPY_CHUNK_BYTES):
        yield payload[begin : begin + COPY_CHUNK_BYTES]


def strip(
    artifact_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    source_path = Path(artifact_path)
    output = Path(output_path)
    if output.resolve() == source_path.resolve():
        raise ValueError("refusing to strip onto itself; pass a distinct --out")

    with Artifact(source_path) as source:
        keep = [o for o in source.objects if not o.name.startswith("dflash2/")]
        removed = len(source.objects) - len(keep)
        if removed == 0:
            raise ValueError(f"{source_path}: no dflash2/* objects to strip")

        resources: dict[str, bytes] = {}
        specs: list[family_inventory.StoredObjectSpec] = []
        for obj in keep:
            if isinstance(obj, ResourceObject):
                resources[obj.name] = bytes(source.payload(obj))
                specs.append(family_inventory.ResourceSpec(obj.name))
            elif isinstance(obj, TensorObject):
                specs.append(
                    family_inventory.TensorSpec(obj.name, tuple(obj.shape), obj.format, obj.layout)
                )
            else:
                raise ValueError(f"unknown object kind for {obj.name}")

        plan = family_conversion_build_plan(tuple(specs), resources)
        output.parent.mkdir(parents=True, exist_ok=True)
        with ArtifactWriter(
            output,
            ArtifactIdentity(source.identity.model_id, source.identity.weights_id),
            plan.specs,
        ) as writer:
            total = len(plan.specs)
            for index, spec in enumerate(plan.specs, start=1):
                name = spec.name
                if isinstance(spec, family_inventory.ResourceSpec):
                    writer.write(name, resources[name])
                else:
                    writer.write(name, _copy_chunks(source.payload(name)))
                if index % 128 == 0 or index == total:
                    print(f"[{index}/{total}] objects written", flush=True)

    return {
        "source": str(source_path),
        "output": str(output),
        "identity": f"{source.identity.model_id}/{source.identity.weights_id}",
        "objects_kept": len(keep),
        "objects_removed": removed,
        "file_bytes": output.stat().st_size,
    }


def family_conversion_build_plan(specs, resources):
    from tools.convert.qwen3_6.common import conversion as family_conversion
    return family_conversion.build_object_plan(specs, resources)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    a = parser.parse_args()
    report = strip(a.artifact, a.out)
    print(f"wrote {report['output']} ({report['file_bytes']} bytes)")
    print(f"kept {report['objects_kept']}, removed {report['objects_removed']} dflash2 objects")


if __name__ == "__main__":
    main()
