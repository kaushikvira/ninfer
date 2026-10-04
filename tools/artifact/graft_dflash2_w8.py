"""Graft the z-lab DFlash2 module (W8G32_F16S / BF16) onto cometkim nvfp4full v1.

Same idea as the gpillon fork's graft (copy v1 payloads verbatim, append the
DFlash2 module objects), but encodes the module in the W8/BF16 format OUR
upstream master expects (bind_dflash2: attention_conv/*, candidate_selector/*,
W8G32_F16S matrices, BF16 norms/codebooks). MTP objects are KEPT — with
--spec dflash2 they are validate-only and never materialized on device.

The source artifact is opened read-only and only READ; all output goes to a
fresh file at --out. The source file is never modified.

Canonical invocation::

    python3 -m tools.artifact.graft_dflash2_w8 \\
      --artifact models/qwen3_8_27b_nvfp4full.ninfer \\
      --dflash2-model /path/to/Qwen3.8-27B-DFlash2 \\
      --out out/qwen3_8_27b_nvfp4full-dflash2.ninfer

``--device cpu`` is supported (the module is a few matrices; W8 encoder is
device-parameterized).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Iterator

import torch

from tools.artifact.container import (
    Artifact,
    ArtifactIdentity,
    ArtifactObject,
    ArtifactWriter,
    ResourceObject,
    TensorObject,
)
from tools.convert.common.quantize import pick_device
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_6.common import inventory as family_inventory
from tools.convert.qwen3_6.common import recipe as family_recipe
from tools.convert.qwen3_8_27b import dflash2_inventory as d2inv
from tools.convert.qwen3_8_27b import dflash2_recipe as d2rec

MODEL_ID = "qwen3.8-27b"
WEIGHTS_ID = "nvfp4full"
COPY_CHUNK_BYTES = 64 << 20


def _validate_source(artifact: Artifact) -> None:
    """Source must be the cometkim nvfp4full v1 image WITHOUT a DFlash2 module."""
    if (
        artifact.identity.model_id != MODEL_ID
        or artifact.identity.weights_id != WEIGHTS_ID
    ):
        raise ValueError(
            f"{artifact.path}: identity {artifact.identity.model_id}/"
            f"{artifact.identity.weights_id} is not {MODEL_ID}/{WEIGHTS_ID}"
        )
    for spec in d2inv.DFLASH2_TENSOR_SPECS:
        if spec.name in artifact._index:
            raise ValueError(f"{artifact.path}: already carries DFlash2 module objects")


def _copy_chunks(payload: memoryview) -> Iterator[memoryview]:
    for begin in range(0, len(payload), COPY_CHUNK_BYTES):
        yield payload[begin : begin + COPY_CHUNK_BYTES]


def graft(
    artifact_path: str | Path,
    dflash2_dir: str | Path,
    output_path: str | Path,
    device: str | None = None,
) -> dict[str, object]:
    started = time.perf_counter()
    source_path = Path(artifact_path)
    dflash2_dir = Path(dflash2_dir)
    output = Path(output_path)
    if output.resolve() == source_path.resolve():
        raise ValueError("refusing to graft an artifact onto itself; pass a distinct --out")

    resolved_device = pick_device(device if device is not None else "cpu")
    module_specs: tuple = d2inv.DFLASH2_TENSOR_SPECS
    module_names = frozenset(spec.name for spec in module_specs)

    # Sanity: the source model config must be the supported DFlash2 config.
    d2rec.validate_config(
        json.loads((dflash2_dir / "config.json").read_text(encoding="utf-8"))
    )
    d2rec.preflight_sources(dflash2_dir)

    with Artifact(source_path) as source:
        _validate_source(source)

        # Build canonical spec list: source objects as-is + module tail.
        # Resources are carried verbatim (their byte length is what the plan uses).
        resources: dict[str, bytes] = {}
        combined: list[family_inventory.StoredObjectSpec] = []
        for obj in source.objects:
            if isinstance(obj, ResourceObject):
                resources[obj.name] = bytes(source.payload(obj))
                combined.append(family_inventory.ResourceSpec(obj.name))
            elif isinstance(obj, TensorObject):
                combined.append(
                    family_inventory.TensorSpec(
                        obj.name, tuple(obj.shape), obj.format, obj.layout
                    )
                )
            else:
                raise ValueError(f"unknown object kind for {obj.name}")
        combined.extend(module_specs)

        plan = family_conversion.build_object_plan(tuple(combined), resources)

        with ShardReader.from_file(dflash2_dir / "model.safetensors") as reader:
            output.parent.mkdir(parents=True, exist_ok=True)
            with ArtifactWriter(
                output,
                ArtifactIdentity(MODEL_ID, WEIGHTS_ID),
                plan.specs,
            ) as writer:
                total = len(plan.specs)
                for index, spec in enumerate(plan.specs, start=1):
                    name = spec.name
                    if name in module_names:
                        tensor = d2rec.materialize_tensor(name, reader)
                        payload = family_conversion.encode_tensor_payload(
                            tensor, spec, resolved_device
                        )
                        writer.write(name, payload)
                        del payload, tensor
                    elif isinstance(spec, family_inventory.ResourceSpec):
                        writer.write(name, resources[name])
                    else:
                        writer.write(name, _copy_chunks(source.payload(name)))
                    if index % 128 == 0 or index == total:
                        print(f"[{index}/{total}] objects written", flush=True)

    elapsed = time.perf_counter() - started
    report: dict[str, object] = {
        "recipe_id": "qwen3_8_27b_nvfp4full-dflash2-w8",
        "grafted_from": {
            "path": str(source_path),
            "bytes": source_path.stat().st_size,
        },
        "source": {"dflash2": {"model_path": str(dflash2_dir)}},
        "artifact": {"path": str(output), "bytes": output.stat().st_size},
        "objects": {"count": len(plan.specs), "grafted": len(module_names)},
        "elapsed_seconds": elapsed,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--dflash2-model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default=None)
    arguments = parser.parse_args()

    report = graft(
        arguments.artifact, arguments.dflash2_model, arguments.out, arguments.device
    )
    report_path = arguments.out.with_suffix(arguments.out.suffix + ".graft.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {report['artifact']['path']} ({report['artifact']['bytes']} bytes)")
    print(f"grafted {report['objects']['grafted']} DFlash2 objects")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
