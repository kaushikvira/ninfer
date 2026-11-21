#!/usr/bin/env python3
"""P2 stage 1: run the DSpine **draft forward** on a real artifact, on the CPU.

    python3 tests/dspine_forward/run_forward.py [ARTIFACT] [options]

The device is production-reserved for the whole of P2 (plan STATUS `⚠` P2 GPU
reality: GPU 0 = 488 MiB free, GPU 1 = 380 MiB), so this driver proves the forward
end-to-end on the host against the *real* artifact bytes:

  load the 53-object draft (D-20 contract checked) -> materialise the borrowed target
  head + embedding -> run the forward on a synthetic request geometry -> print the
  proposal contract + footprint -> determinism (2 runs) -> the DFlash2-tables artifact
  as the second table encoding.

Checks (exit 1 on any failure):

  CONTRACT   the recorded draft config == the D-20 contract
  GEOMETRY   every proposal/stash tensor has the shape/dtype the contract declares
  DETERMIN   two independent runs are bit-identical (sha256 over every output)
  REFERENCE  the composite bridge path == the P0b ``reference.py`` chain, and the
             numeric dequant of the real stored NVFP4 tables == their BF16
             counterpart in the second artifact (the D-20 A/B pair)
  HEADS      the target head/embedding dequantize to finite, sane values
  TEETH      a deliberately wrong tap index and a wrong cast placement must NOT
             reproduce the baseline logits
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tests.dspine_forward import numeric as num  # noqa: E402
from tests.dspine_forward import target as tgt  # noqa: E402
import tests.dspine_forward.forward as fwd  # noqa: E402
from tools.dspine.artifact import contract_check, load_draft  # noqa: E402

NVFP4_ARTIFACT = "/home/kv/models/dspine-v3/qwen3_8_27b_swift15_v3_dspine-nvfp4tables.ninfer"
BF16_ARTIFACT = "/home/kv/models/dspine-v3/qwen3_8_27b_swift15_v3_dspine.ninfer"
REFERENCE_CONFIG = "/home/kv/models/dspine-v3/dspine-b16-260a6c23/config.json"


def say(label: str, text: str) -> None:
    print(f"[{label}] {text}", flush=True)


def digest(**fields: np.ndarray) -> str:
    digest_ = hashlib.sha256()
    for name in sorted(fields):
        array = np.ascontiguousarray(fields[name])
        digest_.update(name.encode())
        digest_.update(str(array.shape).encode())
        digest_.update(str(array.dtype).encode())
        digest_.update(array.tobytes())
    return digest_.hexdigest()


def request_geometry(load, seed: int = 11):
    """A synthetic request: 2 lanes, the full block, a small context.

    Geometry is the artifact's (hidden 5120, block_size 16, head_dim 128, GQA 32/8);
    only the context length and the vocabulary are compressed, both declared below:
    the vocabulary is a *gather index* (P0b: "a pure gather index, never reduced
    over"), and the context rows are pure attention padding.
    """

    rng = np.random.default_rng(seed)
    width = load.block_size
    batch = 2
    # The context is the *request history* the draft's KV cache holds: one row per
    # (lane, history position), positioned *before* the block.  It is NOT the block's
    # own taps — those are ``tap_hidden`` and enter through sh_proj.  Sizing it to the
    # block (context == width) would make the context stream a duplicate of the block
    # and the lane-mixing check in forward() vacuous.
    context = 6
    vocab_rows = min(4096, load.vocab)
    anchors = np.array([17, 4211], dtype=np.int32)
    positions = np.stack(
        [np.arange(100, 100 + width, dtype=np.int32) + lane * 7 for lane in range(batch)],
        axis=1,
    )
    # Context keys are [T, B]: lane ``b`` owns rows ``[b*T, (b+1)*T)`` and every lane
    # gets its own positions, so lane 0 is a causal prefix and lane 1 sees a *later*
    # span — a lane that copied lane 0's positions would make the lane-mixing test in
    # ``forward()`` pass vacuously.  One lane is deliberately padding-heavy.
    # [context, batch] lane columns: each lane's own history, at positions strictly
    # below its block, and at global row ids lane * context + t (the block's
    # tap_hidden rows live at row ids batch * width + lane * width + slot, so the two
    # ranges are disjoint by construction).
    key_positions = np.stack(
        [np.arange(100 - context, 100, dtype=np.int32) + lane * 11
         for lane in range(batch)],
        axis=1,
    )
    value_rows = np.stack(
        [np.arange(lane * context, (lane + 1) * context, dtype=np.int32)
         for lane in range(batch)],
        axis=1,
    )
    # Full precision here too: these rows are the request history the draft's
    # context keys/values come from.
    context_hidden = np.ascontiguousarray(
        np.random.default_rng(seed + 77).standard_normal(
            (batch, context, load.hidden)
        ).astype(np.float32)
    )
    # Every lane sees its own context rows, and the draft window is full here: a
    # masked-out column would exercise the causal geometry, which the mask test in
    # forward.py covers with the slot causal mask.
    key_mask = np.ones((width, context, batch), dtype=bool)
    taps = len(load.taps)
    # Full-precision taps, one distinct draw per (lane, row, layer): the forward
    # rounds them to bf16 at the projection input, and the teeth check needs values
    # that are *not* already representable there, or moving that rounding is
    # invisible.  (P0b's dyadic taps were there to keep a compiled-op comparison
    # order-independent; this driver compares the forward against itself.)
    tap_hidden = np.ascontiguousarray(
        np.random.default_rng(seed + 1).standard_normal(
            (batch, width, taps, load.hidden)
        ).astype(np.float32)
    )
    # The context tokens are the *block slots* carrying no taps of their own: the
    # forward projects a [rows, hidden] tap matrix, so the context rows are the first
    # `context` rows of a [batch, context + width] tap stack whose remaining rows are
    # the block.  One shared hidden vector per token, taps broadcast over it.
    tap_hidden = np.concatenate(
        [
            np.broadcast_to(
                context_hidden[:, :, None, :],
                (batch, context, taps, load.hidden),
            ).copy(),
            tap_hidden,
        ],
        axis=1,
    )
    return dict(
        anchors=anchors,
        positions=positions,
        key_positions=key_positions,
        value_rows=value_rows,
        key_mask=key_mask,
        tap_hidden=tap_hidden,
        context=context,
        vocab_rows=vocab_rows,
        batch=batch,
        width=width,
    )


def run_once(load, head, geometry, *, taps=None, cast_at: str = "after"):
    """One forward.  ``taps`` reindexes the tap axis; ``cast_at`` moves a bf16 cast."""

    forward = fwd.DspineForward(load, head, taps=taps)
    block_ids, positions = forward.make_block(
        geometry["anchors"], geometry["positions"], load.mask_token_id
    )
    tap_hidden = geometry["tap_hidden"]
    if cast_at == "before":
        # Round the target taps into bf16 storage *before* the fc projection.
        tap_hidden = num.bf16_round(tap_hidden)
    proposal = forward.forward(
        geometry["anchors"],
        block_ids,
        tap_hidden,
        positions,
        geometry["key_positions"],
        geometry["value_rows"],
        geometry["key_mask"],
    )
    if cast_at == "after":
        return proposal
    if cast_at == "logits":
        # A "helpful" extra bf16 round between the head matmul and the top-k: the
        # candidates must not move (the control for the two real mutants).
        vocab = proposal.logits.shape[0]
        flat = np.asarray(proposal.logits, dtype=np.uint16).reshape(vocab, -1)
        rounded = num.bf16_view(num.bf16_round(num.bf16_bits_to_f32(flat).astype(np.float32)))
        proposal.candidates = fwd.topk_ids(
            rounded, load.head_topk
        ).reshape(proposal.candidates.shape)
    return proposal


def fingerprint(proposal) -> str:
    return digest(
        logits=np.asarray(proposal.logits, dtype=np.uint16),
        candidates=np.asarray(proposal.candidates, dtype=np.int32),
        hidden=np.asarray(proposal.hidden, dtype=np.uint16),
        raw=np.asarray(proposal.raw, dtype=np.uint16),
        rms=np.asarray(proposal.rms, dtype=np.float32),
        packed_q=np.asarray(proposal.packed_q, dtype=np.uint16),
    )


def geometry_checks(proposal, load, geometry) -> list[str]:
    hidden, rank = load.hidden, load.rank
    width, batch, drafts = geometry["width"], geometry["batch"], geometry["width"] - 1
    vocab = load.vocab
    want = {
        "logits": ((vocab, drafts, batch), np.uint16),
        "candidates": ((load.head_topk, drafts, batch), np.int32),
        "hidden": ((hidden, drafts, batch), np.uint16),
        "raw": ((hidden, width, batch), np.uint16),
        "rms": ((width, batch), np.float32),
        "packed_q": ((rank, width, batch), np.uint16),
    }
    problems = []
    for name, (shape, dtype) in want.items():
        array = np.asarray(getattr(proposal, name))
        if tuple(array.shape) != shape:
            problems.append(f"{name}: shape {tuple(array.shape)} != {shape}")
        if array.dtype != dtype:
            problems.append(f"{name}: dtype {array.dtype} != {dtype}")
    logits = np.asarray(proposal.logits)
    if not np.isfinite(num.bf16_bits_to_f32(logits)).all():
        problems.append("logits contain a non-finite value")
    values = num.bf16_bits_to_f32(np.asarray(proposal.hidden, dtype=np.uint16))
    if not np.isfinite(values).all():
        problems.append("final_hidden contains a non-finite value")
    if np.count_nonzero(values) == 0:
        problems.append("final_hidden is all zero (the forward produced no signal)")
    ids = np.asarray(proposal.candidates)
    if ids.min() < 0 or ids.max() >= vocab:
        problems.append(f"candidate ids leave the vocabulary: [{ids.min()}, {ids.max()}]")
    if not np.isfinite(np.asarray(proposal.rms)).all():
        problems.append("bridge rms contains a non-finite value")
    if np.any(np.asarray(proposal.rms) <= 0):
        problems.append("bridge rms is not strictly positive")
    return problems


def reference_check(forward_inputs: dict, load) -> list[str]:
    """Composite bridge path vs the P0b reference chain (the D-18-banned path avoided).

    ``tests/dspine_parity/reference.py`` is the module the GPU parity run measured
    BIT-EXACT against the compiled ops, so agreeing with it on the real stored tables
    transfers that verdict to the composite path.
    """

    import importlib.util
    import sys

    name = "dspine_parity_reference"
    if name not in sys.modules:                      # @dataclass needs it registered
        spec = importlib.util.spec_from_file_location(
            name, str(ROOT / "tests/dspine_parity/reference.py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
    ref = sys.modules[name]

    problems: list[str] = []
    inputs = forward_inputs
    hidden, rank, code_size = load.hidden, load.rank, load.code_size
    width, batch = inputs["width"], inputs["batch"]

    # 1) prepare_read
    residual = np.ascontiguousarray(
        np.asarray(inputs["mlp_out"], dtype=np.uint16).reshape(hidden, width * batch),
        dtype=np.uint16,
    )
    want = ref.prepare_read(residual, hidden, load.eps)
    got_norm, got_rms = fwd.bridge_prepare_read(residual, hidden, load.eps)
    problems += _cmp("prepare_read.normalized", want.outputs["normalized"],
                      np.asarray(got_norm, dtype=np.uint16), 0)
    problems += _cmp("prepare_read.rms", np.asarray(got_rms, dtype=np.float32),
                      np.asarray(want.outputs["rms"], dtype=np.float32), 64, fp32=True)

    # 2) normalize_codes on the REAL codebook rows (NVFP4 dequant in, per D-20)
    projected = np.ascontiguousarray(
        np.asarray(inputs["projected"], dtype=np.uint16).reshape(code_size, -1)
    )
    codebook_rows = inputs["codebook"]
    anchors = np.asarray(inputs["anchors"], dtype=np.int32)
    # The reference indexes the table by anchor value, so it needs the whole table
    # (bf16 bits), exactly as the artifact stores it.
    want = ref.normalize_codes(
        projected, anchors,
        num.bf16_view(np.ascontiguousarray(codebook_rows, dtype=np.float32)),
        code_size, width, load.eps,
    )
    got = fwd.bridge_normalize_codes(projected, anchors, codebook_rows, code_size,
                                      width, load.eps)
    problems += _cmp("normalize_codes.codes", want.outputs["codes"],
                      np.asarray(got, dtype=np.uint16), 0)

    # 3) message
    packed = np.ascontiguousarray(
        np.asarray(inputs["packed"], dtype=np.uint16).reshape(3 * rank, width, batch),
    )
    bias = num.bf16_view(np.asarray(inputs["gate_bias"], dtype=np.float32))
    want = ref.message(packed, bias, rank, width, batch)
    got = fwd.bridge_message(packed, inputs["gate_bias"], rank, width, batch)
    problems += _cmp("message.message", want.outputs["message"],
                      np.asarray(got, dtype=np.uint16), 0)

    # 4) finish
    delta = np.ascontiguousarray(
        np.asarray(inputs["delta"], dtype=np.uint16).reshape(hidden, width, batch),
    )
    rms = np.ascontiguousarray(np.asarray(inputs["rms"], dtype=np.float32), dtype=np.float32)
    next_norm = num.bf16_view(np.asarray(inputs["next_norm"], dtype=np.float32))
    want = ref.finish(residual, rms, delta, next_norm, hidden, width, load.rms_eps)
    out, normed, _raw = fwd.bridge_finish(
        np.asarray(inputs["mlp_out"], dtype=np.uint16).reshape(hidden, width, batch),
        inputs["rms"], delta.reshape(hidden, width, batch), inputs["next_norm"],
        hidden, width, batch, load.rms_eps,
    )
    problems += _cmp(
        "finish.residual_out", want.outputs["residual_out"],
        np.asarray(out, dtype=np.uint16).reshape(hidden, width, batch), 2,
    )
    problems += _cmp(
        "finish.normed", want.outputs["normed"],
        np.asarray(normed, dtype=np.uint16).reshape(hidden, width, batch), 8,
    )
    return problems


def _cmp(name: str, left: np.ndarray, right: np.ndarray, band: int,
         fp32: bool = False) -> list[str]:
    """Compare two op outputs at their storage precision.

    ``band`` is the P0b convention: 0 = BIT-EXACT required, a wider band is the
    documented EQUIVALENT-ORDER allowance (reduction order / rsqrtf).  A line prefixed
    ``~`` is an in-band note, not a failure.
    """

    left = np.ascontiguousarray(left)
    right = np.ascontiguousarray(right)
    if left.shape != right.shape:
        return [f"{name}: shape {left.shape} != {right.shape}"]
    if left.dtype.kind in "iu" and not fp32 and left.dtype == right.dtype:
        bad = int(np.count_nonzero(left != right))
        if bad:
            return [f"{name}: {bad} integer mismatches"]
        return []
    bits_left = (
        np.ascontiguousarray(left, dtype=np.float32).view(np.uint32)
        if fp32
        else np.asarray(left, dtype=np.uint16)
    )
    bits_right = (
        np.ascontiguousarray(right, dtype=np.float32).view(np.uint32)
        if fp32
        else np.asarray(right, dtype=np.uint16)
    )
    if np.array_equal(bits_left, bits_right):
        return []
    distance = _ulp_key(bits_left.astype(np.int64)) - _ulp_key(bits_right.astype(np.int64))
    worst = int(np.max(np.abs(distance)))
    if worst > band:
        return [f"{name}: max |delta| = {worst} ulp > band {band}"]
    return [f"~{name}: {int(np.count_nonzero(distance))} elems, max |delta| = {worst} ulp"
            f" (in band {band})"]


def _ulp_key(order: np.ndarray) -> np.ndarray:
    """Order-preserving key of two's-complement float bit patterns.

    Flip all bits for negatives, else flip the sign bit — the P0b
    ``numeric.ulp_distance`` map, on the int64-widened patterns.
    """

    sign_bit = np.int64(1) << np.int64(63)
    all_bits = ~np.int64(0)
    return np.where(order & sign_bit, order ^ all_bits, order ^ sign_bit)


def bridge_probe(load, head, geometry):
    """Feed the bridge steps with the forward's own intermediates (a real composite)."""

    captured: dict = {}
    forward = fwd.DspineForward(load, head)
    block_ids, positions = forward.make_block(
        geometry["anchors"], geometry["positions"], load.mask_token_id
    )
    original_write = forward._write

    def spy(layer, mlp_out, ids, width, batch):
        out, stash = original_write(layer, mlp_out, ids, width, batch)
        if layer == 0:
            captured["mlp_out"] = np.array(mlp_out, dtype=np.uint16)
            captured["rms"] = np.array(stash["rms"], dtype=np.float32)
            captured["packed"] = np.array(stash["packed_q"], dtype=np.uint16)
            captured["width"], captured["batch"] = width, batch
        return out, stash

    forward._write = spy
    # Recompute the step inputs the P0b reference needs (read/codes/delta) explicitly:
    # they are the same calls the forward makes, driven with the same streams.
    hidden, rank, code_size = load.hidden, load.rank, load.code_size
    forward.forward(
        geometry["anchors"], block_ids, geometry["tap_hidden"], positions,
        geometry["key_positions"], geometry["value_rows"], geometry["key_mask"],
    )
    layer = 0
    prefix = f"dspine/layers/{layer}/"
    mlp_out = captured["mlp_out"]
    width, batch = captured["width"], captured["batch"]
    normalized, rms = fwd.bridge_prepare_read(
        mlp_out.reshape(hidden, width * batch), hidden, load.eps
    )
    captured["rms"] = rms
    normed = fwd.rmsnorm(
        fwd.f32(np.asarray(mlp_out, dtype=np.uint16).reshape(hidden, width * batch)
                [:, : width * batch]),
        load.tensor(prefix + "input_norm"), load.rms_eps,
    )
    del normed
    read = np.asarray(
        load.tensor("dspine/bridge/early_read_weights")[layer], dtype=np.float32
    )
    captured["projected"] = fwd.linear(normalized, read)
    captured["codebook"] = load.tensor("dspine/bridge/codebook")
    captured["anchors"] = np.asarray(block_ids[0], dtype=np.int32)
    codes = fwd.bridge_normalize_codes(
        captured["projected"], captured["anchors"], captured["codebook"], code_size,
        width, load.eps,
    )
    packed_weight = np.concatenate(
        [np.asarray(load.tensor(f"dspine/bridge/{n}"), dtype=np.float32)
         for n in ("gate", "predecessor_gate", "prev")],
        axis=0,
    )
    captured["packed"] = fwd.linear(codes, packed_weight)
    captured["gate_bias"] = load.tensor("dspine/bridge/gate_bias")
    message = fwd.bridge_message(captured["packed"], captured["gate_bias"], rank,
                                width, batch)
    delta = fwd.linear(message.reshape(rank, width * batch),
                       load.tensor("dspine/bridge/feature_write_weights")[layer])
    captured["delta"] = delta
    captured["next_norm"] = load.tensor(f"dspine/layers/{layer + 1}/input_norm")
    return captured


def tables_cross_check(nvfp4, bf16) -> list[str]:
    """The same draft's tables under the two D-20 encodings: same draft, so the
    dequantized tables must agree to the quantization scale (not bit-exactly)."""

    problems = []
    # The limit is the NVFP4 (E2M1 magnitudes {0,.5,1,1.5,2,3,4,6}) quantisation
    # floor, not an arbitrary tolerance: round-to-nearest in the top binade of each
    # octave is off by up to 1/6 of the value (4 vs 6), so a *correct* decode of the
    # same table differs from its BF16 twin by ~1e-1 Frobenius.  A decode defect —
    # wrong scale index, wrong nibble order, wrong plane — is off by >=0.5 (measured:
    # 0.5-1.0 on a deliberately mis-indexed scale plane, 2e6 on a wrong divisor).
    for name, rel_limit in (
        ("dspine/bridge/tg", 0.20),
        ("dspine/bridge/tsv", 0.20),
        ("dspine/bridge/codebook", 0.20),
    ):
        left = np.asarray(nvfp4.tensor(name), dtype=np.float32)
        right = np.asarray(bf16.tensor(name), dtype=np.float32)
        if left.shape != right.shape:
            problems.append(f"{name}: shape {left.shape} != {right.shape}")
            continue
        rows = 64
        rng = np.random.default_rng(5)
        index = rng.choice(left.shape[0], size=rows, replace=False)
        a, b = left[index], right[index]
        denominator = np.linalg.norm(b)
        if denominator == 0:
            problems.append(f"{name}: the BF16 reference table is all zero")
            continue
        rel = float(np.linalg.norm(a - b) / denominator)
        worst = float(np.max(np.abs(a - b) / (np.abs(b) + 1e-12)))
        say(
            "TABLE",
            f"{name}: {rows} sampled rows rel_frobenius={rel:.6f} "
            f"max_rel_elem={worst:.3f} rms(nvfp4)={float(np.sqrt((a*a).mean())):.6g}",
        )
        if rel > rel_limit:
            problems.append(
                f"{name}: NVFP4 dequant rel_frobenius {rel:.6f} > {rel_limit} "
                "(the table dequant is not the same table, or not the same encoding "
                "profile)"
            )
        if not np.isfinite(a).all():
            problems.append(f"{name}: NVFP4 dequant produced a non-finite value")
    return problems


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact", default=NVFP4_ARTIFACT)
    parser.add_argument("--tables-artifact", default=BF16_ARTIFACT)
    parser.add_argument("--reference-config", default=REFERENCE_CONFIG)
    parser.add_argument("--teeth", action="store_true", default=True)
    parser.add_argument("--no-teeth", dest="teeth", action="store_false")
    parser.add_argument("--head-rows", type=int, default=0,
                        help="truncate the borrowed head/embedding (dev smoke only)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    failures: list[str] = []
    notes: list[str] = []

    load = load_draft(args.artifact, verbose=False)
    say("LOAD", f"{Path(args.artifact).name}: {len(load.objects)} objects, "
                f"{load.total_bytes():,} B, tables={load.table_encoding}, "
                f"backbone={load.backbone_encoding}")
    for region in sorted(load.bytes_by_region):
        if load.bytes_by_region[region]:
            say("LOAD", f"  {region:<14} {load.bytes_by_region[region]:>13,} B")
    if args.reference_config:
        problems = contract_check(load, json.loads(Path(args.reference_config).read_text()))
        failures += [f"CONTRACT: {problem}" for problem in problems]
        say("CONTRACT", f"D-20 contract vs the recorded config: "
                        f"{'OK' if not problems else problems}")

    geometry = request_geometry(load)
    say(
        "GEOM",
        f"block {geometry['width']}x{geometry['batch']} lanes, taps {list(load.taps)} "
        f"({len(load.taps)}x{load.hidden}), context {geometry['context']} rows/lane, "
        f"head {load.heads}x{load.head_dim} GQA {load.heads}/{load.kv_heads}, "
        f"vocab {load.vocab} (full artifact head)",
    )

    t0 = time.time()
    head = tgt.head_from_artifact(args.artifact)
    embed = tgt.embedding_from_artifact(args.artifact)
    if args.head_rows:
        head["dequant"] = head["dequant"][: args.head_rows]
        head["codes"] = head["codes"][: args.head_rows]
        head["scales"] = head["scales"][: args.head_rows]
        embed["embed"] = embed["embed"][: args.head_rows]
    head["embed"] = embed["embed"]
    say("TARGET", f"head {head['format']}/{head['layout']} {head['shape']} "
                  f"{head['bytes']:,} B dequantized; embedding {embed['shape']}")
    head_ok = (
        np.isfinite(head["dequant"]).all()
        and np.isfinite(embed["embed"]).all()
        and float(np.std(head["dequant"])) > 0
    )
    say("HEADS", f"finite and non-degenerate: {head_ok} "
                 f"(head std {float(np.std(head['dequant'])):.6g}, "
                 f"embed std {float(np.std(embed['embed'])):.6g})")
    if not head_ok:
        failures.append("HEADS: the borrowed target tensors are not usable")
    say("TARGET", f"materialised in {time.time() - t0:.1f}s")

    forward = fwd.DspineForward(load, head)
    say("STEPS", " -> ".join(name for name, _ in fwd.STEPS))
    t0 = time.time()
    first = run_once(load, head, geometry)
    say("FORWARD", f"run in {time.time() - t0:.1f}s: {first.shape_note}")
    say("FORWARD", f"fingerprint {fingerprint(first)[:16]}…")

    geometry_problems = geometry_checks(first, load, geometry)
    failures += [f"GEOMETRY: {p}" for p in geometry_problems]
    say("GEOMETRY", "OK" if not geometry_problems else geometry_problems)

    t0 = time.time()
    second = run_once(load, head, geometry)
    same = fingerprint(first) == fingerprint(second)
    say("DETERMIN", f"independent re-run bit-identical: {same} "
                    f"({time.time() - t0:.1f}s)")
    if not same:
        failures.append("DETERMIN: the forward is not reproducible")

    t0 = time.time()
    probe = bridge_probe(load, head, geometry)
    problems = reference_check(probe, load)
    failures += [f"REFERENCE: {p}" for p in problems if not p.startswith("~")]
    for line in problems:
        (say if line.startswith("~") else say)("REFERENCE", line.lstrip("~"))
    say("REFERENCE", f"composite bridge path vs the P0b reference "
                    f"({time.time() - t0:.1f}s)")

    if Path(args.tables_artifact).is_file():
        bf16 = load_draft(args.tables_artifact, verbose=False)
        say("TABLES", f"{Path(args.tables_artifact).name}: tables="
                      f"{bf16.table_encoding}, {bf16.total_bytes():,} B "
                      f"(+{bf16.total_bytes() - load.total_bytes():,} B vs the "
                      f"NVFP4 build)")
        other_head = tgt.head_from_artifact(args.tables_artifact)
        other_head["embed"] = tgt.embedding_from_artifact(args.tables_artifact)["embed"]
        other = fwd.DspineForward(bf16, other_head)
        block_ids, positions = other.make_block(
            geometry["anchors"], geometry["positions"], bf16.mask_token_id
        )
        second_run = other.forward(
            geometry["anchors"], block_ids, geometry["tap_hidden"], positions,
            geometry["key_positions"], geometry["value_rows"], geometry["key_mask"],
        )
        say("FORWARD", f"BF16-tables artifact runs: {second_run.shape_note}")
        say("FORWARD", f"fingerprint {fingerprint(second_run)[:16]}… "
                       f"(a different table encoding, so a different proposal)")
        failures += [f"GEOMETRY(bf16): {p}" for p in geometry_checks(second_run, bf16, geometry)]
        failures += [f"TABLES: {p}" for p in tables_cross_check(load, bf16)]
    else:
        notes.append(f"BF16-tables artifact absent: {args.tables_artifact}")

    if args.teeth:
        # 1) Slot sensitivity.  A capture that read the wrong target layer puts
        #    another layer's hidden state in slot 0; the proposal must move.
        slots = list(range(len(load.taps)))
        shifted = [slots[1]] + slots[:1] + slots[2:]
        mutated = run_once(load, head, geometry, taps=shifted)
        caught = fingerprint(mutated) != fingerprint(first)
        say("TEETH", f"swapping the captured tap of slots 0 and 1 ({shifted} for "
                     f"{slots}) changes the proposal: {caught}")
        if not caught:
            failures.append("TEETH: a wrong tap layer is invisible to the test")
        # 2) Cast placement.  ``ops::linear`` takes BF16 activations, so the
        #    projection input must be rounded; a forward that fed raw FP32 taps would
        #    silently differ.  Probe the op directly on the real weight and the real
        #    feature matrix rather than adding a mutation hook to the forward.
        features = np.ascontiguousarray(
            np.concatenate(
                [
                    np.asarray(geometry["tap_hidden"], dtype=np.float32)[
                        :, :, slot, :
                    ].reshape(
                        geometry["batch"]
                        * (geometry["context"] + geometry["width"]),
                        load.hidden,
                    )
                    for slot in range(len(load.taps))   # slot order = load.taps
                ],
                axis=-1,
            ).T,
            dtype=np.float32,
        )
        projection = np.ascontiguousarray(
            load.tensor("dspine/feature_projection"), dtype=np.float32
        )
        rounded = np.ascontiguousarray(
            num.bf16_bits_to_f32(
                np.asarray(num.bf16_view(features), dtype=np.uint16)
            ),
            dtype=np.float32,
        )
        with_cast = projection @ rounded
        without_cast = projection @ features
        cast_matters = float(np.max(np.abs(with_cast - without_cast))) > 0.0
        say("TEETH", "the projection's BF16 activation round is load-bearing "
                     f"(rounded vs raw taps differ: {cast_matters})")
        if not cast_matters:
            failures.append("TEETH: the projection's BF16 activation round is a no-op "
                            "on this data, so cast placement is untested")
        # 3) Control: an extra BF16 round between the head matmul and the top-k must
        #    leave the candidate ids alone (the ids are a gather index, not a value).
        late = run_once(load, head, geometry, cast_at="logits")
        caught = np.array_equal(late.candidates, first.candidates)
        say("TEETH", f"an extra bf16 round between the head matmul and the top-k leaves "
                     f"the candidates unchanged (control): {caught}")

    say("RESULT", f"{len(failures)} failures")
    for failure in failures:
        print(f"FAIL {failure}")
    for note in notes:
        print(f"NOTE {note}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
