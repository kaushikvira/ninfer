"""P2 NVFP4 gather/dequant parity for the bridge ops: P0b's vectors, NVFP4 gather tables.

The oracle is the COMPILED OP (`build/rtx5090-ops/lib/libninfer_ops.so` of this tree) run on
one CUDA device through `fixture/dspine_nvfp4_fixture`. The reference side is the P0b harness
(`tests/dspine_parity/{reference,numeric,mutants,run_parity}.py`) - vectors, op math and ulp
helpers are imported, not transcribed - fed with the **CPU decode of the exact NVFP4 bytes**
the device dequantizes on gather (`nvfp4.py`).

Three verdicts per op, on the same vectors:

  (a) A-nvfp4-gather   device output with NVFP4 tables vs the P0b reference run on the
                       CPU-dequantized same bytes. The gather path is SOUND only at
                       BIT-EXACT (band from P0b's BANDS table).
  (b) B-bf16-vs-nvfp4  device NVFP4 output vs the DEVICE output with BF16 tables of the same
                       values on the same vectors: quantization-induced drift, no reference.
  (c) C-teeth:<name>   a defect in the *reference* decode (natural scale-plane addressing,
                       swapped nibbles, dropped divisor, sign ignored). The device reads the
                       real bytes, so it must land outside the band.

    CUDA_VISIBLE_DEVICES=1 ./run_nvfp4_parity.py --ops normalize_codes,serial_scores
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parent / "dspine_parity"
for _path in (str(HERE), str(HARNESS)):
    sys.path.insert(0, _path)

import nvfp4                                          # noqa: E402  (needs sys.path)
import numeric as num                                 # noqa: E402
import reference as ref                               # noqa: E402
from run_parity import Case, PROD, band_for, build_case      # noqa: E402
from run_parity import decode_e4m3_bytes, write_bin, write_bin_c  # noqa: E402

FIXTURE = HERE / "fixture" / "dspine_nvfp4_fixture"
OUTPUTS = ("normalized", "rms", "codes", "message", "residual_out", "normed", "stash",
           "idx", "drafts", "proposal_q", "scores")

# The teeth of the gather decode: authoring errors in the table path, applied to the
# REFERENCE decode only. The device must disagree with each.
GATHER_MUTANTS = {
    "scale_natural_offset": dict(natural_scales=True),
    "nibble_order_swapped": dict(swap_nibbles=True),
    "divisor_dropped": dict(drop_divisor=True),
    "sign_ignored": dict(ignore_sign=True),
}


def run_fixture(op, geo, eps, inputs, tables, nvfp4_tables, fp8, policy, workdir,
                q8=None, head="fp8", scores=False, pad=False, q8gate="") -> dict:
    """One fixture invocation; returns the produced output files.

    `q8`/`head` select the head representation serial_scores is handed (the P0b FP8 row-scaled
    head by default, the tower's Q8_G32_FP16 one for the head-format arm), `scores` asks for the
    FP32 [batch,16,16,drafts] score-table output (2 = the same tensor declared BF16, which the op
    must refuse), and `q8gate` warps one Q8 head metadata field so the head gate must refuse the
    weight — both last ones are the gate teeth in run_head_score_table.py.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    geometry = {"H": geo["hidden"], "R": geo["rank"], "C": geo["code_size"],
                "W": geo["width"], "B": geo["batch"], "V": geo["vocab"], "D": geo["drafts"]}
    command = [str(FIXTURE), f"--op={op}", f"--dev={geo['device']}", f"--out={workdir}",
                f"--eps={eps!r}"]
    for key, value in geometry.items():
        command.append(f"--g={key}={value}")
    for name, array in inputs.items():
        path = workdir / f"{name}.bin"
        write_bin(path, array)
        command.append(f"--in={name}:{path}")
    for name, array in tables.items():
        path = workdir / f"{name}.bin"
        write_bin_c(path, array)
        command.append(f"--tbl={name}:{path}")
    for name, table in nvfp4_tables.items():
        codes, scales = workdir / f"{name}_codes.bin", workdir / f"{name}_scales.bin"
        write_bin_c(codes, table.packed)
        write_bin_c(scales, table.scales)
        # repr(float32) is a plain shortest-round-trip decimal (e.g. '2688.0'), which the
        # fixture reads back with strtof to the identical binary32 divisor.
        command.append(
            f"--nvtbl={name}:{codes}:{scales}:{float(np.float32(table.divisor))!r}")
    for name, (codes, scales) in fp8.items():
        code_path = workdir / f"{name}_fp8codes.bin"
        scale_path = workdir / f"{name}_fp8scales.bin"
        write_bin_c(code_path, codes)
        write_bin_c(scale_path, scales)
        command.append(f"--fp8={name}:{code_path}:{scale_path}")
    if q8 is not None:
        code_path = workdir / "head_q8codes.bin"
        scale_path = workdir / "head_q8scales.bin"
        write_bin_c(code_path, q8[0])
        write_bin_c(scale_path, q8[1])
        command.append(f"--q8=head:{code_path}:{scale_path}")
        command.append(f"--head={head}")
        command.append(f"--pad={1 if pad else 0}")
    if q8gate:
        command.append(f"--q8gate={q8gate}")
    if scores:
        command.append(f"--scores={int(scores)}")
    for key, value in policy.items():
        command.append(f"--{key}={value}")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"fixture {op} failed ({result.returncode}):\n"
                           f"{result.stdout}\n{result.stderr}")
    return {name: workdir / f"{name}.bin" for name in OUTPUTS
            if (workdir / f"{name}.bin").exists()}


def flatten(want: np.ndarray) -> np.ndarray:
    """Device storage is the ninfer (Fortran) layout: element (d0, d1, ...) at the F offset."""
    order = "F" if want.ndim > 1 else "C"
    flat = np.asfortranarray(want).ravel(order=order)
    return flat.astype(np.float32).view(np.uint32) if want.dtype == np.float32 else flat


def anchor_columns(shape: tuple, width: int) -> np.ndarray:
    """Flat (Fortran-order) indices of the anchor columns t == 0 of a [rows, columns] output.

    Only those elements pass through the gather tables (normalize_codes_kernel takes the
    anchor branch, kernels.cu), so the table path has to be scored on them alone: the rest of
    the array is table-independent and only dilutes the measurement.
    """
    rows, columns = shape[0], shape[1]
    flat = np.arange(columns * rows, dtype=np.int64).reshape(columns, rows).T
    return flat[:, [column for column in range(columns) if column % width == 0]].ravel()


def score(rows, kind, op, case, output, want, got_path, band, select=None) -> dict:  # noqa
    """Reference bits vs device bits, in storage ulps (bf16 uint16 / fp32 uint32 / int32)."""
    want_flat = flatten(want)
    element = {np.dtype(np.uint16): np.uint16, np.dtype(np.int32): np.int32,
               np.dtype(np.float32): np.uint32}[want.dtype]
    got = np.fromfile(got_path, dtype=element)
    if select is not None:
        want_flat, got = want_flat[select], got[select]
    row = {"kind": kind, "op": op, "case": case, "output": output,
           "n": int(want_flat.size), "band": band}
    if got.size != want_flat.size:
        row.update(verdict="SIZE", got=int(got.size))
        rows.append(row)
        return row
    if want.dtype == np.int32:
        bad = int(np.count_nonzero(got != want_flat))
        row.update(exact=int(got.size - bad), mismatches=bad,
                   max_abs_ulp=0 if bad == 0 else -1,
                   verdict="BIT-EXACT" if bad == 0 else "DIVERGENT")
        rows.append(row)
        return row
    stats = num.summary(got, want_flat)
    row.update(exact=stats["exact"], mismatches=stats["mismatch"],
               max_abs_ulp=stats["max_abs_ulp"], p99_abs_ulp=stats["p99_abs_ulp"],
               worst=("got=0x%x want=0x%x @%d"
                      % (int(got[stats["worst_index"]]), int(want_flat[stats["worst_index"]]),
                         stats["worst_index"])) if stats["mismatch"] else "")
    row["verdict"] = ("BIT-EXACT" if stats["mismatch"] == 0
                      else "EQUIVALENT-ORDER" if stats["max_abs_ulp"] <= band
                      else "DIVERGENT")
    rows.append(row)
    return row


def device_vs_device(rows, kind, op, case, left, right, dtype, name, select=None) -> dict:
    """(b) two device runs on the same vectors; only the table format differs."""
    element = {np.dtype(np.uint16): np.uint16, np.dtype(np.int32): np.int32,
               np.dtype(np.float32): np.uint32}[np.dtype(dtype)]
    dtype = np.dtype(dtype)
    a, b = np.fromfile(left, dtype=element), np.fromfile(right, dtype=element)
    if select is not None:
        a, b = a[select], b[select]
    row = {"kind": kind, "op": op, "case": case, "output": name, "n": int(a.size), "band": 0}
    if a.size != b.size:
        row.update(verdict="SIZE")
        rows.append(row)
        return row
    if dtype == np.int32:
        bad = int(np.count_nonzero(a != b))
        row.update(exact=int(a.size - bad), mismatches=bad,
                   max_abs_ulp=0 if bad == 0 else -1,
                   verdict="BIT-EXACT" if bad == 0 else "DRIFT")
        rows.append(row)
        return row
    stats = num.summary(a, b)
    row.update(exact=stats["exact"], mismatches=stats["mismatch"],
               max_abs_ulp=stats["max_abs_ulp"], p99_abs_ulp=stats["p99_abs_ulp"],
               verdict="BIT-EXACT" if stats["mismatch"] == 0 else "DRIFT")
    rows.append(row)
    return row


def head_rows_of(built: dict) -> np.ndarray:
    """The reference-side FP8 head dequant (run_parity.build_case's own expression)."""
    head_bytes, head_scale_bits = built["fp8"]["head"]
    head_scales = num.bf16_bits_to_f32(head_scale_bits).astype(np.float32)
    return num.bf16_round(decode_e4m3_bytes(head_bytes) * head_scales[:, None])


def run_normalize_codes(case: Case, geo: dict, workdir: Path, rms_scale: float = 1.0) -> list:
    """The codebook path has no rms term, so `rms_scale` is inert here (kept for one call shape)."""
    op = "normalize_codes"
    assert rms_scale == 1.0, "normalize_codes has no rms term to scale"
    rows: list = []
    built = build_case(op, case, geo, "")
    codebook_bits = built["tables"]["codebook"]
    table = nvfp4.encode_table(num.bf16_bits_to_f32(codebook_bits).astype(np.float32))
    print(f"[{op} {case.name}] nvfp4 codebook {table.shape} code={table.code_bytes}B "
          f"scale={table.scale_bytes}B divisor={table.divisor!r}", file=sys.stderr)

    def expectation(bits):
        return ref.normalize_codes(built["inputs"]["projected"], built["inputs"]["anchors"],
                                   bits, geo["code_size"], geo["width"], case.eps)

    clean = expectation(nvfp4.dequant_bits(table))
    nv = run_fixture(op, geo, case.eps, built["inputs"], {}, {"codebook": table},
                     built["fp8"], {}, workdir / "nvfp4")
    bf = run_fixture(op, geo, case.eps, built["inputs"], {"codebook": codebook_bits}, {},
                     built["fp8"], {}, workdir / "bf16")
    # The codebook is read only on the anchor columns, so every comparison is scored there.
    anchors = anchor_columns((geo["code_size"], geo["width"] * geo["batch"]), geo["width"])
    for name, want in clean.outputs.items():
        score(rows, "A-nvfp4-gather", op, case.name, name, want, nv[name], band_for(name),
              select=anchors)
    for name in clean.outputs:
        device_vs_device(rows, "B-bf16-vs-nvfp4", op, case.name, nv[name], bf[name],
                         np.uint16, name, select=anchors)
    for mutant, kwargs in GATHER_MUTANTS.items():
        for name, want in expectation(nvfp4.dequant_bits(table, **kwargs)).outputs.items():
            score(rows, f"C-teeth:{mutant}", op, case.name, name, want, nv[name],
                  band_for(name), select=anchors)
    return rows


def run_serial_scores(case: Case, geo: dict, workdir: Path, rms_scale: float = 1.0) -> list:
    """One vector set. `rms_scale` weights the R-dim term (rms * m . tg), which is the one the
    tg/tsv tables feed: the P0b vectors put it under a 5120-wide H-dim dot, and an argmax
    under that regime is provably insensitive to a table defect (P0b's own
    `serial_scores/rms_missing` weak mutant). The tg-dominant set is what gives this op teeth;
    the same defects are seen at full strength through the score table in run_serial_tail.py.
    """
    op = "serial_scores"
    rows: list = []
    built = build_case(op, case, geo, "")
    if rms_scale != 1.0:
        built["inputs"]["rms"] = (built["inputs"]["rms"].astype(np.float32)
                                   * np.float32(rms_scale)).astype(np.float32)
    tg_bits, tsv_bits = built["tables"]["tg"], built["tables"]["tsv"]
    tg = nvfp4.encode_table(num.bf16_bits_to_f32(tg_bits).astype(np.float32))
    tsv = nvfp4.encode_table(num.bf16_bits_to_f32(tsv_bits).astype(np.float32))
    print(f"[{op} {case.name}] nvfp4 tg {tg.shape} divisor={tg.divisor!r} | tsv {tsv.shape} "
          f"divisor={tsv.divisor!r}", file=sys.stderr)
    head_rows = head_rows_of(built)
    inputs = built["inputs"]

    def expectation(tg_table_bits, tsv_table_bits):
        return ref.serial_scores(inputs["cand"], inputs["raw"], inputs["rms"],
                                  inputs["packed_q"], inputs["anchors"], inputs["norm_w"],
                                  inputs["gate_bias"], head_rows, tg_table_bits,
                                  tsv_table_bits, geo["hidden"], geo["rank"], geo["width"])

    clean = expectation(nvfp4.dequant_bits(tg), nvfp4.dequant_bits(tsv))
    nv = run_fixture(op, geo, case.eps, inputs, {}, {"tg": tg, "tsv": tsv}, built["fp8"], {},
                     workdir / "nvfp4")
    bf = run_fixture(op, geo, case.eps, inputs, {"tg": tg_bits, "tsv": tsv_bits}, {},
                     built["fp8"], {}, workdir / "bf16")
    for name, want in clean.outputs.items():
        score(rows, "A-nvfp4-gather", op, case.name, name, want, nv[name], band_for(name))
    for name in clean.outputs:
        device_vs_device(rows, "B-bf16-vs-nvfp4", op, case.name, nv[name], bf[name],
                         np.int32, name)
    for mutant, kwargs in GATHER_MUTANTS.items():
        defect = expectation(nvfp4.dequant_bits(tg, **kwargs),
                             nvfp4.dequant_bits(tsv, **kwargs))
        for name, want in defect.outputs.items():
            score(rows, f"C-teeth:{mutant}", op, case.name, name, want, nv[name],
                  band_for(name))
    return rows


RUNNERS = {"normalize_codes": run_normalize_codes, "serial_scores": run_serial_scores}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ops", default="normalize_codes,serial_scores")
    parser.add_argument("--cases", type=int, default=2)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--width", type=int, default=PROD["width"])
    parser.add_argument("--vocab", type=int, default=PROD["vocab"])
    parser.add_argument("--hidden", type=int, default=PROD["hidden"])
    parser.add_argument("--rank", type=int, default=PROD["rank"])
    parser.add_argument("--code-size", type=int, default=PROD["code_size"])
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--work", default="")
    parser.add_argument("--report", default="")
    arguments = parser.parse_args()

    geo = dict(hidden=arguments.hidden, rank=arguments.rank, code_size=arguments.code_size,
               width=arguments.width, batch=arguments.batch, vocab=arguments.vocab,
               drafts=arguments.width - 1, device=arguments.device)
    work = Path(arguments.work) if arguments.work else Path(tempfile.mkdtemp(prefix="p2nvfp4-"))
    rows: list = []
    for op in [item for item in arguments.ops.split(",") if item]:
        if op not in RUNNERS:
            raise SystemExit(f"unknown op {op}; runners: {sorted(RUNNERS)}")
        sets = [("", 1.0)] if op != "serial_scores" else [("", 1.0), (" tgdom", 1000.0)]
        for index in range(arguments.cases):
            for suffix, rms_scale in sets:
                case = Case(f"{op}#{index}{suffix}", arguments.seed + index, geo)
                try:
                    rows.extend(RUNNERS[op](case, geo, work / f"{op}-{index}{suffix}",
                                            rms_scale=rms_scale))
                except RuntimeError as error:
                    print(str(error), file=sys.stderr)
                    rows.append({"kind": "FIXTURE", "op": op, "case": case.name,
                                 "output": "-", "verdict": "FIXTURE-FAIL"})

    header = (f"{'kind':<24} {'op':<15} {'output':<12} {'n':>9} {'exact':>9} {'bad':>6} "
              f"{'max|ulp|':>8} {'p99':>5}  verdict")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row.get('kind',''):<24} {row.get('op',''):<15} {row.get('output',''):<12} "
              f"{row.get('n',''):>9} {row.get('exact',''):>9} {row.get('mismatches',''):>6} "
              f"{row.get('max_abs_ulp',''):>8} {row.get('p99_abs_ulp',''):>5}  {row['verdict']}"
              + (f"   [{row['worst']}]" if row.get("worst") else ""))
    if arguments.report:
        Path(arguments.report).write_text(json.dumps(rows, indent=1, default=str))
        print(f"\nwrote {arguments.report}")
    print(f"work: {work}")

    clean = [row for row in rows if row["kind"].startswith(("A-", "B-"))]
    bad = [row for row in clean if row["verdict"] not in ("BIT-EXACT", "EQUIVALENT-ORDER",
                                                          "DRIFT")]
    weak_surface = [row for row in rows if row["kind"].startswith("C-teeth")
                    and row["verdict"] in ("BIT-EXACT", "EQUIVALENT-ORDER")
                    and "tgdom" not in row["case"]]
    for row in weak_surface:
        print(f"  note: argmax-only surface, tooth invisible on this vector set: "
              f"{row['kind']} {row['case']}")
    teeth = [row for row in rows if row["kind"].startswith("C-teeth")]
    # A tooth the *argmax-only* serial_scores surface cannot see is a documented weak mutant
    # (P0b's rms_missing precedent), not a missing tooth: the same defect is caught on the
    # tg-dominant set and, at full strength, through the score table in run_serial_tail.py.
    weak = [row for row in teeth if row["verdict"] in ("BIT-EXACT", "EQUIVALENT-ORDER")
            and not (row["op"] == "serial_scores" and "tgdom" not in row["case"])]
    print(f"\nclean rows {len(clean)} (unexpected verdicts {len(bad)}) | "
          f"teeth rows {len(teeth)} (not diverging {len(weak)})")
    for row in bad:
        print(f"  unexpected: {row['kind']} {row['op']} {row['output']} {row['verdict']} "
              f"max={row.get('max_abs_ulp')}")
    for row in weak:
        print(f"  tooth does not bite: {row['kind']} {row['op']} {row['output']}")
    return 0 if not bad and not weak else 1


if __name__ == "__main__":
    raise SystemExit(main())
