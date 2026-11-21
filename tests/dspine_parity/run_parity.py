"""P0b parity driver: real compiled op (GPU fixture) vs Tencent reference (NumPy).

    ./run_parity.py --ops finish,prepare_read            # measured table
    ./run_parity.py --ops finish --mutant rounding_point  # teeth

Geometry defaults are the prod DSpine dims (hidden 5120, rank 512, code size 1024,
width 16 = 15 drafts + anchor) with batch/seq kept tiny; the vocab axis is the one
deviation (a pure gather index, never reduced over).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numeric as num
import reference as ref
from mutants import apply_mutant

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixture" / "dspine_fixture"

PROD = dict(hidden=5120, rank=512, code_size=1024, width=16, vocab=1024)

# EQUIVALENT-ORDER bands, per output: the max abs-ulp a documented reduction-order /
# intrinsic (rsqrtf, expf) / signed-zero difference may reach before we call it a real
# defect. Elementwise outputs (normalized, codes, message, idx, drafts) are required to
# be BIT-EXACT (band 0) - that is what makes the cast-placement mutants detectable.
# Reduction outputs get a wider band, set from the measured clean-case maxima (fp32 rms
# ~34, bf16 normed ~2) with margin. Every mutant must land outside its output's band.
BANDS = {
    "rms": 64,          # fp32 sum-of-squares reduction order
    "normed": 8,        # bf16, inherits the rms reduction + rsqrtf intrinsic
    "residual_out": 2,  # bf16 elementwise; allow the signed-zero sign-bit artifact
}
DEFAULT_BAND = 0        # everything else must be bit-exact


def band_for(output: str) -> int:
    return BANDS.get(output, DEFAULT_BAND)


def decode_e4m3_bytes(bytes_u8: np.ndarray) -> np.ndarray:
    """Vectorized E4M3 decode (fp8.cuh:50); every result is exact in bf16."""
    bits = np.asarray(bytes_u8, dtype=np.uint32)
    sign = np.where((bits >> 7) & 1, -1.0, 1.0).astype(np.float32)
    exponent = ((bits >> 3) & 0xF).astype(np.int32)
    mantissa = (bits & 0x7).astype(np.float32)
    normal = (1.0 + mantissa / 8.0) * np.ldexp(np.ones(bits.shape, np.float32),
                                              (exponent - 7).astype(np.int32))
    subnormal = (mantissa / 8.0) * np.float32(2.0**-6)
    return (sign * np.where(exponent == 0, subnormal, normal)).astype(np.float32)


def allowed_e4m3_bytes(exp_lo: int = 4, exp_hi: int = 9) -> np.ndarray:
    return np.array([byte for byte in range(256)
                     if exp_lo <= ((byte >> 3) & 0xF) <= exp_hi
                     and not (((byte >> 3) & 0xF) == 15 and (byte & 0x7) == 7)],
                    dtype=np.uint8)


class Case:
    def __init__(self, name: str, seed: int, geo: dict, eps: float = 1e-6):
        self.name = name
        self.seed = seed
        self.geo = geo
        self.eps = eps
        self.rng = np.random.default_rng(seed)

    def rand(self, *shape, lo=-1.0, hi=1.0):
        return self.rng.uniform(lo, hi, size=shape).astype(np.float32)

    def dyadic(self, *shape, **kw):
        return num.dyadic(shape, seed=int(self.rng.integers(0, 2**31)), **kw)

    def fp32(self, *shape, **kw):
        return num.fp32_exact(shape, seed=int(self.rng.integers(0, 2**31)), **kw)

    def ids(self, *shape, vocab: int):
        return self.rng.integers(0, vocab, size=shape).astype(np.int32)


def write_bin(path: Path, array: np.ndarray):
    # Tensor inputs use the ninfer (Fortran) layout: first axis fastest, element
    # (h, c) at offset c*H + h. tobytes(order='F') reproduces that (a bare tobytes()
    # defaults to C-order even for F-contiguous arrays - a NumPy gotcha). 1-D arrays
    # are unaffected. The reference side reshapes back with order='F' to match.
    path.write_bytes(np.asfortranarray(array).tobytes(order="F"))


def write_bin_c(path: Path, array: np.ndarray):
    # Weight tables (codebook/tg/tsv) and the fp8 head are indexed row-major by the
    # kernel (data[row * k + col]), so they are written C-order - the opposite of the
    # Tensor inputs. The reference gathers rows directly, so C-order matches.
    path.write_bytes(np.ascontiguousarray(array).tobytes(order="C"))


def run_fixture(op: str, geo: dict, eps: float, inputs: dict, tables: dict, fp8: dict,
                stash: bool, workdir: Path) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    geometry = {"H": geo["hidden"], "R": geo["rank"], "C": geo["code_size"],
                "W": geo["width"], "B": geo["batch"], "V": geo["vocab"],
                "D": geo["width"] - 1}
    command = [str(FIXTURE), f"--op={op}", f"--dev={geo['device']}", f"--out={workdir}",
               f"--eps={eps!r}", f"--stash={1 if stash else 0}"]
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
    for name, (codes, scales) in fp8.items():
        code_path, scale_path = workdir / f"{name}_codes.bin", workdir / f"{name}_scales.bin"
        write_bin_c(code_path, codes)
        write_bin_c(scale_path, scales)
        command.append(f"--fp8={name}:{code_path}:{scale_path}")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"fixture {op} failed ({result.returncode}):\n"
                           f"{result.stdout}\n{result.stderr}")
    outputs = {}
    for name in ("normalized", "rms", "codes", "message", "residual_out", "normed", "stash",
                 "idx", "drafts"):
        path = workdir / f"{name}.bin"
        if path.exists():
            outputs[name] = path
    return outputs


def compare(op: str, case: Case, expected: ref.ReferenceResult, produced: dict,
            files: dict, mutant: str) -> list[dict]:
    rows = []
    for name, want in expected.outputs.items():
        path = files.get(name)
        if path is None:
            rows.append({"op": op, "case": case.name, "output": name, "verdict": "NO-OUTPUT"})
            continue
        # The fixture writes outputs in the ninfer (Fortran) layout, so flatten the
        # logical `want` array in F-order to align element (h, c) with flat index c*H+h.
        want_f = np.asfortranarray(want)
        if want.dtype == np.uint16:
            element, want_flat = np.uint16, want_f.ravel(order="F")
        elif want.dtype == np.int32:
            element, want_flat = np.int32, want_f.ravel(order="F")
        else:  # fp32: compare the stored bit patterns
            element = np.uint32
            want_flat = want_f.astype(np.float32).view(np.uint32).ravel(order="F")
        got = np.fromfile(path, dtype=element)
        row = {"op": op, "case": case.name, "output": name,
               "shape": "x".join(str(d) for d in want.shape)}
        if got.size != want_flat.size:
            row.update(verdict="SIZE", got=int(got.size), want=int(want_flat.size))
            rows.append(row)
            continue
        if want.dtype == np.int32:
            mismatches = int(np.count_nonzero(got != want_flat))
            row.update(n=int(got.size), exact=int(got.size - mismatches),
                       mismatches=mismatches, max_abs_ulp=0 if mismatches == 0 else -1,
                       verdict="BIT-EXACT" if mismatches == 0 else "DIVERGENT")
        else:
            stats = num.summary(got, want_flat)
            delta = num.ulp_distance(got, want_flat)
            row.update(n=stats["n"], exact=stats["exact"], mismatches=stats["mismatch"],
                       max_abs_ulp=stats["max_abs_ulp"], p99_abs_ulp=stats["p99_abs_ulp"],
                       worst=("got=%d want=%d at %d" % (int(got[stats["worst_index"]]),
                                                        int(want_flat[stats["worst_index"]]),
                                                        stats["worst_index"])
                              if stats["mismatch"] else ""))
            band = band_for(name)
            row["verdict"] = ("BIT-EXACT" if stats["mismatch"] == 0
                              else "EQUIVALENT-ORDER" if stats["max_abs_ulp"] <= band
                              else "DIVERGENT")
        rows.append(row)
    if mutant:
        rows.append({"op": op, "case": case.name, "output": f"mutant={mutant}",
                     "verdict": "see --report"})
    return rows


def build_case(op: str, case: Case, geo: dict, mutant: str) -> dict:
    """Produce the fixture inputs + the reference expectation for one op/case."""
    hidden, rank, code_size = geo["hidden"], geo["rank"], geo["code_size"]
    width, batch, vocab = geo["width"], geo["batch"], geo["vocab"]
    drafts = width - 1
    columns = width * batch
    out = dict(inputs={}, tables={}, fp8={}, stash=False, geo=dict(geo))

    if op == "prepare_read":
        residual = num.bf16_view(case.dyadic(hidden, columns))
        out["inputs"] = {"residual": residual}
        out["expect"] = apply_mutant("prepare_read", {"residual": residual},
                                     dict(hidden=hidden, eps=case.eps), mutant)
    elif op == "normalize_codes":
        projected = num.bf16_view(case.dyadic(code_size, columns, exp_lo=-5, exp_hi=1))
        anchors = case.ids(batch, vocab=vocab)
        codebook = num.bf16_view(case.dyadic(vocab, code_size, exp_lo=-4, exp_hi=0))
        out["inputs"] = {"projected": projected, "anchors": anchors}
        out["tables"] = {"codebook": codebook}
        out["expect"] = apply_mutant("normalize_codes",
                                     {"projected_bits": projected, "anchors": anchors,
                                      "codebook_bits": codebook},
                                     dict(code_size=code_size, width=width, eps=case.eps),
                                     mutant)
    elif op == "message":
        packed = num.bf16_view(case.dyadic(3 * rank, width, batch, exp_lo=-5, exp_hi=0))
        gate_bias = num.bf16_view(case.dyadic(rank, exp_lo=-3, exp_hi=0))
        out["inputs"] = {"packed": packed, "gate_bias": gate_bias}
        out["expect"] = apply_mutant("message", {"packed_bits": packed,
                                                 "gate_bias_bits": gate_bias},
                                      dict(rank=rank, width=width, batch=batch), mutant)
    elif op == "finish":
        residual = num.bf16_view(case.dyadic(hidden, width, batch, exp_lo=-6, exp_hi=1))
        rms = case.fp32(columns, exp_lo=-2, exp_hi=1)
        delta = num.bf16_view(case.dyadic(hidden, width, batch, exp_lo=-6, exp_hi=1))
        next_norm = num.bf16_view(case.dyadic(hidden, exp_lo=-2, exp_hi=1))
        out["inputs"] = {"residual": residual, "rms": rms, "delta": delta,
                         "next_norm": next_norm}
        out["stash"] = True
        out["expect"] = apply_mutant("finish",
                                      {"residual_bits": residual, "rms": rms,
                                       "delta_bits": delta, "next_norm_bits": next_norm},
                                      dict(hidden=hidden, width=width, eps=case.eps), mutant)
    elif op == "serial_scores":
        raw = num.bf16_view(case.dyadic(hidden, width, batch, exp_lo=-6, exp_hi=1))
        rms = case.fp32(width, batch, exp_lo=-2, exp_hi=1)
        packed_q = num.bf16_view(case.dyadic(rank, width, batch, exp_lo=-5, exp_hi=0))
        anchors = case.ids(batch, vocab=vocab)
        norm_w = num.bf16_view(case.dyadic(hidden, exp_lo=-3, exp_hi=0))
        gate_bias = num.bf16_view(case.dyadic(rank, exp_lo=-3, exp_hi=0))
        table = allowed_e4m3_bytes()
        head_bytes = table[case.rng.integers(0, table.size, size=(vocab, hidden))]
        head_scales = np.full(vocab, 1.0, dtype=np.float32)
        tg = num.bf16_view(case.dyadic(vocab, rank, exp_lo=-5, exp_hi=0))
        tsv = num.bf16_view(case.dyadic(vocab, 2 * rank, exp_lo=-5, exp_hi=0))
        cand_rows = []
        for b in range(batch):
            for i in range(drafts):
                row = np.random.default_rng(case.seed + 1000 + b * 100 + i).permutation(vocab)[:16]
                cand_rows.append(row.astype(np.int32))
        # (16, drafts, batch) = (slot, row, batch); slot is the fastest axis.
        cand = np.stack(cand_rows).reshape(batch, drafts, 16).transpose(2, 1, 0).copy()
        head_rows = num.bf16_round(decode_e4m3_bytes(head_bytes)
                                   * head_scales[:, None])
        out["inputs"] = {"cand": cand, "raw": raw, "rms": rms, "packed_q": packed_q,
                         "anchors": anchors, "norm_w": norm_w, "gate_bias": gate_bias}
        out["tables"] = {"tg": tg, "tsv": tsv}
        out["fp8"] = {"head": (head_bytes, num.bf16_view(head_scales))}
        out["expect"] = apply_mutant("serial_scores",
                                      {"cand": cand, "raw_bits": raw, "rms": rms,
                                       "packed_q_bits": packed_q, "anchors": anchors,
                                       "norm_w_bits": norm_w, "gate_bias_bits": gate_bias,
                                       "head_rows": head_rows, "tg_bits": tg,
                                       "tsv_bits": tsv},
                                      dict(hidden=hidden, rank=rank, width=width), mutant)
    elif op == "serial_chain":
        cand = case.ids(16, drafts, batch, vocab=vocab)
        idx = case.rng.integers(0, 16, size=(16, drafts, batch)).astype(np.int32)
        out["inputs"] = {"cand": cand, "idx": idx}
        out["expect"] = apply_mutant("serial_chain", {"cand": cand, "idx": idx}, {}, mutant)
    else:
        raise SystemExit(f"unknown op {op}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ops", default="finish")
    parser.add_argument("--cases", type=int, default=3)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--width", type=int, default=PROD["width"])
    parser.add_argument("--vocab", type=int, default=PROD["vocab"])
    parser.add_argument("--hidden", type=int, default=PROD["hidden"])
    parser.add_argument("--rank", type=int, default=PROD["rank"])
    parser.add_argument("--code-size", type=int, default=PROD["code_size"])
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--mutant", default="")
    parser.add_argument("--work", default="")
    parser.add_argument("--report", default="")
    arguments = parser.parse_args()

    geo = dict(hidden=arguments.hidden, rank=arguments.rank, code_size=arguments.code_size,
               width=arguments.width, batch=arguments.batch, vocab=arguments.vocab,
               device=arguments.device)
    work = Path(arguments.work) if arguments.work else Path(tempfile.mkdtemp(prefix="p0b-"))
    rows = []
    for op in [item for item in arguments.ops.split(",") if item]:
        for index in range(arguments.cases):
            case = Case(f"{op}#{index} seed={arguments.seed + index}",
                        arguments.seed + index, geo)
            built = build_case(op, case, geo, arguments.mutant)
            try:
                files = run_fixture(op, built["geo"], case.eps, built["inputs"],
                                     built["tables"], built["fp8"], built["stash"],
                                     work / f"{op}-{index}")
            except RuntimeError as error:
                print(str(error), file=sys.stderr)
                rows.append({"op": op, "case": case.name, "output": "-", "verdict": "FIXTURE-FAIL"})
                continue
            rows.extend(compare(op, case, built["expect"], {}, files, arguments.mutant))

    header = f"{'op':<15} {'output':<13} {'shape':<16} {'n':>8} {'exact':>8} " \
              f"{'bad':>6} {'max|ulp|':>8} {'p99':>4}  verdict"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row.get('op',''):<15} {row.get('output',''):<13} "
              f"{row.get('shape',''):<16} {row.get('n',''):>8} {row.get('exact',''):>8} "
              f"{row.get('mismatches',''):>6} {row.get('max_abs_ulp',''):>8} "
              f"{row.get('p99_abs_ulp',''):>4}  {row['verdict']}"
              + (f"   [{row['worst']}]" if row.get("worst") else "")
              + (f"   [{row['case']}]" if "shape" not in row else ""))
    if arguments.report:
        Path(arguments.report).write_text(json.dumps(rows, indent=1, default=str))
        print(f"\nwrote {arguments.report}")
    print(f"work: {work}")
    return 0 if all(row["verdict"] in ("BIT-EXACT", "CHECK") for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
