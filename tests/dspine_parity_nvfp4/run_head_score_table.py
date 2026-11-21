"""P2: the TOWER HEAD representation + the SCORE-TABLE producer, and the composition.

Two gaps sit between the verified bridge ops and the sampled tail arm (D-34):

  (1) `dspine_serial_scores` requires the proposal head as FP8_E4M3FN_ROW_BF16, while the tower
      head it reuses is `q8_g32_fp16 [248320, 5120]` (`models/qwen3_5/definition.cpp:99`). The
      answer is dequant-on-gather, per row, the way the codebook / tg / tsv rows already are: a
      dequantized COPY of the head (1.18 GiB fp8 codes + scales, 2.35 GiB bf16) breaks the D-14
      drafter sizing outright.
  (2) The op computed the K x K score matrix and then emitted only its argmax, so the sampled
      tail - whose input IS that FP32 [batch, 16, 16, drafts] table - had no producer.

This harness measures both on the device, through the P0b vectors (`tests/dspine_parity`), with
`q8.py` as the reference side of the Q8 gather:

  A-q8-gather       device `idx` with a Q8 head vs the P0b reference math fed the CPU decode of
                    the SAME Q8 bytes. The gather path is SOUND only at BIT-EXACT.
  A-score-table     the op's FP32 score table vs the table expression of `run_serial_tail`.
  A-table-argmax    argmax(candidate axis) of the device table vs the device `idx`: the layout
                    proof that the emitted table is the very matrix the argmax collapsed, i.e.
                    the table the tail is allowed to consume.
  B-fp8-vs-q8       device FP8-head `idx` vs device Q8-head `idx`: the representation trade, on
                    quantized values of one matrix, so the drift is the format's, not the data's.
  C-teeth:<name>    a defect in the REFERENCE head decode (group-offset scales, packed stride,
                    dropped sign, row-scale broadcast) - must land outside the band.
  D-compose         device table -> device `serial_tail`, prod policy (T=0.9, top_k 20,
                    top_p 1.0, min_p 0.05) and T=0; at T=0 the composition must equal the greedy
                    pair (serial_scores -> serial_chain) byte-for-byte (D-24's free check).

The `idx` outputs are compared as integers, the score table as fp32 storage bits.

    CUDA_VISIBLE_DEVICES=1 ./run_head_score_table.py --cases 2 --device 0
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

import numeric as num                                     # noqa: E402
import q8                                                 # noqa: E402
import reference as ref                                   # noqa: E402
from run_parity import Case, PROD, band_for, build_case              # noqa: E402
from run_nvfp4_parity import device_vs_device                       # noqa: E402
from run_nvfp4_parity import head_rows_of, run_fixture, score       # noqa: E402
from run_serial_tail import score_table                    # noqa: E402

FIXTURE = HERE / "fixture" / "dspine_nvfp4_fixture"

# The score table is a matrix-family reduction: a 5120-term dot through the head plus an R-term
# dot, both accumulated in a fixed device order (16 strided partial sums, then a sequential
# combine) against NumPy's pairwise tree. The right criterion for that family is the normwise
# one (docs/maintainer/op-development.md: reductions and matrix ops are judged normwise, with a
# gross pointwise ceiling kept), not a raw fp32 ulp band: at prod the clean case measured a max
# pointwise distance of 1.4e4 fp32 ulps = 1.7e-3 relative, at a relative Frobenius of ~1e-5, i.e.
# cancellation inside a 5120-wide dot and nothing structural. Measured teeth (the tg term dropped
# from the reference table) sit at a relative Frobenius of ~1, four orders above this band.
SCORE_NORM_BAND = 1.0e-3
SCORE_POINT_CEILING = 2.0e-2   # relative pointwise, the gross ceiling

PROD_POLICY = dict(temp=0.9, topk=20, topp=1.0, minp=0.05)
GREEDY_POLICY = dict(temp=0.0, topk=20, topp=1.0, minp=0.0)


def read_f32(path: Path, shape: tuple) -> np.ndarray:
    """A device fp32 tensor file, in the ninfer (Fortran) storage layout."""
    return np.fromfile(path, dtype=np.float32).reshape(shape, order="F")


def read_i32(path: Path, shape: tuple) -> np.ndarray:
    return np.fromfile(path, dtype=np.int32).reshape(shape, order="F")


def table_from(path: Path, shape) -> np.ndarray:
    if isinstance(shape, dict):
        shape = (shape["batch"], 16, 16, shape["drafts"])
    return read_f32(path, tuple(shape))


def argmax_table(table: np.ndarray) -> np.ndarray:
    """Per (batch, sender, slot) the best candidate slot: the tail's own greedy choice."""
    return np.argmax(table, axis=1).astype(np.int32)     # [B, sender, drafts]


def op_score_table(cand, raw_bits, rms, packed_q_bits, anchors, norm_w_bits, gate_bias_bits,
                   head_rows, tg_bits, tsv_bits, hidden, rank, width) -> np.ndarray:
    """FP32 [batch, 16 (candidate), 16 (sender), drafts], written in the layout the op writes and
    `dspine_serial_tail` reads (first axis fastest, candidate before sender).

    The arithmetic is `run_serial_tail.score_table`'s - the table-level view of
    `reference.serial_scores`' `scores` expression, kept per (sender, candidate) instead of being
    collapsed to the argmax. That helper stores its 16 x 16 block as [sender, candidate]; this one
    stores it as [candidate, sender], the axis order the tail's reader indexes
    (`table[b + batch * (k + 16 * (sender + 16 * i))]`).
    """
    drafts, batch = cand.shape[1], cand.shape[2]
    raw = ref.bf16_to_f32(raw_bits).reshape((hidden, width, batch), order="F").astype(np.float32)
    q_all = ref.bf16_to_f32(packed_q_bits).reshape((rank, width, batch),
                                                  order="F").astype(np.float32)
    norm_w = ref.bf16_to_f32(norm_w_bits).astype(np.float32)
    bias = ref.bf16_to_f32(gate_bias_bits).astype(np.float32)
    tg = ref.bf16_to_f32(tg_bits).astype(np.float32)
    tsv = ref.bf16_to_f32(tsv_bits).astype(np.float32)
    rms_w = rms.reshape((width, batch), order="F")
    table = np.zeros((batch, 16, 16, drafts), dtype=np.float32)
    for b in range(batch):
        for i in range(drafts):
            column = 1 + i
            raw_row = raw[:, column, b] * norm_w
            base = (raw_row[:, None] * head_rows[cand[:, i, b]].T).sum(axis=0).astype(np.float32)
            sid = (np.full(16, int(anchors[b]), dtype=np.int64) if i == 0
                   else cand[:, i - 1, b].astype(np.int64))
            q = q_all[:, column, b]
            rows = tsv[sid]
            m = ref.Sigmoid(q[None, :] + rows[:, :rank] + bias[None, :]) * rows[:, rank:]
            # [sender, candidate] -> store transposed, so axis 1 is the candidate.
            table[b, :, :, i] = (base[None, :]
                                 + np.float32(rms_w[column, b]) * (m @ tg[cand[:, i, b]].T)).T
    return table


def head_inputs(built: dict, geo: dict, pad: bool) -> tuple:
    """The tower head in Q8: the FP8 head's own dequantized values, re-quantized (one matrix,
    two representations - so B-fp8-vs-q8 measures the format and not the data)."""
    values = head_rows_of(built).astype(np.float32)
    head = q8.encode_head(values, pad=pad)
    print(f"[head q8] {q8.profile(head)}", file=sys.stderr)
    return head, values


def score_table_row(rows: list, kind: str, case: Case, want: np.ndarray, got_path: Path) -> dict:
    """The table's normwise verdict: relative Frobenius plus a gross pointwise ceiling."""
    got = table_from(got_path, want.shape)
    delta = got.astype(np.float64) - want.astype(np.float64)
    fro = float(np.linalg.norm(delta) / max(np.linalg.norm(want.astype(np.float64)), 1e-300))
    scale = np.maximum(np.abs(want.astype(np.float64)), np.float64(1e-6))
    point = float(np.max(np.abs(delta) / scale))
    verdict = ("BIT-EXACT" if not delta.any() else "EQUIVALENT-ORDER"
               if fro <= SCORE_NORM_BAND and point <= SCORE_POINT_CEILING else "DIVERGENT")
    row = {"kind": kind, "op": "serial_scores", "case": case.name, "output": "scores",
           "n": int(want.size), "band": f"{SCORE_NORM_BAND:.0e}/{SCORE_POINT_CEILING:.0e}",
           "exact": int(want.size - int(np.count_nonzero(delta))),
           "mismatches": int(np.count_nonzero(delta)),
           "max_abs_ulp": round(fro / SCORE_NORM_BAND, 3),
           "p99_abs_ulp": round(point / SCORE_POINT_CEILING, 3), "verdict": verdict}
    rows.append(row)
    print(f"  [{case.name} {kind}] relative-Frobenius {fro:.3e} (band "
          f"{SCORE_NORM_BAND:.0e}), max relative pointwise {point:.3e} (ceiling "
          f"{SCORE_POINT_CEILING:.0e}) -> {verdict}", file=sys.stderr)
    return row


def run_head_case(case: Case, geo: dict, workdir: Path, pad: bool, rms_scale: float) -> list:
    op = "serial_scores"
    rows: list = []
    built = build_case(op, case, geo, "")
    inputs = built["inputs"]
    if rms_scale != 1.0:
        # The op is linear in rms, and the same bytes go to the device and to the reference: this
        # is a regime choice, not a tolerance. D-27 scales it UP to put the ranking on the table
        # term; the head-format arm needs the opposite regime, where the ranking is the head-term
        # dot alone (D-27's documented argmax weak surface at the natural scale).
        inputs["rms"] = (inputs["rms"].astype(np.float32)
                         * np.float32(rms_scale)).astype(np.float32)
    head, _ = head_inputs(built, geo, pad)
    tables = {"tg": built["tables"]["tg"], "tsv": built["tables"]["tsv"]}

    def expectation(head_rows, tg_bits=None, tsv_bits=None):
        return ref.serial_scores(inputs["cand"], inputs["raw"], inputs["rms"],
                                  inputs["packed_q"], inputs["anchors"], inputs["norm_w"],
                                  inputs["gate_bias"], head_rows,
                                  tables["tg"] if tg_bits is None else tg_bits,
                                  tables["tsv"] if tsv_bits is None else tsv_bits,
                                  geo["hidden"], geo["rank"], geo["width"])

    q8_rows = q8.dequant_head(head)
    device = run_fixture(op, geo, case.eps, inputs, tables, {}, {}, {}, workdir / "q8",
                         q8=(head.codes, head.scales), head="q8", pad=pad, scores=True)
    fp8 = run_fixture(op, geo, case.eps, inputs, tables, {}, built["fp8"], {}, workdir / "fp8",
                      scores=True)
    clean = expectation(q8_rows)
    for name, want in clean.outputs.items():
        score(rows, "A-q8-gather", op, case.name, name, want, device[name], band_for(name))

    # (A) the emitted score table vs the table expression, and its own argmax vs the op's idx.
    want_table = op_score_table(inputs["cand"], inputs["raw"], inputs["rms"],
                                 inputs["packed_q"], inputs["anchors"], inputs["norm_w"],
                                 inputs["gate_bias"], q8_rows, tables["tg"], tables["tsv"],
                                 geo["hidden"], geo["rank"], geo["width"])
    score_table_row(rows, "A-score-table", case, want_table, device["scores"])
    device_table = table_from(device["scores"], geo)
    idx_device = read_i32(device["idx"], (16, geo["drafts"], geo["batch"]))
    drawn = argmax_table(device_table).transpose(1, 2, 0)   # -> [sender, drafts, batch]
    mismatches = int(np.count_nonzero(drawn.reshape(-1) != idx_device.reshape(-1)))
    rows.append({"kind": "A-table-argmax", "op": op, "case": case.name, "output": "idx",
                 "n": int(idx_device.size), "band": 0,
                 "exact": int(idx_device.size - mismatches), "mismatches": mismatches,
                 "max_abs_ulp": 0 if mismatches == 0 else -1,
                 "verdict": "BIT-EXACT" if mismatches == 0 else "DIVERGENT"})

    # (B) the representation trade, device vs device.
    for name in clean.outputs:
        device_vs_device(rows, "B-fp8-vs-q8", op, case.name, device[name], fp8[name], np.int32,
                          name)
    device_vs_device(rows, "B-fp8-vs-q8-scores", op, case.name, device["scores"],
                     fp8["scores"], np.float32, "scores")
    q8_table, fp8_table = table_from(device["scores"], geo), table_from(fp8["scores"], geo)
    mass = float(np.linalg.norm(fp8_table.astype(np.float64)))
    print(f"  [{case.name}] head representation trade on the score table: "
          f"{int(np.count_nonzero(q8_table != fp8_table))}/{q8_table.size} entries moved, "
          f"relative-Frobenius {np.linalg.norm((q8_table - fp8_table).astype(np.float64)) / mass:.3e}, "
          f"table range [{q8_table.min():.4g},{q8_table.max():.4g}]", file=sys.stderr)

    # (C) teeth in the head decode.
    for mutant, kwargs in q8.HEAD_MUTANTS.items():
        if mutant == "packed_stride_read" and head.padded == head.k:
            print(f"  note: {mutant} is inert at an unpadded head (padded == k == {head.k})",
                  file=sys.stderr)
            continue
        defect = expectation(q8.dequant_head(head, **kwargs))
        for name, want in defect.outputs.items():
            score(rows, f"C-teeth:{mutant}", op, case.name, name, want, device[name],
                  band_for(name))
    # A score-table tooth: the table written as if the rms * (m . tg) term were not there.
    base_only = op_score_table(inputs["cand"], inputs["raw"], inputs["rms"], inputs["packed_q"],
                                inputs["anchors"], inputs["norm_w"], inputs["gate_bias"], q8_rows,
                                np.zeros_like(tables["tg"]), tables["tsv"], geo["hidden"],
                                geo["rank"], geo["width"])
    score_table_row(rows, "C-teeth:tg_term_dropped", case, base_only, device["scores"])

    # (D) compose on the device: table -> serial_tail, and the greedy pair for the T=0 arm.
    compose(rows, case, geo, inputs, device, workdir)
    return rows


def compose(rows: list, case: Case, geo: dict, inputs: dict, device: dict, workdir: Path):
    """The table the op emitted drives the sampled tail; the T=0 arm must reproduce the greedy
    pair (serial_scores' idx walked by serial_chain) byte-for-byte."""
    batch, drafts = geo["batch"], geo["drafts"]
    cand_tail = np.ascontiguousarray(inputs["cand"].transpose(2, 1, 0))   # [B, D, 16]
    positions = (np.arange(batch, dtype=np.int32) * 3 + 7)
    greedy = run_fixture("serial_chain", geo, case.eps,
                         {"cand": inputs["cand"], "idx": read_i32(
                             device["idx"], (16, drafts, batch))},
                         {}, {}, {}, {}, workdir / "chain")
    chain = read_i32(greedy["drafts"], (drafts, batch))
    for label, policy in (("t09", PROD_POLICY), ("t00", GREEDY_POLICY)):
        tail = run_fixture("serial_tail", geo, case.eps,
                           {"table": table_from(device["scores"], geo), "cand": cand_tail,
                            "positions": positions},
                           {}, {}, {}, policy, workdir / f"tail-{label}")
        drawn = read_i32(tail["drafts"], (batch, drafts))
        q = read_f32(tail["proposal_q"], (batch, drafts, 16))
        # A round the acceptor can consume: every q row finite, non-negative and normalised
        # (serial_tail_accept_q_error's contract), and every drawn id one of that slot's
        # 16 candidates.
        member = np.array([[int(drawn[b, i]) in set(cand_tail[b, i].tolist())
                            for i in range(drafts)] for b in range(batch)])
        valid = (bool(np.isfinite(q).all()) and bool((q >= 0.0).all())
                 and np.allclose(q.sum(axis=2), np.ones((batch, drafts), np.float32),
                                  rtol=0.0, atol=1e-5) and bool(member.all()))
        rows.append({"kind": f"D-compose:{label}", "op": "serial_tail", "case": case.name,
                     "output": "drafts", "n": int(drawn.size), "band": 0,
                     "exact": int(drawn.size), "mismatches": 0, "max_abs_ulp": 0,
                     "verdict": "GREEN" if valid else "DIVERGENT"})
        print(f"  [{case.name} {label}] drafts {drawn.reshape(-1).tolist()}", file=sys.stderr)
        if label == "t00":
            same = int(np.count_nonzero(drawn.transpose(1, 0) != chain))
            rows.append({"kind": "D-compose:t00-greedy-pair", "op": "serial_tail",
                         "case": case.name, "output": "drafts", "n": int(chain.size),
                         "band": 0, "exact": int(chain.size - same), "mismatches": same,
                         "max_abs_ulp": 0 if same == 0 else -1,
                         "verdict": "BIT-EXACT" if same == 0 else "DIVERGENT"})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=2)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--width", type=int, default=PROD["width"])
    parser.add_argument("--vocab", type=int, default=PROD["vocab"])
    parser.add_argument("--hidden", type=int, default=PROD["hidden"])
    parser.add_argument("--rank", type=int, default=PROD["rank"])
    parser.add_argument("--code-size", type=int, default=PROD["code_size"])
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--pad", type=int, default=1,
                        help="lay the Q8 head out at align_up(hidden,128) (the RowSplit rule)")
    parser.add_argument("--work", default="")
    parser.add_argument("--report", default="")
    arguments = parser.parse_args()

    geo = dict(hidden=arguments.hidden, rank=arguments.rank, code_size=arguments.code_size,
               width=arguments.width, batch=arguments.batch, vocab=arguments.vocab,
               drafts=arguments.width - 1, device=arguments.device)
    work = (Path(arguments.work) if arguments.work
            else Path(tempfile.mkdtemp(prefix="p2head-")))
    rows: list = []
    for index in range(arguments.cases):
        for suffix, rms_scale in (("", 1.0), (" headdom", 1e-3)):
            case = Case(f"head#{index}{suffix}", arguments.seed + index, geo)
            try:
                rows.extend(run_head_case(case, geo, work / f"head-{index}{suffix}",
                                           bool(arguments.pad), rms_scale))
            except RuntimeError as error:
                print(str(error), file=sys.stderr)
                rows.append({"kind": "FIXTURE", "op": "serial_scores", "case": case.name,
                             "output": "-", "verdict": "FIXTURE-FAIL"})

    header = (f"{'kind':<28} {'op':<14} {'output':<9} {'n':>9} {'exact':>9} {'bad':>6} "
              f"{'max|ulp|':>8} {'p99':>6}  verdict")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row.get('kind',''):<28} {row.get('op',''):<14} {row.get('output',''):<9} "
              f"{row.get('n',''):>9} {row.get('exact',''):>9} {row.get('mismatches',''):>6} "
              f"{row.get('max_abs_ulp',''):>8} {row.get('p99_abs_ulp',''):>6}  {row['verdict']}"
              + (f"   [{row['worst']}]" if row.get("worst") else ""))
    if arguments.report:
        Path(arguments.report).write_text(json.dumps(rows, indent=1, default=str))
        print(f"\nwrote {arguments.report}")
    print(f"work: {work}")

    clean = [row for row in rows if row["kind"].startswith(("A-", "D-"))]
    bad = [row for row in clean if row["verdict"] not in ("BIT-EXACT", "EQUIVALENT-ORDER",
                                                          "GREEN")]
    teeth = [row for row in rows if row["kind"].startswith("C-teeth")]
    # Which arm sees which tooth is a property of the vectors (D-27): at the natural scale the
    # 5120-wide head-term dot and the R-wide table-term dot compete, so an argmax or a score table
    # may rank identically under a defect in the term that is not ranking it. The `headdom` arm
    # (rms x 1e-3, so the ranking IS the head dot) is the other half of that coverage. A tooth is
    # PROVEN when it bites on the arm that measures the term it corrupts: scored per tooth over
    # all arms, and a tooth that moves nothing anywhere is the failure.
    arms: dict = {}
    for row in teeth:
        arms.setdefault(row["kind"], []).append(row)
    weak = []
    for kind, rows_of_tooth in sorted(arms.items()):
        bit = [row["case"].split(" ")[-1] for row in rows_of_tooth
               if row["verdict"] not in ("BIT-EXACT", "EQUIVALENT-ORDER")]
        scale = [f"{row['max_abs_ulp']}" for row in rows_of_tooth
                 if row["verdict"] not in ("BIT-EXACT", "EQUIVALENT-ORDER")]
        print(f"  tooth {kind:<32} rows {len(rows_of_tooth)}, bites "
              f"{len(bit)}/{len(rows_of_tooth)} "
              f"(arms: {','.join(sorted(set(bit))) or 'NONE'})"
              + (f", strength {','.join(scale[:3])}" if scale else ""))
        if not bit:
            weak.append({"kind": kind, "verdict": "NO-TEETH"})
    print(f"\nclean rows {len(clean)} (unexpected verdicts {len(bad)}) | "
          f"teeth {len(arms)} kinds ({len(weak)} never bite)")
    for row in bad:
        print(f"  unexpected: {row['kind']} {row['op']} {row['output']} {row['verdict']} "
              f"max={row.get('max_abs_ulp')}")
    for row in weak:
        print(f"  tooth never bites: {row['kind']}")
    return 0 if not bad and not weak else 1


if __name__ == "__main__":
    raise SystemExit(main())
