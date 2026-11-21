"""P2: the NVFP4 gather path's effect on the SAMPLED tail (ops::dspine_serial_tail).

`dspine_serial_tail` takes no Weight - its scores arrive as an FP32 table the bridge path
computes from tg/tsv (include/ninfer/ops/dspine_bridge.h: "the bridge path supplies its own
table from ops::dspine_serial_scores"). So the format question for this op is: how much of
the codebook/tg/tsv quantization reaches the draws, the proposal distributions and the walk
through that table. Measured end to end on the device:

  A-format-invariance  the op is fed the SAME table twice. It has no table to dequantize, so
                       the two runs must be byte-identical - the control that makes B a
                       statement about the table, not about the tail's arithmetic.
  B-bf16-vs-nvfp4      table built from BF16 tg/tsv vs from the NVFP4-dequantized tg/tsv:
                       the quantization-induced drift in idx / drafts / proposal_q.
  C-teeth:<name>       a table built through a broken *reference* decode of the real bytes
                       (wrong scale-plane addressing, swapped nibbles, dropped divisor,
                       sign ignored). If a defect in the gather decode cannot move the
                       draws, the comparison has no teeth on this path.

Policy is K1 by default (T=1.0, top_k=20 = no cut, top_p=1.0, min_p=0, fixed seed), the
kernel-dev standard of D-23; K2/A1 arms are --temperature/--min-p away. The op's own parity
vs the C++ host core is tests/ops/dspine (ninfer_dspine_serial_tail_op_test), not this file.

    CUDA_VISIBLE_DEVICES=1 ./run_serial_tail.py --cases 2
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

import nvfp4                                              # noqa: E402
import numeric as num                                     # noqa: E402
import reference as ref                                   # noqa: E402
from run_parity import Case, PROD                         # noqa: E402
from run_nvfp4_parity import (                            # noqa: E402
    GATHER_MUTANTS,
    device_vs_device,
    head_rows_of,
    run_fixture,
)

FIXTURE = HERE / "fixture" / "dspine_nvfp4_fixture"


def score_table(cand, raw_bits, rms, packed_q_bits, anchors, norm_w_bits, gate_bias_bits,
                head_rows, tg_bits, tsv_bits, hidden, rank, width) -> np.ndarray:
    """FP32 [batch, kTopK(cand), kTopK(sender), drafts]: the table the bridge path hands the
    sampled tail. Same arithmetic as reference.serial_scores' `scores` (propose.py:191-217 =
    kernels.cu serial_scores_kernel's score_s), kept per (batch, sender, candidate) instead
    of being collapsed to the argmax - the table-level view of that same expression, so the
    NVFP4-dequantized tg/tsv propagate into it exactly as they propagate into idx.
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
            table[b, :, :, i] = (base[None, :]
                                 + np.float32(rms_w[column, b]) * (m @ tg[cand[:, i, b]].T))
    return table


def tail_inputs(case: Case, geo: dict) -> dict:
    """The P0b serial_scores vectors (same seeds, same geometry), in the tail's layouts."""
    from run_parity import build_case

    built = build_case("serial_scores", case, geo, "")
    inputs = built["inputs"]
    drafts, batch = geo["drafts"], geo["batch"]
    return dict(
        built=built,
        # cand: serial_scores' (slot, row, batch) -> the tail's [batch, drafts, 16]
        cand=np.ascontiguousarray(inputs["cand"].transpose(2, 1, 0)),
        positions=(np.arange(batch, dtype=np.int32) * 3 + 7),
        head_rows=head_rows_of(built),
        geo=dict(geo, drafts=drafts),
    )


def run_tail(tables, vectors, policy, geo, eps, workdir):
    inputs = vectors["built"]["inputs"]
    table = score_table(vectors["cand"].transpose(2, 1, 0).copy(), inputs["raw"],
                        inputs["rms"], inputs["packed_q"], inputs["anchors"],
                        inputs["norm_w"], inputs["gate_bias"], vectors["head_rows"],
                        tables["tg"], tables["tsv"], geo["hidden"], geo["rank"], geo["width"])
    return run_fixture("serial_tail", geo, eps,
                       {"table": table, "cand": vectors["cand"],
                        "positions": vectors["positions"]},
                       {}, {}, vectors["built"]["fp8"], policy, workdir), table


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
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--draw-seed", type=int, default=20261122)
    parser.add_argument("--work", default="")
    parser.add_argument("--report", default="")
    arguments = parser.parse_args()

    geo = dict(hidden=arguments.hidden, rank=arguments.rank, code_size=arguments.code_size,
               width=arguments.width, batch=arguments.batch, vocab=arguments.vocab,
               drafts=arguments.width - 1, device=arguments.device)
    policy = dict(temp=arguments.temperature, topk=arguments.top_k, topp=arguments.top_p,
                  minp=arguments.min_p, seed=arguments.draw_seed)
    work = Path(arguments.work) if arguments.work else Path(tempfile.mkdtemp(prefix="p2tail-"))
    rows: list = []

    for index in range(arguments.cases):
        case = Case(f"serial_tail#{index}", arguments.seed + index, geo)
        vectors = tail_inputs(case, geo)
        inputs = vectors["built"]["inputs"]
        tg_bits, tsv_bits = vectors["built"]["tables"]["tg"], vectors["built"]["tables"]["tsv"]
        tg = nvfp4.encode_table(num.bf16_bits_to_f32(tg_bits).astype(np.float32))
        tsv = nvfp4.encode_table(num.bf16_bits_to_f32(tsv_bits).astype(np.float32))
        bf16 = {"tg": tg_bits, "tsv": tsv_bits}
        clean = {"tg": nvfp4.dequant_bits(tg), "tsv": nvfp4.dequant_bits(tsv)}

        control, table_a = run_tail(bf16, vectors, policy, geo, case.eps,
                                     work / f"case{index}-control")
        rerun, _ = run_tail(bf16, vectors, policy, geo, case.eps,
                             work / f"case{index}-control2")
        nvfp4_run, table_b = run_tail(clean, vectors, policy, geo, case.eps,
                                       work / f"case{index}-nvfp4")
        print(f"[{case.name}] table {table_a.shape} bf16 score range "
              f"[{table_a.min():.4g},{table_a.max():.4g}] | nvfp4 range "
              f"[{table_b.min():.4g},{table_b.max():.4g}] | policy {policy}", file=sys.stderr)

        for name in ("drafts", "idx", "proposal_q"):
            device_vs_device(rows, "A-format-invariance", "serial_tail", case.name,
                             control[name], rerun[name],
                             np.int32 if name != "proposal_q" else np.float32, name)
        for name in ("drafts", "idx", "proposal_q"):
            device_vs_device(rows, "B-bf16-vs-nvfp4", "serial_tail", case.name,
                             nvfp4_run[name], control[name],
                             np.int32 if name != "proposal_q" else np.float32, name)
        for mutant, kwargs in GATHER_MUTANTS.items():
            defect, _ = run_tail({"tg": nvfp4.dequant_bits(tg, **kwargs),
                                  "tsv": nvfp4.dequant_bits(tsv, **kwargs)},
                                 vectors, policy, geo, case.eps,
                                 work / f"case{index}-{mutant}")
            for name in ("drafts", "idx", "proposal_q"):
                device_vs_device(rows, f"C-teeth:{mutant}", "serial_tail", case.name,
                                 defect[name], control[name],
                                 np.int32 if name != "proposal_q" else np.float32, name)

    header = (f"{'kind':<26} {'op':<13} {'output':<11} {'n':>7} {'same':>7} {'diff':>6} "
              f"{'max|ulp|':>8} {'p99':>6}  verdict")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row.get('kind',''):<26} {row.get('op',''):<13} {row.get('output',''):<11} "
              f"{row.get('n',''):>7} {row.get('exact',''):>7} {row.get('mismatches',''):>6} "
              f"{row.get('max_abs_ulp',''):>8} {row.get('p99_abs_ulp',''):>6}  {row['verdict']}")
    if arguments.report:
        Path(arguments.report).write_text(json.dumps(rows, indent=1, default=str))
        print(f"\nwrote {arguments.report}")
    print(f"work: {work}")

    control = [row for row in rows if row["kind"] == "A-format-invariance"]
    bad = [row for row in control if row["verdict"] != "BIT-EXACT"]
    teeth = [row for row in rows if row["kind"].startswith("C-teeth")]
    draws = [row for row in teeth if row["output"] == "drafts"]
    weak = [row for row in draws if row["mismatches"] == 0]
    print(f"\ncontrol rows {len(control)} (not identical {len(bad)}) | teeth {len(teeth)} "
          f"(drafts rows {len(draws)}, unmoved {len(weak)})")
    for row in bad:
        print(f"  control not identical: {row['op']} {row['output']} {row['verdict']}")
    for row in weak:
        print(f"  tooth does not move the draws: {row['kind']}")
    return 0 if not bad and not weak else 1


if __name__ == "__main__":
    raise SystemExit(main())
