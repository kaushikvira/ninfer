#!/usr/bin/env python3
"""Generates tests/dspine_serial_tail/fixture.h - the committed reference vectors for the
DSpine serial tail (host core + device op).

Reference semantics are Tencent's, transcribed from the sources P0b/this stage cite:

  dspine_mlx/dspine_mlx/propose.py:73-90   filter_probs  (joint top-k / top-p + min-p)
  dspine_mlx/dspine_mlx/propose.py:114-152 serial_sample_chain (softmax(inv_temp), CDF draw)
  dspine_mlx/dspine_mlx/propose.py:102-111 _chain (greedy walk)
  src/ops/sampling/kernels/random.cuh      sampling_uniform (the engine's draw stream)

`propose.py` runs on MLX fp32; this file evaluates the same expressions in NumPy fp32. The
fixture is generated once and committed so the C++ test needs no Python at test time. Scores
are drawn far from every filter boundary (checked below) so a one-ulp softmax difference
cannot flip a decision - what the test pins is the POLICY, not an exp() implementation.

Usage:  python3 tests/dspine_serial_tail/make_fixture.py [--check]
        --check regenerates in memory and verifies the committed header is identical.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HEADER = HERE / "fixture.h"
PARITY_REF = Path("/tmp/ninfer-dspine-parity/tests/dspine_parity")

SLOTS = 16
MASK64 = (1 << 64) - 1


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return (x ^ (x >> 31)) & MASK64


def sampling_uniform(seed: int, position: int, purpose: int, sub: int) -> float:
    """random.cuh:14-28 in exact 64-bit integer arithmetic."""
    key = seed & MASK64
    key = splitmix64(key ^ (((position & 0xFFFFFFFF) * 0xD1B54A32D192ED03) & MASK64))
    key = splitmix64(key ^ (((purpose & 0xFFFFFFFF) << 21) & MASK64)
                     ^ ((sub * 0x2545F4914F6CDD1D) & MASK64))
    bits = (key >> 40) & 0xFFFFFF
    return np.float32(bits) * np.float32(1.0 / 16777216.0)


def softmax(values: np.ndarray) -> np.ndarray:
    scaled = values.astype(np.float32)
    peak = np.max(scaled)
    w = np.exp(scaled - peak).astype(np.float32)
    return (w / np.sum(w, dtype=np.float32)).astype(np.float32)


def filter_probs(p: np.ndarray, top_k: int, top_p: float, min_p: float):
    """propose.py:73-90, evaluated on one [K] probability row."""
    n = p.shape[0]
    lane = np.arange(n)
    p_c = p[:, None]
    p_d = p[None, :]
    ahead = (p_d > p_c) | ((p_d == p_c) & (lane[None, :] < lane[:, None]))
    rank = np.sum(ahead.astype(np.int32), axis=1)
    before = np.sum(np.where(ahead, p_d, np.float32(0.0)), axis=1, dtype=np.float32)
    keep = (rank < int(top_k)) & (before <= np.float32(top_p))
    if min_p > 0:
        keep = keep & (p >= np.float32(min_p) * np.max(p))
    w = np.where(keep, p, np.float32(0.0)).astype(np.float32)
    q = (w / np.sum(w, dtype=np.float32)).astype(np.float32)
    return q, keep


def serial_sample_chain(table: np.ndarray, cand: np.ndarray, policy, seed: int, position: int):
    """propose.py:114-152. table: [drafts, slots(sender), slots(cand)]; cand: [slots, drafts].

    Every (slot, sender) row gets its filtered distribution and chosen index, matching the C++
    core's output shape (the cheap twin of ops::dspine_serial_scores' idx); the walk consumes
    only the sender row it reaches.
    """
    drafts = table.shape[0]
    temperature, top_k, top_p, min_p = policy
    toks = np.zeros(drafts, dtype=np.int32)
    q_rows = np.zeros((SLOTS, drafts), dtype=np.float32)
    idx = np.zeros((SLOTS, drafts), dtype=np.int32)
    if temperature <= 0.0:
        # Greedy degenerate: argmax over the row, lowest slot index on ties (_chain/_greedy).
        idx[:, :] = greedy_idx(table, cand)
        sender = 0
        for i in range(drafts):
            choice = int(idx[sender, i])
            toks[i] = cand[choice, i]
            q_rows[:, i] = 0.0
            q_rows[choice, i] = 1.0
            sender = choice
        return toks, q_rows, idx
    inv_temp = np.float32(1.0) / np.float32(temperature)
    rows = []
    for sender in range(SLOTS):
        row_q = np.zeros((SLOTS, drafts), dtype=np.float32)
        row_keep = np.zeros((SLOTS, drafts), dtype=bool)
        for i in range(drafts):
            p = softmax(table[i, sender] * inv_temp)
            q, keep = filter_probs(p, top_k, top_p, min_p)
            row_q[:, i] = q
            row_keep[:, i] = keep
        rows.append((row_q, row_keep))
    sender = 0
    for i in range(drafts):
        for s in range(SLOTS):
            q = rows[s][0][:, i]
            keep = rows[s][1][:, i]
            cdf = np.cumsum(q, dtype=np.float32)
            threshold = np.float32(sampling_uniform(seed, position, 5, i)) * cdf[-1]
            choice = int(np.sum(cdf <= threshold))
            last = int(np.max(np.where(keep, np.arange(SLOTS), 0)))
            idx[s, i] = min(choice, last)
        # Only the sender the walk reaches generates the draft, so its row is the q the
        # acceptor verifies against.
        choice = int(idx[sender, i])
        q_rows[:, i] = rows[sender][0][:, i]
        toks[i] = cand[choice, i]
        sender = choice
    return toks, q_rows, idx


def greedy_idx(table: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """The per-(sender, slot) chosen candidate index the tail reports for greedy rows."""
    drafts = table.shape[0]
    idx = np.zeros((SLOTS, drafts), dtype=np.int32)
    for i in range(drafts):
        for sender in range(SLOTS):
            idx[sender, i] = int(np.argmax(table[i, sender]))
    return idx


def margin_ok(table: np.ndarray, policy) -> bool:
    """True when no filter decision sits within 1e-5 of a boundary (see module docstring)."""
    temperature, top_k, top_p, min_p = policy
    if temperature <= 0.0:
        return True
    inv_temp = np.float32(1.0) / np.float32(temperature)
    for i in range(table.shape[0]):
        for sender in range(SLOTS):
            p = softmax(table[i, sender] * inv_temp)
            n = p.shape[0]
            lane = np.arange(n)
            ahead = (p[None, :] > p[:, None]) | ((p[None, :] == p[:, None]) &
                                                 (lane[None, :] < lane[:, None]))
            rank = np.sum(ahead.astype(np.int32), axis=1)
            before = np.sum(np.where(ahead, p[None, :], np.float32(0.0)), axis=1, dtype=np.float32)
            # Only a decision that moves meaningful mass can flip the round: a boundary at
            # p ~ 1e-9 changes nothing downstream, and pinning such tails would make the
            # fixture reject every well-behaved draw.
            for k in range(n):
                if p[k] <= 1e-6:
                    continue
                if top_p < 1.0 and abs(before[k] - np.float32(top_p)) < 1e-5:
                    return False
                if top_k < SLOTS:
                    # The top-k cut is exact; what can flip is the ordering across it, so
                    # require the mass at the cut and the next candidate to be separated.
                    ordered = np.sort(p)[::-1]
                    if abs(ordered[top_k - 1] - ordered[top_k]) < 1e-5:
                        return False
                if min_p > 0 and abs(p[k] - np.float32(min_p) * np.max(p)) < 1e-6:
                    return False
    return True


def bits(value: float) -> int:
    return int(np.float32(value).view(np.uint32))


def c_floats(name: str, array: np.ndarray) -> str:
    flat = np.asarray(array, dtype=np.float32).ravel()
    rows = []
    for start in range(0, flat.size, 6):
        chunk = ", ".join(f"0x{bits(v):08x}u" for v in flat[start:start + 6])
        rows.append("    " + chunk + ",")
    return f"inline constexpr std::uint32_t {name}[{flat.size}] = {{\n" + "\n".join(rows) + "\n};\n"


def c_ints(name: str, array: np.ndarray) -> str:
    flat = np.asarray(array, dtype=np.int32).ravel()
    rows = []
    for start in range(0, flat.size, 16):
        rows.append("    " + ", ".join(str(int(v)) for v in flat[start:start + 16]) + ",")
    return f"inline constexpr std::int32_t {name}[{flat.size}] = {{\n" + "\n".join(rows) + "\n};\n"


def c_bools(name: str, array: np.ndarray) -> str:
    flat = np.asarray(array, dtype=bool).ravel()
    rows = []
    for start in range(0, flat.size, 32):
        rows.append("    " + ", ".join("1" if v else "0" for v in flat[start:start + 32]) + ",")
    return f"inline constexpr std::uint8_t {name}[{flat.size}] = {{\n" + "\n".join(rows) + "\n};\n"


def c_u64(name: str, values) -> str:
    body = ", ".join(f"{int(v)}ull" for v in values)
    return f"inline constexpr std::uint64_t {name}[{len(values)}] = {{{body}}};\n"


def chain_reference_match(table, cand, toks) -> str:
    """Provenance: the P0b reference (tests/dspine_parity/reference.py) walks the same chain."""
    sys.path.insert(0, str(PARITY_REF))
    try:
        import reference  # type: ignore  # noqa: PLC0415
    except Exception as error:  # pragma: no cover - generator-only convenience
        return f"not cross-checked ({error})"
    drafts = table.shape[0]
    # reference index order is (candidate-row, draft-slot, batch): idx[j, i, b] holds the
    # chosen candidate of slot i for sender j (reference.py serial_chain walk).
    idx = np.zeros((SLOTS, drafts, 1), dtype=np.int32)
    for i in range(drafts):
        for sender in range(SLOTS):
            idx[sender, i, 0] = int(np.argmax(table[i, sender]))
    cand3 = np.zeros((SLOTS, drafts, 1), dtype=np.int32)
    for i in range(drafts):
        for k in range(SLOTS):
            cand3[k, i, 0] = cand[k, i]
    walks = reference.serial_chain(cand3, idx).outputs["drafts"][:, 0]
    if not np.array_equal(walks, toks):
        raise SystemExit(f"greedy chain disagrees with the P0b reference: {walks} vs {toks}")
    return "matches tests/dspine_parity/reference.py serial_chain (P0b BIT-EXACT oracle)"


CASES = [
    # name, drafts, batch, temperature, top_k, top_p, min_p, seeds, positions
    ("prod_t09_minp005", 15, 1, 0.9, 20, 1.0, 0.05, [0x5DEECE66D], [777]),
    ("greedy_t0", 15, 1, 0.0, 20, 1.0, 0.0, [0x5DEECE66D], [777]),
    ("topk4_t10", 15, 1, 1.0, 4, 1.0, 0.0, [11], [1024]),
    ("topp08_t12", 15, 1, 1.2, 20, 0.8, 0.0, [12], [2048]),
    ("minp02_t07", 15, 1, 0.7, 20, 1.0, 0.2, [13], [4096]),
    ("lowtemp_t005", 15, 1, 0.05, 20, 1.0, 0.0, [14], [8192]),
    ("prod_batch2_drafts3", 3, 2, 0.9, 20, 1.0, 0.05, [21, 22], [300, 301]),
]


def main() -> int:
    rng = np.random.default_rng(20261122)
    body = [
        "// Generated by tests/dspine_serial_tail/make_fixture.py - do not edit.",
        "// Reference: dspine_mlx/dspine_mlx/propose.py (filter_probs, serial_sample_chain);",
        "// draw stream: src/ops/sampling/kernels/random.cuh. See the generator for provenance.",
        "#pragma once",
        "",
        "#include <cstdint>",
        "",
        "namespace ninfer::test::dspine_serial_tail {",
        "",
        f"inline constexpr std::int32_t kSlots = {SLOTS};",
        "",
    ]
    provenance = []
    for name, drafts, batch, temperature, top_k, top_p, min_p, seeds, positions in CASES:
        policy = (temperature, top_k, top_p, min_p)
        # Logit-like scores: N(0, 3) per (slot, sender, candidate). The tail only ever reads
        # one sender row per slot, but a full table keeps the fixture reusable by the device
        # op test, which exercises every sender.
        table = None
        for _attempt in range(64):
            proposal = rng.normal(0.0, 3.0, size=(drafts, SLOTS, SLOTS, batch)).astype(np.float32)
            if margin_ok(proposal[:, :, :, 0], policy):
                table = proposal
                break
        if table is None:
            raise SystemExit(f"case {name}: no draw clear of every filter boundary")
        # Distinct ids inside each (slot, draft, batch) candidate list, as the sparse acceptor
        # requires; ids are drawn from a vocabulary wider than the candidate set.
        cand = np.zeros((SLOTS, drafts, batch), dtype=np.int32)
        for i in range(drafts):
            for b in range(batch):
                cand[:, i, b] = rng.choice(np.arange(128), size=SLOTS, replace=False)
        toks = np.zeros((drafts, batch), dtype=np.int32)
        q_rows = np.zeros((SLOTS, drafts, batch), dtype=np.float32)
        idx = np.zeros((SLOTS, drafts, batch), dtype=np.int32)
        for b in range(batch):
            t, q, i = serial_sample_chain(table[:, :, :, b], cand[:, :, b], policy, seeds[b],
                                          positions[b])
            toks[:, b] = t
            q_rows[:, :, b] = q
            idx[:, :, b] = i
        uniforms = [
            [float(sampling_uniform(seeds[b], positions[b], 5, i)) for i in range(drafts)]
            for b in range(batch)
        ]
        body.append(f"// --- case {name}: T={temperature} top_k={top_k} top_p={top_p} "
                    f"min_p={min_p}, drafts={drafts}, batch={batch}")
        body.append(c_floats(f"k{name}Table", table))
        body.append(c_ints(f"k{name}Cand", cand))
        body.append(c_ints(f"k{name}Drafts", toks))
        body.append(c_floats(f"k{name}Q", q_rows))
        body.append(c_ints(f"k{name}Idx", idx))

        body.append(c_u64(f"k{name}Seed", seeds))
        body.append(c_ints(f"k{name}Position", np.array(positions, dtype=np.int32)))
        body.append(c_floats(f"k{name}Uniform", np.array(uniforms, dtype=np.float32)))
        body.append("")
        body.append(f"inline constexpr std::int32_t k{name}DraftsCount = {drafts};")
        body.append(f"inline constexpr std::int32_t k{name}Batch = {batch};")
        body.append(f"inline constexpr float k{name}Temperature = {float(temperature)!r}f;")
        body.append(f"inline constexpr std::int32_t k{name}TopK = {top_k};")
        body.append(f"inline constexpr float k{name}TopP = {float(top_p)!r}f;")
        body.append(f"inline constexpr float k{name}MinP = {float(min_p)!r}f;")
        body.append("")
        if policy[0] <= 0.0:
            provenance.append(f"//   {name}: {chain_reference_match(table[:, :, :, 0], cand[:, :, 0], toks[:, 0])}")
    body.append("} // namespace ninfer::test::dspine_serial_tail")
    header = "\n".join(body) + "\n"
    if provenance:
        header = header.replace(
            "inline constexpr std::int32_t kSlots = 16;",
            "inline constexpr std::int32_t kSlots = 16;\n//\n// Greedy-case provenance:\n"
            + "\n".join(provenance),
        )

    if "--check" in sys.argv:
        if HEADER.read_text() != header:
            print(f"{HEADER} is stale - re-run without --check", file=sys.stderr)
            return 1
        print(f"{HEADER} is current")
        return 0
    HEADER.write_text(header)
    print(f"wrote {HEADER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
