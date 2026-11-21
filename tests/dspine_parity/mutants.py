"""Deliberately-wrong reference variants (the harness's TEETH).

Each mutant is a real defect class the parity test must catch: a moved rounding
point, a dropped factor, a reversed chain, a tie-break flip. Running
`--mutant NAME` computes the reference WITH the defect and compares it to the
compiled op: a sound op + a working harness yields DIVERGENT, proving the
comparison can see that defect. A harness that only agrees with itself would
pass the mutant too - that is the failure mode this file exists to rule out.
"""

from __future__ import annotations

import numpy as np

import numeric as num
import reference as ref

Sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))


def _call(op, inputs, geo):
    if op == "prepare_read":
        return ref.prepare_read(inputs["residual"], geo["hidden"], geo["eps"])
    if op == "normalize_codes":
        return ref.normalize_codes(inputs["projected_bits"], inputs["anchors"],
                                   inputs["codebook_bits"], geo["code_size"],
                                   geo["width"], geo["eps"])
    if op == "message":
        return ref.message(inputs["packed_bits"], inputs["gate_bias_bits"],
                           geo["rank"], geo["width"], geo["batch"])
    if op == "finish":
        return ref.finish(inputs["residual_bits"], inputs["rms"],
                          inputs["delta_bits"], inputs["next_norm_bits"],
                          geo["hidden"], geo["width"], geo["eps"])
    if op == "serial_scores":
        return ref.serial_scores(inputs["cand"], inputs["raw_bits"], inputs["rms"],
                                 inputs["packed_q_bits"], inputs["anchors"],
                                 inputs["norm_w_bits"], inputs["gate_bias_bits"],
                                 inputs["head_rows"], inputs["tg_bits"],
                                 inputs["tsv_bits"], geo["hidden"], geo["rank"],
                                 geo["width"])
    if op == "serial_chain":
        return ref.serial_chain(inputs["cand"], inputs["idx"])
    raise SystemExit(f"no reference for op {op}")


def _mutate_prepare_read(clean, inputs, geo):
    # DEFECT: the RMS clamp/sqrt is dropped - rms = mean(raw^2) instead of
    # sqrt(max(mean, FLT_MIN)). (kernels.cu:82 vs bridge.py:119)
    raw = ref.bf16_to_f32(inputs["residual"]).reshape((geo["hidden"], -1), order="F").astype(np.float32)
    mean = (raw * raw).sum(axis=-1, dtype=np.float64) / geo["hidden"]
    rms = mean.astype(np.float32)
    normalized = raw / (rms[:, None] + np.float32(geo["eps"]))
    return ref.ReferenceResult({"normalized": num.bf16_view(normalized), "rms": rms},
                               {"normalized": "MUTANT rms_no_sqrt"})


def _mutate_normalize_codes(clean, inputs, geo):
    # DEFECT: the sqrt(C) scale factor is dropped (kernels.cu:110 vs bridge.py:131).
    projected = ref.bf16_to_f32(inputs["projected_bits"]).reshape((geo["code_size"], -1), order="F")
    norm = np.sqrt((projected * projected).sum(axis=0)).astype(np.float32)
    codes = projected / (norm[None, :] + np.float32(geo["eps"]))
    columns = codes.shape[1]
    batch = columns // geo["width"]
    for b in range(batch):
        row = ref.bf16_to_f32(inputs["codebook_bits"][int(inputs["anchors"][b])])
        codes[:, b * geo["width"]] = row
    return ref.ReferenceResult({"codes": num.bf16_view(codes)},
                               {"codes": "MUTANT no_sqrt_c"})


def _mutate_message(clean, inputs, geo):
    # DEFECT: the gate is rounded to bf16 BEFORE the value multiply - the
    # reference rounds only the product (bridge.py:146-148).
    packed = ref.bf16_to_f32(inputs["packed_bits"]).reshape((3 * geo["rank"], geo["width"],
                                                            geo["batch"]), order="F").astype(np.float32)
    bias = ref.bf16_to_f32(inputs["gate_bias_bits"]).astype(np.float32)
    rank, width = geo["rank"], geo["width"]
    q, s, v = packed[:rank], packed[rank:2 * rank], packed[2 * rank:]
    out = np.zeros((rank, width, geo["batch"]), dtype=np.float32)
    for t in range(1, width):
        gate = num.bf16_round(Sigmoid(q[:, t, :] + s[:, t - 1, :] + bias[:, None]))
        out[:, t, :] = gate * v[:, t - 1, :]
    return ref.ReferenceResult({"message": num.bf16_view(out)},
                               {"message": "MUTANT sigmoid_bf16"})


def _mutate_finish(clean, inputs, geo):
    # DEFECT: the intermediate bf16 rounding of the refined residual is skipped
    # (the norm is computed from the unrounded fp32 sum) - kernels.cu:147-150
    # round before the variance; this mutant does not.
    raw = ref.bf16_to_f32(inputs["residual_bits"]).reshape((geo["hidden"], geo["width"], -1), order="F")
    delta = ref.bf16_to_f32(inputs["delta_bits"]).reshape((geo["hidden"], geo["width"], -1), order="F")
    norm_w = ref.bf16_to_f32(inputs["next_norm_bits"]).astype(np.float32)
    rms_f = inputs["rms"].reshape((geo["width"], -1), order="F")
    mask = np.zeros((geo["width"], 1), dtype=np.float32)
    mask[1:, 0] = 1.0
    out = (raw + (rms_f[None, :, :] * delta) * mask[None, :, :]).astype(np.float32)
    variance = (out * out).sum(axis=0, dtype=np.float64) / geo["hidden"]
    inv = (1.0 / np.sqrt(variance.astype(np.float32) + np.float32(geo["eps"]))).astype(np.float32)
    y = num.bf16_round(out * inv[None, :, :]).astype(np.float32)
    normed = num.bf16_round(y * norm_w[:, None, None])
    return ref.ReferenceResult({"residual_out": num.bf16_view(out),
                                "normed": num.bf16_view(normed)},
                               {"normed": "MUTANT rounding_point"})


def _mutate_finish_no_refine(clean, inputs, geo):
    # DEFECT: the refinement term (rms * delta) is dropped entirely - the residual is
    # never updated (kernels.cu:143 vs bridge.py:165). residual_out = f, normed from f.
    raw = ref.bf16_to_f32(inputs["residual_bits"]).reshape((geo["hidden"], geo["width"], -1), order="F")
    norm_w = ref.bf16_to_f32(inputs["next_norm_bits"]).astype(np.float32)
    out = raw.astype(np.float32)
    variance = (out * out).sum(axis=0, dtype=np.float32) / np.float32(geo["hidden"])
    inv = (1.0 / np.sqrt(variance + np.float32(geo["eps"]))).astype(np.float32)
    y = num.bf16_round(out * inv[None, :, :]).astype(np.float32)
    normed = num.bf16_round(y * norm_w[:, None, None])
    return ref.ReferenceResult({"residual_out": num.bf16_view(out),
                                "normed": num.bf16_view(normed)},
                               {"residual_out": "MUTANT no_refine"})


def _mutate_serial_scores(clean, inputs, geo):
    # DEFECT: the rms factor is dropped from the score (kernels.cu:281 vs
    # propose.py:214) - the bridge refinement term vanishes.
    drafts, batch = inputs["cand"].shape[1], inputs["cand"].shape[2]
    raw = ref.bf16_to_f32(inputs["raw_bits"]).reshape((geo["hidden"], geo["width"], batch), order="F")
    q_all = ref.bf16_to_f32(inputs["packed_q_bits"]).reshape((geo["rank"], geo["width"], batch), order="F")
    norm_w = ref.bf16_to_f32(inputs["norm_w_bits"]).astype(np.float32)
    bias = ref.bf16_to_f32(inputs["gate_bias_bits"]).astype(np.float32)
    tg = ref.bf16_to_f32(inputs["tg_bits"]).astype(np.float32)
    tsv = ref.bf16_to_f32(inputs["tsv_bits"]).astype(np.float32)
    rank = geo["rank"]
    idx = np.zeros((16, drafts, batch), dtype=np.int32)
    for b in range(batch):
        for i in range(drafts):
            column = 1 + i
            base = ((raw[:, column, b] * norm_w)[:, None]
                    * inputs["head_rows"][inputs["cand"][:, i, b]].T).sum(axis=0).astype(np.float32)
            sid = (np.full(16, int(inputs["anchors"][b]), dtype=np.int64) if i == 0
                   else inputs["cand"][:, i - 1, b].astype(np.int64))
            rows = tsv[sid]
            m = Sigmoid(q_all[:, column, b][None, :] + rows[:, :rank] + bias[None, :]) * rows[:, rank:]
            scores = base[None, :] + (m @ tg[inputs["cand"][:, i, b]].T)
            idx[:, i, b] = np.argmax(scores, axis=-1).astype(np.int32)
    return ref.ReferenceResult({"idx": idx}, {"idx": "MUTANT rms_missing"})


def _mutate_serial_scores_base_missing(clean, inputs, geo):
    # DEFECT: the base_k term (dot over the full hidden) is dropped - the score is only
    # the bridge-refinement term (kernels.cu:281 vs propose.py:213-214). The argmax then
    # follows the R-dim dot, not the H-dim base, so the selected candidate changes.
    drafts, batch = inputs["cand"].shape[1], inputs["cand"].shape[2]
    q_all = ref.bf16_to_f32(inputs["packed_q_bits"]).reshape((geo["rank"], geo["width"], batch), order="F")
    bias = ref.bf16_to_f32(inputs["gate_bias_bits"]).astype(np.float32)
    tg = ref.bf16_to_f32(inputs["tg_bits"]).astype(np.float32)
    tsv = ref.bf16_to_f32(inputs["tsv_bits"]).astype(np.float32)
    rank = geo["rank"]
    idx = np.zeros((16, drafts, batch), dtype=np.int32)
    for b in range(batch):
        for i in range(drafts):
            column = 1 + i
            sid = (np.full(16, int(inputs["anchors"][b]), dtype=np.int64) if i == 0
                   else inputs["cand"][:, i - 1, b].astype(np.int64))
            rows = tsv[sid]
            m = Sigmoid(q_all[:, column, b][None, :] + rows[:, :rank] + bias[None, :]) * rows[:, rank:]
            scores = (m @ tg[inputs["cand"][:, i, b]].T)
            idx[:, i, b] = np.argmax(scores, axis=-1).astype(np.int32)
    return ref.ReferenceResult({"idx": idx}, {"idx": "MUTANT base_missing"})


def _mutate_serial_chain(clean, inputs, geo):
    # DEFECT: the chain is walked one step too far - the NEXT row's index is
    # consumed before reading the current row (propose.py:107-110 order).
    cand, idx = inputs["cand"], inputs["idx"]
    drafts, batch = cand.shape[1], cand.shape[2]
    out = np.zeros((drafts, batch), dtype=np.int32)
    for b in range(batch):
        j = int(idx[0, 0, b])
        for i in range(drafts):
            j = int(idx[j, (i + 1) % drafts, b])
            out[i, b] = cand[j, i, b]
    return ref.ReferenceResult({"drafts": out}, {"drafts": "MUTANT reversed"})


MUTANTS = {
    ("prepare_read", "rms_no_sqrt"): _mutate_prepare_read,
    ("normalize_codes", "no_sqrt_c"): _mutate_normalize_codes,
    ("message", "sigmoid_bf16"): _mutate_message,
    ("finish", "rounding_point"): _mutate_finish,
    ("finish", "no_refine"): _mutate_finish_no_refine,
    ("serial_scores", "rms_missing"): _mutate_serial_scores,
    ("serial_scores", "base_missing"): _mutate_serial_scores_base_missing,
    ("serial_chain", "reversed"): _mutate_serial_chain,
}


def apply_mutant(op, inputs, geo, mutant):
    clean = _call(op, inputs, geo)
    if not mutant or mutant == "none":
        return clean
    function = MUTANTS.get((op, mutant))
    if function is None:
        known = sorted(name for (op_name, name) in MUTANTS if op_name == op)
        raise SystemExit(f"no mutant '{mutant}' for {op}; known: {known}")
    return function(clean, inputs, geo)
