"""P0b reference side: Tencent's documented op math, evaluated in NumPy fp32 with
BF16 storage at exactly the reference's rounding points.

Every function cites the reference lines it transcribes. The compiled fork op is the
oracle; this module is the thing it must agree with.

  dspine_mlx/dspine_mlx/bridge.py               DSpineCodeBridge.__call__  (bridge family)
  dspine_mlx/dspine_mlx/sglang/kernels/reference.py  chain / serial chain
  dspine_mlx/dspine_mlx/propose.py              RefinementTail.greedy / _chain
  sglang_dspine/kernels/dspine.py               triton prepare_read/normalize_codes/
                                               feature_message/feature_finish
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from numeric import bf16_round, bf16_view, f32_to_bf16_bits

# torch.finfo(float32).tiny - the bridge read-RMS clamp floor (bridge.py:13, :119)
F32_TINY = np.float32(1.1754943508222875e-38)

Sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))


def decode_e4m3(byte: int) -> float:
    """OCP FP8 E4M3 (1/4/3, bias 7) - mirrors ops/common/quant/fp8.cuh:50 decode_e4m3."""
    sign = -1.0 if (byte >> 7) & 1 else 1.0
    exponent = (byte >> 3) & 0xF
    mantissa = byte & 0x7
    if exponent == 0:
        return sign * (mantissa / 8.0) * 2.0**-6
    return sign * (1.0 + mantissa / 8.0) * float(2.0**(exponent - 7))


def encode_e4m3(value: float) -> int:
    """Exact-only encoder: raises when `value` is not representable in E4M3."""
    if value == 0.0:
        return 0
    sign = 0x80 if value < 0 else 0
    magnitude = abs(value)
    exponent = math.floor(math.log2(magnitude))
    while exponent - 7 < 1:  # push into the normal range
        exponent += 1
    significand = magnitude / float(2.0**(exponent - 7))
    mantissa = (significand - 1.0) * 8.0
    if abs(mantissa - round(mantissa)) > 1e-9 or not 0 <= round(mantissa) <= 7:
        raise ValueError(f"value {value} is not exactly E4M3 representable")
    if not 1 <= exponent <= 14:
        raise ValueError(f"exponent {exponent} out of E4M3 range")
    return sign | (exponent << 3) | int(round(mantissa))


def head_matrix_from_codes(codes: np.ndarray, scales_f32: np.ndarray) -> np.ndarray:
    """Dequantize a FP8_E4M3FN_ROW_BF16 head exactly as kernels.cu:225-227 does:
    code = bf16(decode_e4m3(byte) * row_scale)."""
    flat = codes.view(np.uint8).astype(np.uint32)
    decoded = np.array([decode_e4m3(int(b)) for b in flat.ravel()], dtype=np.float32)
    decoded = decoded.reshape(flat.shape)
    scaled = decoded * scales_f32[:, None]
    return bf16_round(scaled)


@dataclass
class ReferenceResult:
    outputs: dict          # name -> np array (uint16 bf16 bits / int32)
    notes: dict            # name -> str, the rounding point that op applies


# --- 1) dspine_prepare_read -------------------------------------------------------------
# bridge.py:115-121 + sglang_dspine/kernels/dspine.py:12-33
#   rms        = sqrt(max(mean(fp32(raw)^2), F32_TINY))
#   normalized = bf16(fp32(raw) / (rms + eps))
def prepare_read(residual_bits: np.ndarray, hidden: int, eps: float) -> ReferenceResult:
    raw = bf16_to_f32(residual_bits).reshape((hidden, -1), order="F")
    f32 = raw.astype(np.float32)
    squares = f32 * f32
    # One RMS per column over the hidden axis (kernels.cu:76-84). The reference
    # (triton tl.sum / mlx mx.mean) accumulates in FP32, so we do too - an fp64
    # accumulation is *more* precise than the reference and would not be comparable.
    mean = np.ascontiguousarray(squares, dtype=np.float32).sum(axis=0, dtype=np.float32)
    mean = np.maximum((mean / np.float32(hidden)), F32_TINY)
    rms = np.sqrt(mean).astype(np.float32)
    normalized = f32 / (rms[None, :] + np.float32(eps))
    return ReferenceResult(
        {"normalized": bf16_view(normalized), "rms": rms.astype(np.float32)},
        {"normalized": "bf16(f32(raw) / (rms + eps)); rms fp32, clamp FLT_MIN",
         "rms": "fp32 sqrt(max(mean(raw^2), FLT_MIN))"},
    )


# --- 2) dspine_normalize_codes ---------------------------------------------------------
# bridge.py:123-136 + sglang_dspine/kernels/dspine.py:36-59
#   codes    = bf16(sqrt(C) * fp32(projected) / (sqrt(sum(fp32(projected)^2)) + eps))
#   codes[t=0] = bf16(sqrt(C) * fp32(codebook[anchor]))
def normalize_codes(projected_bits: np.ndarray, anchors: np.ndarray, codebook_bits: np.ndarray,
                    code_size: int, width: int, eps: float) -> ReferenceResult:
    projected = bf16_to_f32(projected_bits).reshape((code_size, -1), order="F").astype(np.float32)
    sqrt_c = np.float32(math.sqrt(code_size))
    # FP32 accumulation (triton tl.sum / mlx), then the kernel's exact op order:
    # scale = sqrt_c / (norm + eps); codes = bf16(p * scale)  (kernels.cu:110-117).
    norm = np.sqrt((projected * projected).sum(axis=0, dtype=np.float32)).astype(np.float32)
    scale = sqrt_c / (norm + np.float32(eps))
    codes = projected * scale[None, :]
    columns = codes.shape[1]
    batch = columns // width
    for batch_index in range(batch):
        anchor = int(anchors[batch_index])
        row = bf16_to_f32(codebook_bits[anchor]).astype(np.float32)
        codes[:, batch_index * width] = bf16_round(sqrt_c * row).astype(np.float32)
    return ReferenceResult(
        {"codes": bf16_view(codes)},
        {"codes": "bf16(sqrt(C)*p/(||p||+eps)); anchor column = bf16(sqrt(C)*codebook[a])"},
    )


# --- 3) dspine_message ----------------------------------------------------------------
# bridge.py:140-148 + sglang_dspine/kernels/dspine.py:61-75
#   message[t] = bf16(sigmoid(f32(q[t]) + f32(s[t-1]) + bias) * f32(v[t-1])), message[0] = 0
def message(packed_bits: np.ndarray, gate_bias_bits: np.ndarray, rank: int, width: int,
            batch: int) -> ReferenceResult:
    packed = bf16_to_f32(packed_bits).reshape((3 * rank, width, batch), order="F").astype(np.float32)
    bias = bf16_to_f32(gate_bias_bits).astype(np.float32)
    q, s, v = packed[:rank], packed[rank:2 * rank], packed[2 * rank:]
    out = np.zeros((rank, width, batch), dtype=np.float32)
    for t in range(1, width):
        gate = Sigmoid(q[:, t, :] + s[:, t - 1, :] + bias[:, None])
        out[:, t, :] = gate * v[:, t - 1, :]
    return ReferenceResult(
        {"message": bf16_view(out)},
        {"message": "bf16(sigmoid(q_t + s_{t-1} + bias) * v_{t-1}); col 0 = 0; "
                    "fp32 sigmoid 1/(1+exp(-x))"},
    )


# --- 4) dspine_finish -----------------------------------------------------------------
# bridge.py:158-172 + sglang_dspine/kernels/dspine.py:77-110
#   out    = bf16(fp32(raw) + rms * fp32(delta) * (t != 0)); residual <- out
#   y      = bf16(fp32(out) * rsqrt(mean(fp32(out)^2) + eps))
#   normed = bf16(fp32(y) * fp32(next_norm))
def finish(residual_bits: np.ndarray, rms: np.ndarray, delta_bits: np.ndarray,
            next_norm_bits: np.ndarray, hidden: int, width: int,
            eps: float) -> ReferenceResult:
    raw = bf16_to_f32(residual_bits).reshape((hidden, width, -1), order="F").astype(np.float32)
    delta = bf16_to_f32(delta_bits).reshape((hidden, width, -1), order="F").astype(np.float32)
    norm_w = bf16_to_f32(next_norm_bits).astype(np.float32)
    rms_f = rms.reshape((width, -1), order="F")
    mask = np.zeros((width, 1), dtype=np.float32)
    mask[1:, 0] = 1.0
    out = raw + (rms_f[None, :, :] * delta) * mask[None, :, :]
    out = bf16_round(out).astype(np.float32)
    # FP32 accumulation (kernels.cu:155-160). The kernel uses the rsqrtf intrinsic
    # (approximate); we use 1/sqrt (correctly rounded) - the <=2-ulp intrinsic gap is
    # part of the documented EQUIVALENT-ORDER band, not a defect.
    variance = (out * out).sum(axis=0, dtype=np.float32) / np.float32(hidden)
    inv = (1.0 / np.sqrt(variance + np.float32(eps))).astype(np.float32)
    y = bf16_round(out * inv[None, :, :]).astype(np.float32)
    normed = bf16_round(y * norm_w[:, None, None])
    return ReferenceResult(
        {"residual_out": bf16_view(out), "normed": bf16_view(normed)},
        {"residual_out": "bf16(f + rms*delta*(t!=0)) (in-place refine)",
         "normed": "bf16(bf16(out*rsqrt(mean(out^2)+eps)) * next_norm); fp32 rsqrt"},
    )


# --- 5) dspine_serial_scores ---------------------------------------------------------
# propose.py:191-217 (RefinementTail.greedy) + sglang/kernels/reference.py:27-69 chain table
#   base_k = sum_h(raw_h * norm_w_h * head_h[cand_k])
#   sid_j  = (i == 0 ? anchor : cand[i-1, j])
#   m_j    = sigmoid(q + tsv[sid_j][:R] + bias) * tsv[sid_j][R:]
#   score  = base_k + rms * dot(m_j, tg[cand_k]);  idx_j = argmax_k score
def serial_scores(cand: np.ndarray, raw_bits: np.ndarray, rms: np.ndarray,
                  packed_q_bits: np.ndarray, anchors: np.ndarray, norm_w_bits: np.ndarray,
                  gate_bias_bits: np.ndarray, head_rows: np.ndarray, tg_bits: np.ndarray,
                  tsv_bits: np.ndarray, hidden: int, rank: int, width: int) -> ReferenceResult:
    drafts, batch = cand.shape[1], cand.shape[2]
    raw = bf16_to_f32(raw_bits).reshape((hidden, width, batch), order="F").astype(np.float32)
    q_all = bf16_to_f32(packed_q_bits).reshape((rank, width, batch), order="F").astype(np.float32)
    norm_w = bf16_to_f32(norm_w_bits).astype(np.float32)
    bias = bf16_to_f32(gate_bias_bits).astype(np.float32)
    tg = bf16_to_f32(tg_bits).astype(np.float32)
    tsv = bf16_to_f32(tsv_bits).astype(np.float32)
    rms_w = rms.reshape((width, batch), order="F")

    idx = np.zeros((16, drafts, batch), dtype=np.int32)
    for b in range(batch):
        for i in range(drafts):
            column = 1 + i
            raw_row = raw[:, column, b] * norm_w
            base = (raw_row[:, None] * head_rows[cand[:, i, b]].T).sum(axis=0)
            base = base.astype(np.float32)
            sid = (np.full(16, int(anchors[b]), dtype=np.int64) if i == 0
                   else cand[:, i - 1, b].astype(np.int64))
            q = q_all[:, column, b]
            rows = tsv[sid]                                     # [K, 2R]
            m = Sigmoid(q[None, :] + rows[:, :rank] + bias[None, :]) * rows[:, rank:]
            tg_cand = tg[cand[:, i, b]]                          # [K, R]
            scores = base[None, :] + np.float32(rms_w[column, b]) * (m @ tg_cand.T)
            idx[:, i, b] = np.argmax(scores, axis=-1).astype(np.int32)
    return ReferenceResult(
        {"idx": idx},
        {"idx": "argmax_k(base_k + rms*dot(m_j, tg_k)); fp32 reductions, ties -> lowest k"},
    )


# --- 6) dspine_serial_chain ---------------------------------------------------------
# propose.py:102-111 (_chain) == sglang/kernels/reference.py:13-24 (chain)
#   j = idx[0, 0]; drafts[i] = cand[i, j]; j = idx[i+1, j]
def serial_chain(cand: np.ndarray, idx: np.ndarray) -> ReferenceResult:
    # cand/idx are (16, drafts, batch) = (slot, row, batch); the walk (propose.py:102-111,
    # reference.py:13-24) is j = idx[0,0]; drafts[i] = cand[i, j]; j = idx[i+1, j] in the
    # reference's (row, slot) order, which is cand[j, i, b] / idx[j, i+1, b] here.
    drafts, batch = cand.shape[1], cand.shape[2]
    out = np.zeros((drafts, batch), dtype=np.int32)
    for b in range(batch):
        j = int(idx[0, 0, b])
        for i in range(drafts):
            out[i, b] = cand[j, i, b]
            if i + 1 < drafts:
                j = int(idx[j, i + 1, b])
    return ReferenceResult({"drafts": out}, {"drafts": "integer chain walk, no rounding"})


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    from numeric import bf16_bits_to_f32
    return bf16_bits_to_f32(bits)
