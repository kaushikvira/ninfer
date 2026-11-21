"""P2 stage 1: the DSpine **draft forward** as a host twin of the engine path.

Every step is one op the engine runs (``STEPS`` names the device symbol):

  target taps -> ``dspine/feature_projection`` (linear) -> ``dspine/context_norm`` (rmsnorm)
  -> [draft layers] block embed + causal self-attention over (context ++ block)
     + SiLU-gated MLP, each layer followed by exactly one code-bridge write
       (``dspine_prepare_read`` -> read linear -> ``dspine_normalize_codes``
        -> packed linear -> ``dspine_message`` -> write linear -> ``dspine_finish``)
  -> proposal rows (the last write's fused ``dspine/final_norm``)
  -> the target ``lm_head`` linear -> per-slot BF16 logits + I32 top-k candidates.

``dspine_serial_scores`` / ``dspine_serial_chain`` are the sibling stage (the serial
tail, ``/tmp/ninfer-dspine-p2b``); what it consumes is ``Proposal`` here ==
``ninfer::models::dspine::Proposal`` in ``include/ninfer/models/dspine_proposal.h``.

Numerics — plan D-18 bans hand CUDA transcriptions, so the six bridge steps reuse the
**validated P0b reference math** (``tests/dspine_parity/reference.py``, cited per
function); the layer/linear/rmsnorm steps mirror the shipped op contracts:

* ``ops/linear``: "Activations and output are BF16", ``LinearPolicy::A16Only`` —
  BF16 in, BF16 out, fp32 math, fp32 accumulation (the P0b EQUIVALENT-ORDER band);
* ``ops/rmsnorm``: fp32 ``1/sqrt(mean(x^2)+eps)``, bf16 storage, no unit offset;
* bridge tables are dequantized per their **own** encoding — BF16 direct, NVFP4
  ``decode(q)*decode(s)/d_w`` — the path P0b recorded UNMEASURED and D-20 made
  load-bearing for the 1.3157 GiB sizing;
* the draft LM head is the target head dequantized and requantized to the geometry the
  V3 tower ships (``q8_g32_fp16`` / ``AllowA8``); ``fp8_e4m3fn_row_bf16`` (the DFlash2
  geometry) is supported too.  Inside the engine the head is the already-resident
  ``text/output_head``, so this choice belongs to the host twin, not to the device path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from tests.dspine_forward import numeric as num

F32_TINY = np.float32(1.1754943508222875e-38)  # torch.finfo(fp32).tiny — bridge.py:13
Sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))

#: The forward in engine-op order (the digest's op list, generated, not typed).
STEPS: tuple[tuple[str, str], ...] = (
    ("taps", "target hidden taps, concatenated in target_layer_ids order"),
    ("context", "linear(dspine/feature_projection) + rmsnorm(dspine/context_norm)"),
    ("embed", "embedding(block ids) -> the residual stream"),
    ("layer.input_norm", "rmsnorm"),
    ("layer.attention", "linear_split(qkv) + rmsnorm(q/k, head_dim) + rope + "
                        "sliding_window_attention(context ++ block)"),
    ("layer.attention.output", "linear + residual add"),
    ("layer.mlp", "rmsnorm + linear_swiglu(gate_up) + linear_add(down)"),
    ("bridge.read", "ops::dspine_prepare_read + linear(layer_read_weight)"),
    ("bridge.codes", "ops::dspine_normalize_codes (codebook gather, NVFP4|BF16)"),
    ("bridge.packed", "linear([gate; predecessor_gate; prev])"),
    ("bridge.message", "ops::dspine_message"),
    ("bridge.write", "linear(feature_write_weights[layer])"),
    ("bridge.finish", "ops::dspine_finish (residual refine + fused next-norm + stash)"),
    ("proposal.logits", "linear(text/output_head) over the proposal rows"),
    ("proposal.candidates", "top-k ids per slot (the serial tail's cand tensor)"),
)


@dataclass
class Proposal:
    """Host mirror of ``ninfer::models::dspine::Proposal`` (see the header)."""

    logits: np.ndarray        # BF16 storage bits, [V, drafts, B]
    candidates: np.ndarray    # I32, [head_topk, drafts, B]
    anchors: np.ndarray       # I32, [B]
    hidden: np.ndarray        # BF16 storage bits, [H, drafts, B]
    raw: np.ndarray           # BF16 storage bits, [H, width, B] (last-write stash)
    rms: np.ndarray           # FP32, [width, B]
    packed_q: np.ndarray      # BF16 storage bits, [R, width, B]
    shape_note: str = ""


# ------------------------------------------------------------------ bf16 plumbing


def f32(bits: np.ndarray) -> np.ndarray:
    """bf16 storage bits -> fp32 values (exact widening)."""

    return num.bf16_bits_to_f32(np.asarray(bits, dtype=np.uint16)).astype(np.float32)


def rmsnorm(rows: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """``ops::rmsnorm`` over axis 0: fp32 math, bf16 storage, no unit offset."""

    f32_rows = np.asarray(rows, dtype=np.float32)
    variance = np.ascontiguousarray(f32_rows * f32_rows, dtype=np.float32).sum(
        axis=0, dtype=np.float32
    ) / np.float32(f32_rows.shape[0])
    inv = (1.0 / np.sqrt(variance + np.float32(eps))).astype(np.float32)
    return num.bf16_round(f32_rows * inv[None, :] * weight.astype(np.float32)[:, None])


def linear(x_bits: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """``ops::linear``: BF16 in / BF16 out, fp32 math, fp32 accumulation."""

    return num.bf16_view(
        np.ascontiguousarray(np.asarray(weight, dtype=np.float32), dtype=np.float32)
        @ np.ascontiguousarray(f32(x_bits), dtype=np.float32)
    )


def swiglu(hidden_bits: np.ndarray, gate_up: np.ndarray, intermediate: int) -> np.ndarray:
    """``ops::linear_swiglu``: silu(gate @ x) * (up @ x), bf16 in/out, fp32 math."""

    gate = np.ascontiguousarray(np.asarray(gate_up[:intermediate], dtype=np.float32),
                                dtype=np.float32) @ f32(hidden_bits)
    up = np.ascontiguousarray(np.asarray(gate_up[intermediate:], dtype=np.float32),
                              dtype=np.float32) @ f32(hidden_bits)
    return num.bf16_round(gate * Sigmoid(gate) * up)


def rope(values: np.ndarray, positions: np.ndarray, head_dim: int,
          theta: float) -> np.ndarray:
    """``ops::rope``: rotate_half, absolute positions, bf16 in/out.

    ``values`` is ``[heads * head_dim, tokens]`` and ``positions`` the ``[tokens]``
    absolute position of each column.  rotate_half pairs row ``r`` with row
    ``r + head_dim/2`` of the same head and rotates them by
    ``position * theta^(-2r/head_dim)`` (the convention the Swift/Qwen tower ships:
    ``inv_freq[i] = theta^(-i / (head_dim/2))``, one frequency per pair).
    """

    source = np.array(f32(values), dtype=np.float32)
    flat = np.asarray(positions).reshape(-1)
    half = head_dim // 2
    heads = source.shape[0] // head_dim
    if source.shape[1] != flat.shape[0]:
        raise ValueError(
            f"rope: {source.shape[1]} value columns for {flat.shape[0]} positions"
        )
    inv_freq = np.exp(
        -np.arange(half, dtype=np.float64) * (math.log(theta) / half)
    )
    angle = np.ascontiguousarray(
        flat[None, :].astype(np.float64) * inv_freq[:, None], dtype=np.float64
    )                                                                # [half, tokens]
    cos = np.ascontiguousarray(np.cos(angle), dtype=np.float32)
    sin = np.ascontiguousarray(np.sin(angle), dtype=np.float32)
    out = np.zeros(source.shape, dtype=np.float32)
    for head in range(heads):
        band = slice(head * head_dim, (head + 1) * head_dim)
        first = np.ascontiguousarray(source[band][:half], dtype=np.float32)
        second = np.ascontiguousarray(source[band][half:], dtype=np.float32)
        out[band][:half] = first * cos - second * sin
        out[band][half:] = second * cos + first * sin
    return num.bf16_round(out)


def softmax_attention(query: np.ndarray, key: np.ndarray, value: np.ndarray,
                      mask: np.ndarray, head_dim: int, scale: float,
                      kv_heads: int | None = None) -> np.ndarray:
    """``ops::sliding_window_attention`` / ``context_softmax_attention`` geometry.

    ``query`` [heads*head_dim, Q], ``key``/``value`` [kv_heads*head_dim, K],
    ``mask`` [Q, K] (draft window + lane causality).  BF16 operands, FP32 scores and
    softmax, BF16 weights, FP32 value accumulation, BF16 output.

    GQA: ``kv_heads`` is caller-supplied because the key row count alone cannot
    separate ``heads`` from ``kv_heads`` (the ``linear_split`` geometry implies it).
    ``heads // kv_heads`` query heads share one KV group, and ``linear_split`` maps
    head → group as ``head = group_index * group + member``
    (``src/ops/linear/linear_split.h:107-109``), so head ``h`` reads group
    ``h // (heads // kv_heads)``.  Each head keeps its own scores, its own softmax
    over the visible keys, and its own value rows.

    Rounding points follow the CUDA ``context_softmax`` kernel P0b certified
    bit-exact against the reference: the probabilities are stored as BF16
    (``src/ops/context/cuda/context_softmax_attention.cu:330-337``), every
    ``weight * value`` product rounds to BF16, and the reduction accumulates in FP32
    (``.cu:345-360``).  The ``context_flash_attention_template`` variant the engine
    picks for the short draft window is dtype-parameterised (``.cu:201-207``): BF16
    operands into FP32 compute, FP32 softmax, BF16-packed output — the same points.
    """

    queries = np.ascontiguousarray(f32(query), dtype=np.float32)
    key_matrix = np.ascontiguousarray(f32(key), dtype=np.float32)
    value_matrix = np.ascontiguousarray(f32(value), dtype=np.float32)
    for name, rows in (("query", queries.shape[0]), ("key", key_matrix.shape[0]),
                        ("value", value_matrix.shape[0])):
        if rows % head_dim:
            raise ValueError(f"{name} rows {rows} is not a multiple of head_dim {head_dim}")
    heads = queries.shape[0] // head_dim
    kv_heads = heads if kv_heads is None else int(kv_heads)
    if kv_heads * head_dim != key_matrix.shape[0] or key_matrix.shape != value_matrix.shape:
        raise ValueError(
            f"attention geometry is inconsistent: {heads} query heads, {kv_heads} key "
            f"heads at head_dim {head_dim}"
        )
    if heads % kv_heads:
        raise ValueError(f"heads {heads} is not a multiple of kv heads {kv_heads}")
    extent, keys = queries.shape[1], key_matrix.shape[1]
    visible = np.asarray(mask, dtype=bool)
    if visible.shape != (extent, keys):
        raise ValueError(f"mask {visible.shape} is not [queries, keys] {(extent, keys)}")
    group = heads // kv_heads
    key_heads = key_matrix.reshape(kv_heads, head_dim, keys).repeat(group, axis=0)
    value_heads = value_matrix.reshape(kv_heads, head_dim, keys).repeat(group, axis=0)
    out = np.zeros((heads * head_dim, extent), dtype=np.float32)
    for head in range(heads):
        key_head = np.ascontiguousarray(key_heads[head], dtype=np.float32)  # [hd, K]
        query_head = np.ascontiguousarray(
            queries[head * head_dim : (head + 1) * head_dim], dtype=np.float32
        )                                                                    # [hd, Q]
        scores = (
            np.ascontiguousarray(np.moveaxis(key_head, 0, 1), dtype=np.float32)
            @ query_head
        ).astype(np.float32) * np.float32(scale)                              # [K, Q]
        scores = np.where(visible.T, scores, np.float32(-np.inf))
        # A key column that no query in this block sees (a padded context row, a
        # block position outside the draft window) has max = -inf; subtracting that
        # from itself is NaN and would poison every output of the head.  Clamping the
        # shift to 0 for such columns keeps their weight exactly exp(-inf) = 0.
        shift = np.max(scores, axis=0, keepdims=True)
        shift = np.where(np.isfinite(shift), shift, np.float32(0.0))
        exponent = np.exp(scores - shift)
        total = exponent.sum(axis=0, keepdims=True, dtype=np.float64)
        if (total <= 0).any():
            raise ValueError(
                "a draft query sees no key at all (the block mask must keep at least "
                "the query's own slot visible)"
            )
        probabilities = (exponent / total).astype(np.float32)
        weights = f32(num.bf16_view(probabilities))                           # [K, Q]
        value_head = np.ascontiguousarray(value_heads[head], dtype=np.float32)  # [hd, K]
        for column in range(extent):
            # One bf16-rounded product per (head_dim, key) pair, summed in fp32.
            products = f32(
                num.bf16_view(
                    value_head
                    * np.ascontiguousarray(weights[:, column], dtype=np.float32)[None, :]
                )
            )                                                                 # [hd, K]
            out[head * head_dim : (head + 1) * head_dim, column] = products.sum(
                axis=-1, dtype=np.float32
            )
    return num.bf16_view(out)


def topk_ids(logits_bits: np.ndarray, k: int) -> np.ndarray:
    """Top-k ids per column, ascending by id, ties -> the lower id.

    ``dspine_mlx/dspine_mlx/propose.py:31-36`` (``_topk_ids_full``) — the full stable
    argsort form that ``topk_ids`` itself falls back to when the vocab is narrow
    (``nch < k``), which is the case for the P0b-standard vocab of 1024.
    """

    values = np.ascontiguousarray(f32(logits_bits), dtype=np.float64)
    rows, columns = values.shape
    out = np.zeros((k, columns), dtype=np.int32)
    for column in range(columns):
        order = np.lexsort((np.arange(rows), -values[:, column]))[:k]
        out[:, column] = np.sort(order).astype(np.int32)
    return out


# ------------------------------------------------------------------ bridge steps
# Transcribed from the validated P0b reference (tests/dspine_parity/reference.py),
# which is the module the GPU parity run measured BIT-EXACT against the compiled ops.
# Shapes here are the engine's [rows, width, batch] / column-major layout.


def bridge_prepare_read(residual_bits: np.ndarray, hidden: int, eps: float):
    """``ops::dspine_prepare_read``  (reference.prepare_read; bridge.py:115-121)."""

    flat = np.asarray(residual_bits, dtype=np.uint16).reshape(hidden, -1)
    values = f32(flat).astype(np.float32)
    mean = np.ascontiguousarray(values * values, dtype=np.float32).sum(
        axis=0, dtype=np.float32
    ) / np.float32(hidden)
    mean = np.maximum(mean, F32_TINY)
    rms = np.sqrt(mean).astype(np.float32)
    return num.bf16_view(values / (rms[None, :] + np.float32(eps))), rms


def bridge_normalize_codes(projected_bits: np.ndarray, anchors: np.ndarray,
                           codebook_rows: np.ndarray, code_size: int, width: int,
                           eps: float) -> np.ndarray:
    """``ops::dspine_normalize_codes``  (reference.normalize_codes; bridge.py:123-136).

    ``codebook_rows[a]`` is the dequantized codebook row of token ``a`` (fp32): BF16
    direct, or NVFP4 ``decode(q)*decode(s)/d_w``.
    """

    projected = f32(projected_bits).astype(np.float32)
    sqrt_c = np.float32(math.sqrt(code_size))
    norm = np.sqrt(
        (projected * projected).sum(axis=0, dtype=np.float32)
    ).astype(np.float32)
    codes = projected * (sqrt_c / (norm + np.float32(eps)))[None, :]
    columns = codes.shape[1]
    for lane in range(columns // width):
        row = np.asarray(codebook_rows[int(anchors[lane])], dtype=np.float32)
        codes[:, lane * width] = num.bf16_round(sqrt_c * row).astype(np.float32)
    return num.bf16_view(codes)


def bridge_message(packed_bits: np.ndarray, gate_bias: np.ndarray, rank: int,
                   width: int, batch: int) -> np.ndarray:
    """``ops::dspine_message``  (reference.message; bridge.py:140-148)."""

    packed = f32(np.asarray(packed_bits, dtype=np.uint16).reshape(3 * rank, width,
                                                                 batch)).astype(np.float32)
    bias = np.asarray(gate_bias, dtype=np.float32)
    q, s, v = packed[:rank], packed[rank:2 * rank], packed[2 * rank:]
    out = np.zeros((rank, width, batch), dtype=np.float32)
    for slot in range(1, width):
        out[:, slot, :] = Sigmoid(q[:, slot, :] + s[:, slot - 1, :] + bias[:, None]) * v[
            :, slot - 1, :
        ]
    return num.bf16_view(out)


def bridge_finish(residual_bits: np.ndarray, rms: np.ndarray, delta_bits: np.ndarray,
                  next_norm: np.ndarray, hidden: int, width: int, batch: int,
                  eps: float):
    """``ops::dspine_finish``  (reference.finish; bridge.py:158-172).

    Returns ``(residual_out, normed, raw_stash)`` — ``raw_stash`` is the op's
    ``stash_raw`` (the pre-refinement stream the serial tail consumes).
    """

    raw = f32(np.asarray(residual_bits, dtype=np.uint16).reshape(hidden, width,
                                                                batch)).astype(np.float32)
    delta = f32(np.asarray(delta_bits, dtype=np.uint16).reshape(hidden, width,
                                                               batch)).astype(np.float32)
    gain = np.asarray(next_norm, dtype=np.float32)
    rms_w = np.ascontiguousarray(np.asarray(rms, dtype=np.float32).reshape(width,
                                                                          batch),
                                dtype=np.float32)
    # Slot 0 (the anchor) is the accepted prompt token: it carries no code delta, so
    # its refinement is gated off.  Both factors are [width, batch] and broadcast
    # against [hidden, width, batch] with a leading 1 (this environment's NumPy pads
    # to the *shorter* rank, so a bare [width, batch] would not align).
    anchor_mask = np.zeros((width, batch), dtype=np.float32)
    anchor_mask[1:, :] = 1.0
    out = num.bf16_round(
        raw
        + delta
        * np.ascontiguousarray(rms_w * anchor_mask, dtype=np.float32)[None, :, :]
    )
    values = out.astype(np.float32)
    variance = (values * values).sum(axis=0, dtype=np.float32) / np.float32(hidden)
    inv = (1.0 / np.sqrt(variance + np.float32(eps))).astype(np.float32)
    normed = num.bf16_view(
        num.bf16_round(values * inv[None, :, :]).astype(np.float32)
        * gain[:, None, None]
    )
    return num.bf16_view(out), normed, num.bf16_view(raw)


# --------------------------------------------------------------------- the forward


class DspineForward:
    """The draft forward over one loaded draft (``tools.dspine.artifact.DspineLoad``)."""

    def __init__(self, load, head: dict, *, taps: tuple[int, ...] | None = None):
        self.load = load
        self.head = head                     # {"dequant": [V,H] f32, "embed": [V,H] f32}
        # ``self.taps`` is the **slot order** of the tap axis, not the target layer
        # ids: the artifact's ``feature_projection`` has ``len(taps) * hidden`` input
        # rows and slot ``i`` holds the hidden state of target layer
        # ``load.taps[i]`` (``dspine_recipe.py`` writes the slots in
        # ``target_layer_ids`` order).  The default is the artifact's order; the
        # override exists so a caller can prove the forward is sensitive to which
        # captured layer a slot carries (the teeth check in ``run_forward.py``).
        self.taps = tuple(range(len(load.taps)) if taps is None else taps)
        if sorted(self.taps) != list(range(len(load.taps))):
            raise ValueError(
                f"taps {self.taps} is not a permutation of the {len(load.taps)} "
                f"slots the artifact's feature_projection expects"
            )
        self.rope_theta = float(load.config["rope_parameters"]["rope_theta"])
        self.log: list[dict] = []

    # -- the head ---------------------------------------------------------------

    def head_logits(self, hidden_bits: np.ndarray) -> np.ndarray:
        """BF16 logits from the A8-geometry head (``ops::linear`` route)."""

        weights = np.ascontiguousarray(self.head["codes"], dtype=np.float32) * np.repeat(
            self.head["scales"], self.head["group_size"], axis=1
        )
        return num.bf16_view(weights @ np.ascontiguousarray(f32(hidden_bits),
                                                             dtype=np.float32))

    # -- the block --------------------------------------------------------------

    def make_block(self, anchors: np.ndarray, positions: np.ndarray,
                   mask_token_id: int) -> tuple[np.ndarray, np.ndarray]:
        """The DFlash block: slot 0 = the anchor id, slots 1.. = the mask token.

        Mirrors ``ops::prepare_masked_block`` (``execution/draft.cpp``), which the
        engine runs before the proposal forward for both DFlash2 and DSpine.
        """

        anchors = np.asarray(anchors, dtype=np.int32)
        width = int(positions.shape[0])
        ids = np.full((width, anchors.size), int(mask_token_id), dtype=np.int32)
        ids[0] = anchors
        return ids, np.asarray(positions, dtype=np.int32)

    # -- the whole forward ------------------------------------------------------

    def forward(self, anchors: np.ndarray, block_ids: np.ndarray,
                tap_hidden: np.ndarray, positions: np.ndarray,
                key_positions: np.ndarray, value_rows: np.ndarray,
                key_mask: np.ndarray) -> Proposal:
        """Run one draft block per batch lane.

        ``anchors``     [B] I32, ``block_ids`` [width, B] I32 (slot 0 = the anchor),
        ``tap_hidden``  [B, context + width, len(taps), H] fp32 — the target's hidden
        taps: each lane's ``context`` request-history rows, then its block rows
        (the MLX twin's ``context_len`` + ``block_size`` split, ``model.py:302-306``);
        ``positions``   [width, B] I32 absolute positions of the block rows,
        ``key_positions`` [T, B] I32 absolute positions of the context keys,
        ``value_rows``  [T, B] I32 context row ids (negative = padding),
        ``key_mask``    [width, T, B] bool — draft window + lane causality.
        """

        load = self.load
        hidden, rank, code_size = load.hidden, load.rank, load.code_size
        anchors = np.asarray(anchors, dtype=np.int32)
        block_ids = np.asarray(block_ids, dtype=np.int32)
        width, batch = int(block_ids.shape[0]), int(block_ids.shape[1])
        drafts = width - 1
        columns = width * batch
        key_positions = np.asarray(key_positions, dtype=np.int32)
        value_rows = np.asarray(value_rows, dtype=np.int32)
        context_len = int(value_rows.shape[0])

        # 1) The tap stack is ``[batch, context_len + width, taps, hidden]``: each
        #    lane's request-history rows, then its block rows — the MLX twin's
        #    ``context_len`` + ``block_size`` split (``model.py:302-306`` splits
        #    ``hidden_sh`` = history from ``hidden_blk`` = block).  ``sh_proj`` +
        #    ``sh_norm`` run over the *concatenated* stack (``model.py:255``), and the
        #    result is then split.  Column order is lane-major in both halves
        #    (``lane * rows + row``).
        stacked = np.ascontiguousarray(np.asarray(tap_hidden, dtype=np.float32))
        if stacked.shape[1] != context_len + width:
            raise ValueError(
                f"the tap stack carries {stacked.shape[1]} rows per lane for a context "
                f"of {context_len} plus a block of {width}"
            )
        features = np.concatenate(
            [
                stacked[:, :, slot, :].reshape(batch * (context_len + width), hidden)
                for slot in self.taps
            ],
            axis=-1,
        ).T
        projected = rmsnorm(
            f32(linear(num.bf16_view(features),
                      self.load.tensor("dspine/feature_projection"))),
            self.load.tensor("dspine/context_norm"),
            load.rms_eps,
        )
        context = np.zeros((hidden, context_len * batch), dtype=np.float32)
        block_taps = np.zeros((hidden, columns), dtype=np.float32)
        for lane in range(batch):
            history = lane * (context_len + width)
            context[:, lane * context_len : (lane + 1) * context_len] = (
                projected[:, history : history + context_len]
            )
            block_taps[:, lane * width : (lane + 1) * width] = (
                projected[:, history + context_len : history + context_len + width]
            )

        # 2) the residual stream starts at the token embedding of the block ids.
        residual = num.bf16_view(
            np.ascontiguousarray(self.head["embed"], dtype=np.float32)[
                block_ids.T.reshape(-1)
            ].astype(np.float32).T
        ).reshape(hidden, width, batch)

        stash: dict[str, np.ndarray] = {}
        for layer in range(load.layers):
            residual, stash = self._layer(
                layer, residual, context, block_ids, positions, key_positions,
                value_rows, key_mask, width, batch, block_taps,
            )

        # 3) proposal rows: the last write fused dspine/final_norm into the stream
        #    (model.py:223-232; sglang_dspine/models/dspine.py:527-531), so slots
        #    1..width-1 enter the head as they are.
        proposal_bits = np.asarray(stash["normed"], dtype=np.uint16)[:, 1:, :]
        proposal_bits = proposal_bits.reshape(hidden, drafts * batch)
        logits = self.head_logits(proposal_bits).reshape(
            self.head["dequant"].shape[0], drafts, batch
        )
        candidates = topk_ids(
            logits.reshape(self.head["dequant"].shape[0], drafts * batch),
            load.head_topk,
        ).reshape(load.head_topk, drafts, batch)
        vocab = self.head["dequant"].shape[0]
        return Proposal(
            logits=logits,
            candidates=candidates,
            anchors=anchors,
            hidden=proposal_bits.reshape(hidden, drafts, batch),
            raw=np.asarray(stash["raw"], dtype=np.uint16),
            rms=np.asarray(stash["rms"], dtype=np.float32).reshape(width, batch),
            packed_q=np.asarray(stash["packed_q"], dtype=np.uint16),
            shape_note=(
                f"logits [{vocab}, {drafts}, {batch}] bf16, "
                f"cand [{load.head_topk}, {drafts}, {batch}] i32, "
                f"final_hidden [{hidden}, {drafts}, {batch}] bf16, "
                f"stash raw [{hidden}, {width}, {batch}] rms [{width}, {batch}] "
                f"packed_q [{rank}, {width}, {batch}]"
            ),
        )

    # -- one draft layer --------------------------------------------------------

    def _layer(self, layer: int, residual: np.ndarray, context: np.ndarray,
               block_ids: np.ndarray, positions: np.ndarray, key_positions: np.ndarray,
               value_rows: np.ndarray, key_mask: np.ndarray, width: int, batch: int,
               block_taps: np.ndarray | None = None
               ) -> tuple[np.ndarray, dict]:
        load = self.load
        hidden, rank = load.hidden, load.rank
        prefix = f"dspine/layers/{layer}/"
        columns = width * batch
        flat = residual.reshape(hidden, columns)

        normed = rmsnorm(f32(flat), self.load.tensor(prefix + "input_norm"),
                         load.rms_eps)
        # The block's query/key/value rows come from the block's own projected taps;
        # the request history (``context``) contributes keys/values only.
        block_source = flat if block_taps is None else block_taps

        qkv = np.ascontiguousarray(self.load.tensor(prefix + "attention/query_key_value"),
                                   dtype=np.float32)
        head_dim = load.head_dim
        # ``linear_split`` stores QKV **head-interleaved**: the packed object is
        # ``heads`` blocks of ``stride`` rows, each block (query ``head_dim`` |
        # key ``head_dim // group`` | value ``head_dim // group``) for one query head
        # (``linear_split.h:100-112``; the artifact's 6144 = 32 × (128 + 32 + 32)
        # confirms it).  The GQA heads of one group therefore carry *distinct*
        # ``head_dim // group`` key/value slices, and concatenating a group's slices
        # rebuilds the ``head_dim``-wide band that head addresses
        # (``Q::linear_split_project`` reads head ``h``'s key at
        # ``offsets.key + head * head_dim``).  Head-major concatenation puts them in
        # the ``[heads * head_dim, tokens]`` order the attention kernel expects.
        group = load.heads // load.kv_heads
        bands = load.qkv_bands()
        stride = sum(bands[0])
        query_w = np.concatenate(
            [qkv[head * stride : head * stride + head_dim]
             for head in range(load.heads)], axis=0
        )
        key_projection = np.concatenate(
            [qkv[head * stride + head_dim
                 : head * stride + head_dim + bands[head][1]]
             for head in range(load.heads)], axis=0
        )
        value_projection = np.concatenate(
            [qkv[head * stride + head_dim + bands[head][1]
                 : head * stride + head_dim + bands[head][1] + bands[head][2]]
             for head in range(load.heads)], axis=0
        )
        # ``[kv_heads * head_dim, tokens]``: the GQA geometry the packed object
        # stores, and what ``softmax_attention`` maps onto the ``heads`` queries with
        # the ``linear_split`` head-to-group rule.
        for name, projection, rows in (
            ("query", query_w, load.query_width()),
            ("key", key_projection, load.key_width()),
            ("value", value_projection, load.key_width()),
        ):
            if projection.shape[0] != rows:
                raise ValueError(
                    f"packed qkv {qkv.shape} split yields {projection.shape[0]} {name} "
                    f"rows, expected {rows}"
                )

        def head_norm(bits: np.ndarray, weight: np.ndarray) -> np.ndarray:
            """Per-head ``rms_norm(head_dim)`` over a head's head_dim **rows**.

            ``linear_split`` writes head ``h`` into the row band
            ``h * head_dim … + head_dim`` (``linear_split.h:107-109``), and the engine
            normalises each head's head_dim values per token
            (``src/models/qwen3_5/execution/draft.cpp`` q_norm/k_norm with
            ``head_dim`` as the reduce extent).  The ``[head_dim, tokens]`` block fed
            here is exactly that axis-0 extent, so the norm is over axis 0.
            """

            heads = int(bits.shape[0]) // head_dim
            columns = int(bits.shape[1])
            values = f32(bits)
            out = np.zeros((heads * head_dim, columns), dtype=np.float32)
            for head in range(heads):
                band = slice(head * head_dim, (head + 1) * head_dim)
                block = np.ascontiguousarray(values[band], dtype=np.float32)
                # rmsnorm over axis 0 (the head_dim rows of this head), one scale per
                # column — the shipped op's axis convention (ops/rmsnorm/rmsnorm.h).
                variance = np.ascontiguousarray(
                    block * block, dtype=np.float32
                ).sum(axis=0, dtype=np.float32) / np.float32(head_dim)
                inv = (1.0 / np.sqrt(variance + np.float32(load.rms_eps))).astype(
                    np.float32
                )
                out[band] = num.bf16_bits_to_f32(
                    np.asarray(
                        num.bf16_round(
                            block
                            * np.ascontiguousarray(inv[None, :], dtype=np.float32)
                            * np.ascontiguousarray(
                                weight.astype(np.float32)[:, None], dtype=np.float32
                            )
                        ),
                        dtype=np.uint16,
                    )
                ).astype(np.float32)
            return num.bf16_view(out)

        query = rope(
            head_norm(
                linear(block_source, query_w),
                self.load.tensor(prefix + "attention/query_norm"),
            ),
            positions.T.reshape(-1), head_dim, self.rope_theta,
        )
        # ``[heads * head_dim, tokens]``: every head's key rows are head_dim wide, so
        # the per-head norms and the RoPE bands are uniform across heads.
        block_keys = rope(
            head_norm(linear(block_source, key_projection),
                      self.load.tensor(prefix + "attention/key_norm")),
            positions.T.reshape(-1), head_dim, self.rope_theta,
        )
        block_values = linear(block_source, value_projection)
        # The context rows share the block's k-norm + RoPE path
        # (model.py DraftAttention._context_kv); the engine materialises them once per
        # append (draft.cpp append_context_impl) — same math, per token.
        # ``context`` is the request-history half of the projected tap stack: it
        # contributes keys and values, never queries
        # (``model.py:311`` ``_context_kv(context_sh, context_pos)``; the engine
        # materialises the same rows once per append in ``draft.cpp``
        # ``append_context_impl``).  Rows flagged padding in ``value_rows`` are
        # dropped by the mask below.
        key_positions = np.asarray(key_positions, dtype=np.int32)
        value_rows = np.asarray(value_rows, dtype=np.int32)
        if context.shape[1] != key_positions.size:
            raise ValueError(
                f"context stream has {context.shape[1]} columns for "
                f"{key_positions.shape} context keys"
            )
        context_len = int(value_rows.shape[0])
        if context.shape[1] != context_len * batch:
            raise ValueError(
                f"context stream has {context.shape[1]} columns for "
                f"{context.shape[1] // max(batch, 1)}x{batch} context keys"
            )
        context_keys = rope(
            head_norm(linear(context, key_projection),
                      self.load.tensor(prefix + "attention/key_norm")),
            key_positions.T.reshape(-1), head_dim, self.rope_theta,
        )
        context_values = linear(context, value_projection)
        # ``value_rows`` is [context_len, batch] with **negative = padding**: the rows
        # the draft-cache pages do not hold for this lane.  The rows enter the kernel
        # as lane-major context columns (``lane * context_len + row``), so the
        # visibility vector is that slice transposed and flattened.
        context_valid = value_rows >= 0
        causal = np.array([[other <= query_slot for other in range(width)]
                           for query_slot in range(width)])
        attention = np.zeros((load.query_width(), columns), dtype=np.float32)
        for lane in range(batch):
            # A block query of lane `lane` sees its own context rows (the engine's
            # per-lane paged/cyclic KV rows — a lane never reads another lane's
            # history) plus its own block, causally.
            visible = np.concatenate([
                np.asarray(context_valid[:, lane], dtype=bool),
                np.ones(width, dtype=bool),
            ])
            mask = (
                np.concatenate(
                    [np.asarray(key_mask, dtype=bool)[:, :, lane], causal], axis=1
                )
                & visible[None, :]
            )
            # Lane `lane` attends its own context rows plus its own block; the other
            # lanes' columns are masked out of that lane's softmax entirely.
            lane_keys = np.concatenate(
                [context_keys[:, lane * context_len : (lane + 1) * context_len],
                 block_keys[:, lane * width : (lane + 1) * width]],
                axis=1,
            )
            lane_values = np.concatenate(
                [context_values[:, lane * context_len : (lane + 1) * context_len],
                 block_values[:, lane * width : (lane + 1) * width]],
                axis=1,
            )
            attention[:, lane * width : (lane + 1) * width] = softmax_attention(
                query[:, lane * width : (lane + 1) * width],
                lane_keys,
                lane_values,
                mask,
                head_dim,
                1.0 / math.sqrt(head_dim),
                kv_heads=load.kv_heads,
            )
        attention_bits = num.bf16_view(attention)

        out = linear(attention_bits, self.load.tensor(prefix + "attention/output"))
        hidden_stream = num.bf16_view(f32(out) + f32(flat))

        post = rmsnorm(f32(hidden_stream),
                       self.load.tensor(prefix + "post_attention_norm"), load.rms_eps)
        intermediate = swiglu(post, self.load.tensor(prefix + "mlp/gate_up"),
                             load.intermediate)
        down = linear(intermediate, self.load.tensor(prefix + "mlp/down"))
        mlp_out = num.bf16_view(f32(down) + f32(hidden_stream))

        return self._write(layer, mlp_out.reshape(hidden, width, batch), block_ids,
                           width, batch)

    # -- one code-bridge write --------------------------------------------------

    def _write(self, layer: int, mlp_out: np.ndarray, block_ids: np.ndarray,
               width: int, batch: int) -> tuple[np.ndarray, dict]:
        load = self.load
        hidden, rank, code_size = load.hidden, load.rank, load.code_size
        columns = width * batch
        last = layer == load.layers - 1

        normalized, rms = bridge_prepare_read(
            mlp_out.reshape(hidden, columns), hidden, load.eps
        )
        read_weight = (
            self.load.tensor("dspine/bridge/read_weight")
            if last
            else self.load.tensor("dspine/bridge/early_read_weights")[layer]
        )
        projected = linear(normalized, read_weight)
        anchors = np.asarray(block_ids[0], dtype=np.int32)
        codes = bridge_normalize_codes(
            projected, anchors, self.load.tensor("dspine/bridge/codebook"),
            code_size, width, load.eps,
        )
        packed_weight = np.concatenate(
            [
                np.asarray(self.load.tensor(f"dspine/bridge/{name}"), dtype=np.float32)
                for name in ("gate", "predecessor_gate", "prev")
            ],
            axis=0,
        )
        packed = linear(codes, packed_weight)
        gate_bias = self.load.tensor("dspine/bridge/gate_bias")
        message = bridge_message(packed, gate_bias, rank, width, batch)
        delta = linear(
            message.reshape(rank, columns),
            self.load.tensor("dspine/bridge/feature_write_weights")[layer],
        )
        next_norm = (
            self.load.tensor(f"dspine/layers/{layer + 1}/input_norm")
            if not last
            else self.load.tensor("dspine/final_norm")
        )
        out, normed, raw = bridge_finish(
            mlp_out, rms, delta.reshape(hidden, width, batch), next_norm, hidden, width,
            batch, load.rms_eps,
        )
        stash = {
            "raw": raw,
            "rms": rms,
            "packed_q": np.asarray(packed, dtype=np.uint16).reshape(
                3 * rank, width, batch
            )[:rank],
            "normed": normed,
        }
        return out.reshape(hidden, width, batch), stash
