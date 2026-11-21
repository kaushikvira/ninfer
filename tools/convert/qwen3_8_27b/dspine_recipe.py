"""Source contract and recipes for the Qwen3.8-27B DSpine suffix.

DSpine (Tencent, arXiv 2609.36173) shares the DFlash 5-layer sliding-attention
backbone but replaces the DFlash2 conv/selector machinery with a selected-prefix
code bridge written after every layer and a serial-refinement decode tail.
The artifact suffix keeps the backbone linears in NVFP4 (weight-only, A16Only)
and the bridge plus all norms/codebook in BF16.

This is the DSpine counterpart of ``tools.convert.dflash``.  Two differences are
deliberate:

* ``REVISION`` is the ``DSpineDraftModel`` release (D-13).  The public fork pins
  ``30de2dad``, whose ``config.json`` declares ``DFlashDraftModel``; that
  revision is rejected by :func:`validate_config`.
* object formats are this repository's persistent format names (``nvfp4``,
  ``bf16``), not the fork's ``NVFP4``/``BF16`` aliases.  The fork names layouts
  ``blockscale-k16-m128x4-v1``; this repository persists
  ``block_scale_k16_m128x4_v1``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch

from .recipe import (
    Concat,
    ShardReader,
    SourcePreflight,
    SourceTensor,
    TensorRecipe,
    check_members,
    expression_shape,
    materialize_recipe,
    preflight_source_reader,
    source,
    source_requirements as recipe_source_requirements,
)

REPOSITORY = "tencent/Qwen3.8-27B-DSpine-b16"
REVISION = "260a6c23070ca873ff1a5f33b7119ac3f6a245ee"

DSPINE_LAYERS = tuple(range(5))
PREFIX = "dspine/"
# The released checkpoint accepts mask tokens below the tokenizer's control row.
MASK_TOKEN_LIMIT = 248077
# fc.weight is the concatenation of one hidden-wide slice per tapped target layer.
TARGET_FEATURE_WIDTH = 25600

_ROOT_CONFIG = {
    "architectures": ["DSpineDraftModel"],
    "model_type": "qwen3",
    "dtype": "bfloat16",
    "hidden_act": "silu",
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "num_hidden_layers": 5,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "attention_bias": False,
    "is_causal": True,
    "layer_types": ["sliding_attention"] * 5,
    "use_sliding_window": True,
    "max_window_layers": 5,
    "sliding_window": 8192,
    "vocab_size": 248320,
    "draft_vocab_size": 248320,
    "num_target_layers": 64,
    "max_position_embeddings": 262144,
    "rms_norm_eps": 1e-06,
    "tie_word_embeddings": False,
    "block_size": 16,
}
_ROPE_CONFIG = {
    "rope_theta": 10000000,
    "rope_type": "default",
}
_SECOND_PASS_CONFIG = {
    "first_exact": 4,
    "head_topk": 16,
    "mode": "serial_table",
}
_BRIDGE_CONFIG = {
    "code_size": 1024,
    "eps": 1e-06,
    "kappa": 10.0,
    "rank": 512,
}
_LAST_WRITE_REFINE = {
    "head_topk": 16,
    "kind": "exact_sender",
    "version": 1,
    "weight": 1.0,
}
# Inference-relevant dflash_config members; training-only fields
# (teacher layers/schedule, gt_training, checkpointing) are deliberately dropped.
_DRAFT_CONFIG = {
    "draft_block_causal": True,
    "mask_token_id": 248070,
    "selected_prefix_code_bridge": _BRIDGE_CONFIG,
    "selected_prefix_decode_mode": "serial",
    "selected_prefix_feature_layers": [1, 2, 3, 4, 5],
    "selected_prefix_feature_write_mode": "independent",
    "selected_prefix_feature_writes": True,
    "selected_prefix_last_write_refine": _LAST_WRITE_REFINE,
    "selected_prefix_terminal": False,
    "target_layer_ids": [1, 16, 31, 46, 61],
}

_VERSION_SCALARS = (
    "draft_block_causal_version",
    "last_write_refine_version",
    "selected_prefix_bridge.independent_write_version",
    "selected_prefix_bridge.terminal_disabled_version",
)


def validate_config(config: Mapping[str, object]) -> dict[str, object]:
    """Validate every checkpoint field consumed by the fixed DSpine runtime."""

    check_members("DSpine config", config, _ROOT_CONFIG)
    rope = config.get("rope_parameters")
    draft = config.get("dflash_config")
    second_pass = config.get("code_bridge_second_pass")
    if not isinstance(rope, Mapping) or not isinstance(draft, Mapping):
        raise ValueError(
            "DSpine config.json must contain rope_parameters and dflash_config"
        )
    if not isinstance(second_pass, Mapping):
        raise ValueError("DSpine config.json must contain code_bridge_second_pass")
    check_members("DSpine config.rope_parameters", rope, _ROPE_CONFIG)
    check_members("DSpine config.code_bridge_second_pass", second_pass, _SECOND_PASS_CONFIG)
    check_members("DSpine config.dflash_config", draft, _DRAFT_CONFIG)
    return {
        "architecture": config["architectures"][0],
        "model_type": config["model_type"],
        "dtype": config["dtype"],
        "hidden_size": config["hidden_size"],
        "intermediate_size": config["intermediate_size"],
        "num_hidden_layers": config["num_hidden_layers"],
        "num_attention_heads": config["num_attention_heads"],
        "num_key_value_heads": config["num_key_value_heads"],
        "head_dim": config["head_dim"],
        "layer_types": list(config["layer_types"]),
        "is_causal": config["is_causal"],
        "sliding_window": config["sliding_window"],
        "vocab_size": config["vocab_size"],
        "num_target_layers": config["num_target_layers"],
        "max_position_embeddings": config["max_position_embeddings"],
        "rms_norm_eps": config["rms_norm_eps"],
        "block_size": config["block_size"],
        "rope_parameters": {name: rope[name] for name in _ROPE_CONFIG},
        "code_bridge_second_pass": dict(second_pass),
        "dflash_config": {name: draft[name] for name in _DRAFT_CONFIG},
    }


def validate_base_compatibility(
    base_summary: Mapping[str, object],
    dspine_summary: Mapping[str, object],
) -> None:
    """Validate the shared geometry between target and DSpine checkpoints."""

    text = base_summary.get("text")
    rope = base_summary.get("rope")
    draft_rope = dspine_summary.get("rope_parameters")
    draft_config = dspine_summary.get("dflash_config")
    if not isinstance(text, Mapping) or not isinstance(rope, Mapping):
        raise ValueError("base config summary is missing text or rope facts")
    if not isinstance(draft_rope, Mapping) or not isinstance(draft_config, Mapping):
        raise ValueError("DSpine config summary is incomplete")

    pairs = (
        ("hidden_size", text.get("hidden_size"), dspine_summary.get("hidden_size")),
        ("vocab_size", text.get("vocab_size"), dspine_summary.get("vocab_size")),
        (
            "num_target_layers",
            text.get("num_hidden_layers"),
            dspine_summary.get("num_target_layers"),
        ),
        (
            "max_position_embeddings",
            text.get("max_position_embeddings"),
            dspine_summary.get("max_position_embeddings"),
        ),
        ("rope_theta", rope.get("rope_theta"), draft_rope.get("rope_theta")),
    )
    for name, base_value, draft_value in pairs:
        if base_value != draft_value:
            raise ValueError(
                f"base/DSpine {name} mismatch: {base_value!r} != {draft_value!r}"
            )

    target_layer_ids = draft_config.get("target_layer_ids")
    if (
        not isinstance(target_layer_ids, list)
        or target_layer_ids != sorted(set(target_layer_ids))
    ):
        raise ValueError("DSpine target_layer_ids must be unique and strictly increasing")
    target_layers = int(dspine_summary["num_target_layers"])
    if not target_layer_ids or not all(
        isinstance(layer, int) and 0 <= layer < target_layers
        for layer in target_layer_ids
    ):
        raise ValueError("DSpine target_layer_ids are outside the target")
    if not 0 <= int(draft_config["mask_token_id"]) < MASK_TOKEN_LIMIT:
        raise ValueError("DSpine mask token is not tokenizer-addressable")
    if len(target_layer_ids) * int(dspine_summary["hidden_size"]) != TARGET_FEATURE_WIDTH:
        raise ValueError("DSpine target-feature width does not match fc.weight")


# --- object inventory -------------------------------------------------------

NVFP4 = "nvfp4"
BF16 = "bf16"
NVFP4_LAYOUT = "block_scale_k16_m128x4_v1"
CONTIGUOUS_LAYOUT = "contiguous_le_v1"


def _build_specs() -> tuple[TensorRecipe, ...]:
    recipes: list[TensorRecipe] = [
        TensorRecipe(
            "dspine/feature_projection",
            source("fc.weight", (5120, 25600)),
        ),
        TensorRecipe(
            "dspine/context_norm",
            source("hidden_norm.weight", (5120,)),
        ),
    ]
    for layer in DSPINE_LAYERS:
        source_prefix = f"layers.{layer}."
        object_prefix = f"dspine/layers/{layer}/"
        recipes.extend(
            (
                TensorRecipe(
                    object_prefix + "input_norm",
                    source(source_prefix + "input_layernorm.weight", (5120,)),
                ),
                TensorRecipe(
                    object_prefix + "attention/query_key_value",
                    Concat(
                        (
                            source(
                                source_prefix + "self_attn.q_proj.weight",
                                (4096, 5120),
                            ),
                            source(
                                source_prefix + "self_attn.k_proj.weight",
                                (1024, 5120),
                            ),
                            source(
                                source_prefix + "self_attn.v_proj.weight",
                                (1024, 5120),
                            ),
                        ),
                        0,
                    ),
                ),
                TensorRecipe(
                    object_prefix + "attention/query_norm",
                    source(source_prefix + "self_attn.q_norm.weight", (128,)),
                ),
                TensorRecipe(
                    object_prefix + "attention/key_norm",
                    source(source_prefix + "self_attn.k_norm.weight", (128,)),
                ),
                TensorRecipe(
                    object_prefix + "attention/output",
                    source(
                        source_prefix + "self_attn.o_proj.weight", (5120, 4096)
                    ),
                ),
                TensorRecipe(
                    object_prefix + "post_attention_norm",
                    source(
                        source_prefix + "post_attention_layernorm.weight", (5120,)
                    ),
                ),
                TensorRecipe(
                    object_prefix + "mlp/gate_up",
                    Concat(
                        (
                            source(
                                source_prefix + "mlp.gate_proj.weight",
                                (17408, 5120),
                            ),
                            source(
                                source_prefix + "mlp.up_proj.weight",
                                (17408, 5120),
                            ),
                        ),
                        0,
                    ),
                ),
                TensorRecipe(
                    object_prefix + "mlp/down",
                    source(
                        source_prefix + "mlp.down_proj.weight", (5120, 17408)
                    ),
                ),
            )
        )
    recipes.extend(
        (
            TensorRecipe("dspine/final_norm", source("norm.weight", (5120,))),
            TensorRecipe(
                "dspine/bridge/codebook",
                source("selected_prefix_bridge.codebook", (248320, 1024)),
            ),
            TensorRecipe(
                "dspine/bridge/early_read_weights",
                source(
                    "selected_prefix_bridge.early_read_weights", (4, 1024, 5120)
                ),
            ),
            TensorRecipe(
                "dspine/bridge/read_weight",
                source("selected_prefix_bridge.read_weight", (1024, 5120)),
            ),
            TensorRecipe(
                "dspine/bridge/feature_write_weights",
                source(
                    "selected_prefix_bridge.feature_write_weights", (5, 5120, 512)
                ),
            ),
            TensorRecipe(
                "dspine/bridge/gate",
                source("selected_prefix_bridge.gate.weight", (512, 1024)),
            ),
            TensorRecipe(
                "dspine/bridge/gate_bias",
                source("selected_prefix_bridge.gate.bias", (512,)),
            ),
            TensorRecipe(
                "dspine/bridge/predecessor_gate",
                source(
                    "selected_prefix_bridge.predecessor_gate.weight", (512, 1024)
                ),
            ),
            TensorRecipe(
                "dspine/bridge/prev",
                source("selected_prefix_bridge.prev.weight", (512, 1024)),
            ),
        )
    )
    return tuple(recipes)


DSPINE_RECIPE_SPECS = _build_specs()
DSPINE_RECIPES_BY_NAME = {item.object_name: item for item in DSPINE_RECIPE_SPECS}

# Backbone linears carry NVFP4; the bridge, norms and codebook stay BF16.
_NVFP4_OBJECTS = frozenset(
    ["dspine/feature_projection"]
    + [
        f"dspine/layers/{layer}/{suffix}"
        for layer in DSPINE_LAYERS
        for suffix in (
            "attention/query_key_value",
            "attention/output",
            "mlp/gate_up",
            "mlp/down",
        )
    ]
)

# The codebook and the two refinement tables are gather-only: the engine's
# DSpine gather kernels dequantize rows on access, which trades ~870 MiB of
# draft VRAM for quantization error in the serial tail.
NVFP4_GATHER_OBJECTS = frozenset(
    ["dspine/bridge/codebook", "dspine/bridge/tg", "dspine/bridge/tsv"]
)


def object_format(object_name: str, gather_tables_nvfp4: bool = False) -> str:
    if object_name in _NVFP4_OBJECTS:
        return NVFP4
    if gather_tables_nvfp4 and object_name in NVFP4_GATHER_OBJECTS:
        return NVFP4
    return BF16


# Vocabulary-wide greedy refinement tables derived from the target lm_head and
# the bridge at attach time (see scripts/attach_dspine_head.compute_serial_tables).
# TG[v] = bf16(lm_head[v] * final_norm) @ write_w[-1]   [V, R]
# TSV[v] = bf16(sqrt(C) * codebook[v]) @ [predecessor_gate; prev].T   [V, 2R]
# Both are tower-specific: they MUST be recomputed for the target they attach to.
SERIAL_TABLE_OBJECTS = {
    "dspine/bridge/tg": (248320, 512),
    "dspine/bridge/tsv": (248320, 1024),
}


def object_shape(object_name: str) -> tuple[int, ...]:
    if object_name in SERIAL_TABLE_OBJECTS:
        return SERIAL_TABLE_OBJECTS[object_name]
    return expression_shape(DSPINE_RECIPES_BY_NAME[object_name].expression)


def source_requirements() -> dict[str, SourceTensor]:
    return recipe_source_requirements(DSPINE_RECIPE_SPECS)


def required_source_names() -> frozenset[str]:
    return (
        frozenset(recipe_source_requirements(DSPINE_RECIPE_SPECS))
        | frozenset(_VERSION_SCALARS)
        | frozenset({"selected_prefix_bridge.out.weight"})
    )


def preflight_sources(model_dir: str | Path) -> SourcePreflight:
    model = Path(model_dir)
    requirements = recipe_source_requirements(DSPINE_RECIPE_SPECS)
    with ShardReader.from_file(model / "model.safetensors") as reader:
        actual_names = frozenset(reader.names)
        required_names = required_source_names()
        if actual_names != required_names:
            missing = sorted(required_names - actual_names)
            extra = sorted(actual_names - required_names)
            details = []
            if missing:
                details.append(f"missing={missing[:8]!r}")
            if extra:
                details.append(f"extra={extra[:8]!r}")
            raise ValueError(
                "DSpine source inventory differs from its exact tensor contract"
                + (": " + ", ".join(details) if details else "")
            )
        for name in _VERSION_SCALARS:
            value = reader.get(name)
            if value.dtype != torch.int64 or int(value.item()) != 1:
                raise ValueError(f"{name}: expected version scalar 1, got {value!r}")
        out_weight = reader.get("selected_prefix_bridge.out.weight")
        if out_weight.shape != (1024, 512) or bool(out_weight.any()):
            raise ValueError(
                "selected_prefix_bridge.out.weight must be all-zero (terminal disabled)"
            )
        return preflight_source_reader(reader, DSPINE_RECIPE_SPECS)


def materialize_tensor(object_name: str, reader: ShardReader) -> torch.Tensor:
    return materialize_recipe(DSPINE_RECIPES_BY_NAME[object_name], reader)


# --- v3 directory entries ----------------------------------------------------

_QKV_RANGES = {
    "query": (0, 20971520),
    "key": (20971520, 26214400),
    "value": (26214400, 31457280),
    "context_key": (20971520, 26214400),
    "context_value": (26214400, 31457280),
}
_GATE_UP_RANGES = {
    "gate": (0, 89128960),
    "up": (89128960, 178257920),
}


def build_bindings() -> dict[str, object]:
    bindings: dict[str, object] = {
        "dspine/feature_projection": {"object": "dspine/feature_projection"},
        "dspine/context_norm": {"object": "dspine/context_norm"},
        "dspine/final_norm": {"object": "dspine/final_norm"},
    }
    for layer in DSPINE_LAYERS:
        prefix = f"dspine/layers/{layer}/"
        qkv = prefix + "attention/query_key_value"
        gate_up = prefix + "mlp/gate_up"
        for name in ("input_norm", "post_attention_norm"):
            bindings[prefix + name] = {"object": prefix + name}
        for name in ("query_norm", "key_norm", "output"):
            bindings[prefix + "attention/" + name] = {
                "object": prefix + "attention/" + name
            }
        for name, (begin, end) in _QKV_RANGES.items():
            bindings[prefix + "attention/" + name] = {
                "parts": [{"object": qkv, "range": [begin, end]}]
            }
        for name, (begin, end) in _GATE_UP_RANGES.items():
            bindings[prefix + "mlp/" + name] = {
                "parts": [{"object": gate_up, "range": [begin, end]}]
            }
        bindings[prefix + "mlp/down"] = {"object": prefix + "mlp/down"}
    for name in (
        "codebook",
        "early_read_weights",
        "read_weight",
        "feature_write_weights",
        "gate",
        "gate_bias",
        "predecessor_gate",
        "prev",
    ):
        bindings["dspine/bridge/" + name] = {"object": "dspine/bridge/" + name}
    for name in SERIAL_TABLE_OBJECTS:
        bindings[name] = {"object": name}
    return bindings


def build_uses() -> list[dict[str, object]]:
    uses: list[dict[str, object]] = [
        {
            "parameter": "dspine/feature_projection",
            "input": "dspine/target_features",
            "activation_policy": "A16Only",
        }
    ]
    for layer in DSPINE_LAYERS:
        prefix = f"dspine/layers/{layer}/"
        for name in ("query", "key", "value"):
            uses.append(
                {
                    "parameter": prefix + "attention/" + name,
                    "input": prefix + "query_projection_input",
                    "activation_policy": "A16Only",
                }
            )
        for name in ("context_key", "context_value"):
            uses.append(
                {
                    "parameter": prefix + "attention/" + name,
                    "input": "dspine/context_input",
                    "activation_policy": "A16Only",
                }
            )
        uses.append(
            {
                "parameter": prefix + "attention/output",
                "input": prefix + "attention_output",
                "activation_policy": "A16Only",
            }
        )
        for name in ("gate", "up"):
            uses.append(
                {
                    "parameter": prefix + "mlp/" + name,
                    "input": prefix + "mlp_input",
                    "activation_policy": "A16Only",
                }
            )
        uses.append(
            {
                "parameter": prefix + "mlp/down",
                "input": prefix + "mlp_product",
                "activation_policy": "A16Only",
            }
        )
    for name, site in (
        ("read_weight", "dspine/bridge/read_input"),
        ("early_read_weights", "dspine/bridge/early_read_input"),
        ("gate", "dspine/bridge/packed_input"),
        ("predecessor_gate", "dspine/bridge/packed_input"),
        ("prev", "dspine/bridge/packed_input"),
        ("feature_write_weights", "dspine/bridge/message_input"),
    ):
        uses.append(
            {
                "parameter": "dspine/bridge/" + name,
                "input": site,
                "activation_policy": "A16Only",
            }
        )
    uses.append(
        {
            "parameter": "text/output_head",
            "input": "dspine/final_hidden",
            "activation_policy": "AllowA8",
        }
    )
    return uses


def build_component(config_summary: Mapping[str, object]) -> dict[str, object]:
    return {
        "config": {
            "architectures": [config_summary["architecture"]],
            "model_type": config_summary["model_type"],
            "hidden_size": config_summary["hidden_size"],
            "intermediate_size": config_summary["intermediate_size"],
            "num_hidden_layers": config_summary["num_hidden_layers"],
            "num_attention_heads": config_summary["num_attention_heads"],
            "num_key_value_heads": config_summary["num_key_value_heads"],
            "head_dim": config_summary["head_dim"],
            "rms_norm_eps": config_summary["rms_norm_eps"],
            "rope_parameters": dict(config_summary["rope_parameters"]),
            "max_position_embeddings": config_summary["max_position_embeddings"],
            "layer_types": list(config_summary["layer_types"]),
            "sliding_window": config_summary["sliding_window"],
            "is_causal": config_summary["is_causal"],
            "block_size": config_summary["block_size"],
            "vocab_size": config_summary["vocab_size"],
            "num_target_layers": config_summary["num_target_layers"],
            "code_bridge_second_pass": dict(config_summary["code_bridge_second_pass"]),
            "dflash_config": dict(config_summary["dflash_config"]),
        },
        "target": "text",
    }


__all__ = [
    "BF16",
    "CONTIGUOUS_LAYOUT",
    "DSPINE_LAYERS",
    "DSPINE_RECIPES_BY_NAME",
    "DSPINE_RECIPE_SPECS",
    "NVFP4",
    "NVFP4_GATHER_OBJECTS",
    "NVFP4_LAYOUT",
    "PREFIX",
    "REPOSITORY",
    "REVISION",
    "SERIAL_TABLE_OBJECTS",
    "build_bindings",
    "build_component",
    "build_uses",
    "materialize_tensor",
    "object_format",
    "object_shape",
    "preflight_sources",
    "required_source_names",
    "source_requirements",
    "validate_base_compatibility",
    "validate_config",
]
