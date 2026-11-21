from __future__ import annotations

import pytest

from tools.artifact.formats import get_format
from tools.artifact.layouts import get_layout
from tools.artifact.schema import TensorObject, binding_parts
from tools.convert.qwen3_8_27b import dspine_recipe as dsp
from tools.convert.qwen3_8_27b.recipe import expression_shape


def _release_config(**overrides) -> dict:
    """The `DSpine-b16` release contract, written out as the test target."""

    config = {
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
        "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
        "code_bridge_second_pass": {
            "first_exact": 4,
            "head_topk": 16,
            "mode": "serial_table",
        },
        "dflash_config": {
            "draft_block_causal": True,
            "mask_token_id": 248070,
            "selected_prefix_code_bridge": {
                "code_size": 1024,
                "eps": 1e-06,
                "kappa": 10.0,
                "rank": 512,
            },
            "selected_prefix_decode_mode": "serial",
            "selected_prefix_feature_layers": [1, 2, 3, 4, 5],
            "selected_prefix_feature_write_mode": "independent",
            "selected_prefix_feature_writes": True,
            "selected_prefix_last_write_refine": {
                "head_topk": 16,
                "kind": "exact_sender",
                "version": 1,
                "weight": 1.0,
            },
            "selected_prefix_terminal": False,
            "target_layer_ids": [1, 16, 31, 46, 61],
        },
    }
    config.update(overrides)
    return config


def _base_summary() -> dict:
    return {
        "text": {
            "hidden_size": 5120,
            "vocab_size": 248320,
            "num_hidden_layers": 64,
            "max_position_embeddings": 262144,
        },
        "rope": {"rope_theta": 10000000},
    }


def _objects() -> dict[str, TensorObject]:
    packed = [
        f"dspine/layers/{layer}/{suffix}"
        for layer in dsp.DSPINE_LAYERS
        for suffix in ("attention/query_key_value", "mlp/gate_up")
    ]
    return {
        name: TensorObject(
            name,
            dsp.object_shape(name),
            dsp.object_format(name),
            dsp.NVFP4_LAYOUT,
            0,
            8,
        )
        for name in packed
    }


def test_revision_pins_the_dspine_draft_release():
    assert dsp.REPOSITORY == "tencent/Qwen3.8-27B-DSpine-b16"
    assert dsp.REVISION.startswith("260a6c23")


def test_stale_dflash_release_is_rejected():
    config = _release_config(architectures=["DFlashDraftModel"])
    with pytest.raises(ValueError, match="architectures"):
        dsp.validate_config(config)


def test_release_contract_and_target_geometry():
    summary = dsp.validate_config(_release_config())
    dsp.validate_base_compatibility(_base_summary(), summary)
    assert summary["block_size"] == 16
    assert summary["dflash_config"]["target_layer_ids"] == [1, 16, 31, 46, 61]


@pytest.mark.parametrize(
    "key, value",
    [
        ("sliding_window", 2048),
        ("block_size", 8),
        ("num_target_layers", 28),
    ],
)
def test_runtime_geometry_is_not_negotiable(key, value):
    with pytest.raises(ValueError, match="checkpoint config mismatch"):
        dsp.validate_config(_release_config(**{key: value}))


def test_base_must_share_the_tower_geometry():
    summary = dsp.validate_config(_release_config())
    base = _base_summary()
    base["text"]["vocab_size"] = 151936
    with pytest.raises(ValueError, match="vocab_size mismatch"):
        dsp.validate_base_compatibility(base, summary)


def test_inventory_shape_and_uniqueness():
    names = [recipe.object_name for recipe in dsp.DSPINE_RECIPE_SPECS]
    assert len(names) == 51
    assert len(set(names)) == len(names)
    assert all(name.startswith(dsp.PREFIX) for name in names)
    for recipe in dsp.DSPINE_RECIPE_SPECS:
        assert dsp.object_shape(recipe.object_name) == expression_shape(
            recipe.expression
        )


def test_formats_and_layouts_are_registered_persistent_names():
    for recipe in dsp.DSPINE_RECIPE_SPECS:
        name = recipe.object_name
        fmt = dsp.object_format(name)
        get_format(fmt)
        get_layout(
            dsp.NVFP4_LAYOUT
            if fmt == dsp.NVFP4
            else dsp.CONTIGUOUS_LAYOUT
        )
    backbone = {
        f"dspine/layers/{layer}/mlp/gate_up" for layer in dsp.DSPINE_LAYERS
    }
    backbone.add("dspine/feature_projection")
    assert backbone <= dsp._NVFP4_OBJECTS
    assert "dspine/final_norm" not in dsp._NVFP4_OBJECTS
    assert dsp.object_format("dspine/bridge/codebook") == dsp.BF16
    assert (
        dsp.object_format("dspine/bridge/codebook", gather_tables_nvfp4=True)
        == dsp.NVFP4
    )


def test_serial_tables_carry_the_full_vocabulary():
    assert dsp.SERIAL_TABLE_OBJECTS == {
        "dspine/bridge/tg": (248320, 512),
        "dspine/bridge/tsv": (248320, 1024),
    }
    for name in dsp.SERIAL_TABLE_OBJECTS:
        assert dsp.object_shape(name) == dsp.SERIAL_TABLE_OBJECTS[name]


def test_packed_qkv_and_gate_up_ranges_cover_their_parents():
    objects = _objects()
    for layer in dsp.DSPINE_LAYERS:
        prefix = f"dspine/layers/{layer}/"
        qkv = prefix + "attention/query_key_value"
        gate_up = prefix + "mlp/gate_up"
        bindings = dsp.build_bindings()
        for name, parent in (
            ("attention/query", qkv),
            ("attention/key", qkv),
            ("attention/value", qkv),
            ("attention/context_key", qkv),
            ("attention/context_value", qkv),
            ("mlp/gate", gate_up),
            ("mlp/up", gate_up),
        ):
            parts = binding_parts(bindings[prefix + name], objects, prefix + name)
            assert [part[0] for part in parts] == [parent]
        qkv_rows = sum(
            parts[0][2] - parts[0][1]
            for parts in (
                binding_parts(bindings[prefix + f"attention/{role}"], objects)
                for role in ("query", "key", "value")
            )
        )
        assert qkv_rows == 6144 * 5120


def test_every_use_names_a_declared_parameter():
    bindings = dsp.build_bindings()
    bindings["text/output_head"] = {"object": "weight/000000"}
    seen = set()
    for use in dsp.build_uses():
        assert use["parameter"] in bindings, use
        assert use["activation_policy"] == "A16Only" or use[
            "parameter"
        ] == "text/output_head"
        key = (use["parameter"], use["input"])
        assert key not in seen
        seen.add(key)
    assert ("text/output_head", "dspine/final_hidden") in seen


def test_component_targets_the_text_tower():
    summary = dsp.validate_config(_release_config())
    component = dsp.build_component(summary)
    assert component["target"] == "text"
    assert component["config"]["architectures"] == ["DSpineDraftModel"]
    assert component["config"]["dflash_config"]["mask_token_id"] == 248070
