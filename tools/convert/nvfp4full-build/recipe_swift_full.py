"""Full nvfp4full-style recipe for the abliterated Swift build (mirrors prod profile).

- text: every projection NVFP4 W4A4 (fused parents via normalized checkpoint),
  gdn a/b projections BF16, token_embedding + output_head Q8 (W8G32) like prod.
- vision: official _optional allocation (q6/q8) like prod.
- mtp: BF16 (upstream route supports it).
"""
from __future__ import annotations

from tools.convert.methods import import_encoded
from tools.convert.official_recipes import Q8, _assign, _optional

FP8 = "fp8_e4m3fn_row_bf16"
NVFP4 = "nvfp4"


def configure(model, recipe, sources):
    quantized = sources["quantized"]

    # vocabulary roles: Q8 groupwise like prod (cometkim nvfp4full)
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q8)

    for name, parameter in model.parameters.items():
        if not name.startswith("text/") or name in (
            "text/token_embedding",
            "text/output_head",
        ):
            continue
        if not parameter.projection or name.endswith(
            ("/gdn/a_projection", "/gdn/b_projection")
        ):
            recipe.assign(name)
            continue
        recipe.assign(
            name,
            format=NVFP4,
            method=import_encoded,
            source=model.source(name, quantized, NVFP4),
            activation_policy="AllowA4",
        )

    # vision tower: official q6/q8 allocation (same as prod)
    _optional(model, recipe)
