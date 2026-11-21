#include "models/common/dspine/config.h"
#include "models/common/dspine/dspine_serial_tail.h"
#include "artifact/schema.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

namespace ninfer::models::dspine {
namespace {
using artifact::ArtifactError;
using artifact::Json;
using artifact::require_members;

std::uint32_t integer(const Json& value, std::string_view label, bool positive = true) {
    const auto n = artifact::require_u64(value, label, positive);
    if (n > std::numeric_limits<std::uint32_t>::max()) {
        throw ArtifactError(std::string(label) + ": config dimension exceeds u32");
    }
    return static_cast<std::uint32_t>(n);
}

std::uint32_t dimension(const Json& config, const char* name) {
    return integer(config.at(name), name);
}

float positive_float(const Json& config, const char* name) {
    const auto& value = config.at(name);
    if (!value.is_number()) { throw ArtifactError(std::string(name) + " must be a real value"); }
    const float result = value.get<float>();
    if (!std::isfinite(result) || result <= 0) {
        throw ArtifactError(std::string(name) + " must be positive finite FP32");
    }
    return result;
}

// A recipe-pinned boolean. The engine implements one value for these, so the artifact's
// declaration is checked here and not carried into the config.
void require_flag(const Json& config, const char* name, bool expected) {
    const auto& value = config.at(name);
    if (!value.is_boolean()) {
        throw ArtifactError(std::string(name) + " must be a boolean");
    }
    if (value.get<bool>() != expected) {
        throw ArtifactError(std::string(name) + " must be " + (expected ? "true" : "false"));
    }
}

// A recipe-pinned enumeration: only the values the engine has a path for are accepted, and a
// rejection names the artifact's declaration next to the supported set.
std::string require_mode(const Json& config, const char* name,
                         std::initializer_list<std::string_view> supported) {
    const auto& value = config.at(name);
    if (!value.is_string()) {
        throw ArtifactError(std::string(name) + " must be a string");
    }
    const auto declared = value.get<std::string>();
    for (const auto candidate : supported) {
        if (declared == candidate) { return declared; }
    }
    std::string names;
    for (const auto candidate : supported) {
        if (!names.empty()) { names += ", "; }
        names += std::string(candidate);
    }
    throw ArtifactError(std::string(name) + " is \"" + declared + "\", which this engine does not "
                        "implement (supported: " + names + ")");
}

std::string architecture(const Json& config) {
    const auto& names = config.at("architectures");
    if (!names.is_array() || names.size() != 1) {
        throw ArtifactError("architectures must name one implementation");
    }
    return artifact::require_id(names[0], "architecture");
}

} // namespace

Config parse_config(const artifact::Component& component, TargetGeometry target) {
    if (component.target != "text") throw ArtifactError("draft component target must be text");
    const auto& value = component.config;
    // The vocabulary is the converter's (tools/convert/qwen3_8_27b/dspine_recipe.py:
    // _ROOT_CONFIG + _SECOND_PASS_CONFIG): it is an allowlist, so a member the recipe never
    // emits is still an unknown member and fails.
    require_members(
        value,
        {"architectures", "model_type", "hidden_size", "intermediate_size",
         "num_attention_heads", "num_key_value_heads", "head_dim", "num_hidden_layers",
         "max_position_embeddings", "rms_norm_eps", "rope_parameters", "layer_types",
         "vocab_size", "num_target_layers", "block_size", "is_causal",
         "code_bridge_second_pass", "dflash_config"},
        {"sliding_window"}, "dspine config");
    if (architecture(value) != "DSpineDraftModel" || value.at("model_type") != "qwen3") {
        throw ArtifactError("dspine architecture/model_type mismatch");
    }
    Config out;
    out.hidden_size             = dimension(value, "hidden_size");
    out.intermediate_size       = dimension(value, "intermediate_size");
    out.num_hidden_layers       = dimension(value, "num_hidden_layers");
    out.num_attention_heads     = dimension(value, "num_attention_heads");
    out.num_key_value_heads     = dimension(value, "num_key_value_heads");
    out.head_dim                = dimension(value, "head_dim");
    out.vocab_size              = dimension(value, "vocab_size");
    out.max_position_embeddings = dimension(value, "max_position_embeddings");
    out.rms_norm_eps            = positive_float(value, "rms_norm_eps");
    const auto& rope = value.at("rope_parameters");
    require_members(rope, {"rope_theta"}, {"rope_type"}, "dspine RoPE");
    out.rope_theta = positive_float(rope, "rope_theta");
    // The draft forward rotates with the plain RoPE the target uses; a scaled rope on the draft
    // has no path here.
    if (rope.contains("rope_type")) { (void)require_mode(rope, "rope_type", {"default"}); }
    const auto& layers = value.at("layer_types");
    if (!layers.is_array() || layers.size() != out.num_hidden_layers) {
        throw ArtifactError("dspine layer_types length mismatch");
    }
    for (const auto& layer : layers) {
        if (layer != "sliding_attention" && layer != "full_attention") {
            throw ArtifactError("unknown dspine layer_type");
        }
    }
    if (value.contains("sliding_window")) {
        out.sliding_window = dimension(value, "sliding_window");
    }
    // The draft is causal by construction; the tail's block contract has no non-causal draft.
    require_flag(value, "is_causal", true);
    out.block_size = dimension(value, "block_size");
    if (out.block_size < 2) { throw ArtifactError("dspine block_size must be at least 2"); }
    // The head top-k the bridge names for its refinement; code_bridge_second_pass and
    // selected_prefix_last_write_refine must agree on one width.
    std::uint32_t refinement_width = 0;
    const auto& second_pass        = value.at("code_bridge_second_pass");
    require_members(second_pass, {"first_exact", "head_topk", "mode"}, {},
                    "DSpine code-bridge second pass");
    out.second_pass_mode = require_mode(second_pass, "mode", {"serial_table"});
    // "serial_table" redoes the LAST draft layer only: the layers below it run once, exactly, so
    // first_exact is a draft-layer count equal to num_hidden_layers - 1 (the reference's
    // "serial_table must redo the last layer only", dspine_mlx/dspine_mlx/config.py:219).
    out.second_pass_first_exact = integer(second_pass.at("first_exact"), "first_exact", false);
    if (out.second_pass_first_exact != out.num_hidden_layers - 1) {
        throw ArtifactError("code_bridge_second_pass.first_exact must be the draft layer count "
                            "minus one: serial_table redoes the last layer only");
    }
    out.second_pass_head_topk = dimension(second_pass, "head_topk");
    refinement_width          = out.second_pass_head_topk;
    const auto& config = value.at("dflash_config");
    // The recipe's inference-relevant dflash_config members (_DRAFT_CONFIG); its training-only
    // fields never reach the artifact.
    require_members(config,
                    {"target_layer_ids", "mask_token_id", "selected_prefix_code_bridge",
                     "selected_prefix_feature_layers"},
                    {"selected_prefix_last_write_refine", "draft_block_causal",
                     "selected_prefix_decode_mode", "selected_prefix_feature_write_mode",
                     "selected_prefix_feature_writes", "selected_prefix_terminal"},
                    "DSpine dflash_config");
    const auto& taps = config.at("target_layer_ids");
    if (!taps.is_array() || taps.empty()) {
        throw ArtifactError("target_layer_ids must be nonempty");
    }
    std::set<std::uint32_t> unique;
    for (const auto& tap : taps) {
        const auto index = integer(tap, "target layer index", false);
        if (index >= target.layers || !unique.insert(index).second) {
            throw ArtifactError("invalid target layer index");
        }
        out.target_layer_ids.push_back(index);
    }
    out.mask_token_id = integer(config.at("mask_token_id"), "mask token", false);
    if (out.mask_token_id >= target.vocabulary) {
        throw ArtifactError("mask token exceeds target embedding rows");
    }
    out.num_target_layers = dimension(value, "num_target_layers");
    if (out.num_target_layers != target.layers) {
        throw ArtifactError("dspine num_target_layers differs from the target");
    }
    const auto& bridge = config.at("selected_prefix_code_bridge");
    require_members(bridge, {"code_size", "rank", "eps"}, {"kappa"}, "DSpine code bridge");
    out.bridge.code_size = dimension(bridge, "code_size");
    out.bridge.rank      = dimension(bridge, "rank");
    out.bridge.eps       = positive_float(bridge, "eps");
    if (bridge.contains("kappa")) { out.bridge.kappa = positive_float(bridge, "kappa"); }
    const auto& features = config.at("selected_prefix_feature_layers");
    if (!features.is_array() || features.empty()) {
        throw ArtifactError("selected_prefix_feature_layers must be nonempty");
    }
    // The checkpoint numbers the bridge's feature writes from 1: the reference converts with
    // `- 1` (dspine_mlx/dspine_mlx/config.py:193, sglang_dspine/models/dspine.py:112), and the
    // port requires a write on EVERY draft layer - bind_weights binds one
    // feature_write_weights row per layer, so a partial set has no binding.
    for (const auto& layer : features) {
        const auto written = integer(layer, "feature layer index");
        if (written > out.num_hidden_layers) {
            throw ArtifactError("feature layer index exceeds draft layers");
        }
        out.feature_layers.push_back(written - 1);
    }
    std::sort(out.feature_layers.begin(), out.feature_layers.end());
    if (out.feature_layers.size() != out.num_hidden_layers ||
        out.feature_layers.front() != 0 ||
        out.feature_layers.back() + 1 != out.num_hidden_layers) {
        throw ArtifactError("DSpine requires a bridge write on every draft layer");
    }
    if (config.contains("selected_prefix_last_write_refine")) {
        const auto& refine = config.at("selected_prefix_last_write_refine");
        require_members(refine, {"head_topk"}, {"kind", "version", "weight"},
                        "DSpine last-write refine");
        // The predecessor-gate refine the bridge implements: version 1 of the exact-sender row.
        out.head_topk = dimension(refine, "head_topk");
        if (refine.contains("kind")) { (void)require_mode(refine, "kind", {"exact_sender"}); }
        if (refine.contains("version") && integer(refine.at("version"), "refine version") != 1) {
            throw ArtifactError("selected_prefix_last_write_refine version must be 1");
        }
        if (refine.contains("weight")) {
            const auto& weight = refine.at("weight");
            if (!weight.is_number() || !std::isfinite(weight.get<float>()) ||
                weight.get<float>() < 0) {
                throw ArtifactError("selected_prefix_last_write_refine weight must be a "
                                    "nonnegative finite FP32");
            }
        }
        if (refinement_width && out.head_topk != refinement_width) {
            throw ArtifactError("the bridge's refinement widths disagree: last_write_refine "
                                "names " + std::to_string(out.head_topk) +
                                ", code_bridge_second_pass names " +
                                std::to_string(refinement_width));
        }
        refinement_width = out.head_topk;
    }
    if (!refinement_width) { throw ArtifactError("the bridge names no refinement width"); }
    // The serial tail scores kSerialTailSlots candidates per slot; a draft trained for another
    // width has no kernel here.
    if (refinement_width !=
        static_cast<std::uint32_t>(common::dspine::kSerialTailSlots)) {
        throw ArtifactError("the serial tail scores " +
                            std::to_string(common::dspine::kSerialTailSlots) +
                            " candidates per slot; the draft names head_topk " +
                            std::to_string(refinement_width));
    }
    out.head_topk = refinement_width;
    // The recipe pins the rest of the bridge's behaviour, and each pin is a shape of the path
    // this engine binds: a block-causal draft, one independent feature write per draft layer,
    // serial bridge decoding, and no terminal code (the tail ends on the codebook walk).
    if (config.contains("draft_block_causal")) { require_flag(config, "draft_block_causal", true); }
    if (config.contains("selected_prefix_feature_writes")) {
        require_flag(config, "selected_prefix_feature_writes", true);
    }
    if (config.contains("selected_prefix_terminal")) {
        require_flag(config, "selected_prefix_terminal", false);
    }
    if (config.contains("selected_prefix_decode_mode")) {
        out.decode_mode = require_mode(config, "selected_prefix_decode_mode", {"serial"});
    }
    if (config.contains("selected_prefix_feature_write_mode")) {
        (void)require_mode(config, "selected_prefix_feature_write_mode", {"independent"});
    }
    return out;
}

} // namespace ninfer::models::dspine
