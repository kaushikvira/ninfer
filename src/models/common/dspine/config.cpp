#include "models/common/dspine/config.h"
#include "artifact/schema.h"

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
    require_members(
        value,
        {"architectures", "model_type", "hidden_size", "intermediate_size",
         "num_attention_heads", "num_key_value_heads", "head_dim", "num_hidden_layers",
         "max_position_embeddings", "rms_norm_eps", "rope_parameters", "layer_types",
         "vocab_size", "num_target_layers", "block_size", "dflash_config"},
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
    require_members(value.at("rope_parameters"), {"rope_theta"}, {}, "dspine RoPE");
    out.rope_theta = positive_float(value.at("rope_parameters"), "rope_theta");
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
    out.block_size = dimension(value, "block_size");
    if (out.block_size < 2) { throw ArtifactError("dspine block_size must be at least 2"); }
    const auto& config = value.at("dflash_config");
    require_members(config,
                    {"target_layer_ids", "mask_token_id", "selected_prefix_code_bridge",
                     "selected_prefix_feature_layers"},
                    {"selected_prefix_last_write_refine"}, "DSpine dflash_config");
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
    for (const auto& layer : features) {
        const auto index = integer(layer, "feature layer index", false);
        if (index >= out.num_hidden_layers) {
            throw ArtifactError("feature layer index exceeds draft layers");
        }
        out.feature_layers.push_back(index);
    }
    if (config.contains("selected_prefix_last_write_refine")) {
        const auto& refine = config.at("selected_prefix_last_write_refine");
        require_members(refine, {"head_topk"}, {}, "DSpine last-write refine");
        out.head_topk = dimension(refine, "head_topk");
    }
    return out;
}

} // namespace ninfer::models::dspine
