#include "models/common/dspine/weights.h"
#include "models/common/bindings.h"

#include "artifact/formats.h"

#include <array>

namespace ninfer::models::dspine {
namespace {

// A vocabulary-indexed bridge table is BF16 in the reference artifact and NVFP4 block-scaled in
// the sized one. ops::dspine_normalize_codes and ops::dspine_serial_scores validate the
// representation they receive, so the binding follows the artifact's declaration within this set.
constexpr std::array kBridgeTableFormats{QType::BF16, QType::NVFP4};

} // namespace

Weights bind_weights(Bindings& b, const Config& config, TargetGeometry target,
                     const std::string& component) {
    const auto h = target.hidden_size;
    const auto q = config.query_width();
    const auto k = config.key_width();
    const auto d = config.head_dim;
    Weights out;
    out.feature_projection = b.parameter(
        component + "/feature_projection",
        {h, artifact::checked_mul(config.target_layer_ids.size(), h, "target features")},
        {component + "/target_features"});
    out.context_norm = b.direct(component + "/context_norm", {h});
    out.final_norm   = b.direct(component + "/final_norm", {h});
    out.layers.reserve(config.num_hidden_layers);
    for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
        const auto p = component + "/layers/" + std::to_string(i) + "/";
        BlockWeights layer;
        layer.input_norm          = b.direct(p + "input_norm", {h});
        layer.post_attention_norm = b.direct(p + "post_attention_norm", {h});
        layer.attention.query_key_value =
            b.parameter(p + "attention/query_key_value",
                        {artifact::checked_add(q, artifact::checked_add(k, k, "KV width"),
                                               "QKV width"),
                         h},
                        {p + "query_projection_input"});
        layer.attention.query_norm = b.direct(p + "attention/query_norm", {d});
        layer.attention.key_norm   = b.direct(p + "attention/key_norm", {d});
        layer.attention.output =
            b.parameter(p + "attention/output", {h, q}, {p + "attention_output"});
        const auto intermediate = config.intermediate_size;
        layer.gate_up = b.parameter(p + "mlp/gate_up",
                                    {artifact::checked_mul(2, intermediate, "gate_up width"), h},
                                    {p + "mlp_input"});
        layer.down =
            b.parameter(p + "mlp/down", {h, intermediate}, {p + "mlp_product"});
        out.layers.push_back(std::move(layer));
    }
    const auto c = config.bridge.code_size;
    const auto r = config.bridge.rank;
    const auto v = target.vocabulary;
    const auto codebook_name = component + "/bridge/codebook";
    const auto tg_name       = component + "/bridge/tg";
    const auto tsv_name      = component + "/bridge/tsv";
    out.bridge.codebook = b.direct_as_declared(codebook_name, {v, c}, kBridgeTableFormats);
    out.bridge.early_read_weights = b.direct(component + "/bridge/early_read_weights",
                                             {config.num_hidden_layers - 1, c, h});
    out.bridge.read_weight = b.direct(component + "/bridge/read_weight", {c, h});
    out.bridge.feature_write_weights =
        b.direct(component + "/bridge/feature_write_weights", {config.num_hidden_layers, h, r});
    out.bridge.gate             = b.direct(component + "/bridge/gate", {r, c});
    out.bridge.gate_bias        = b.direct(component + "/bridge/gate_bias", {r});
    out.bridge.predecessor_gate = b.direct(component + "/bridge/predecessor_gate", {r, c});
    out.bridge.prev             = b.direct(component + "/bridge/prev", {r, c});
    out.bridge.tg  = b.direct_as_declared(tg_name, {v, r}, kBridgeTableFormats);
    out.bridge.tsv = b.direct_as_declared(tsv_name, {v, 2 * r}, kBridgeTableFormats);
    // ops::dspine_serial_scores indexes tg and tsv as one representation (tg.qtype ==
    // tsv.qtype); a mixed pair is an artifact defect, so it fails while binding.
    const auto tg_format  = b.binder.declared_format(tg_name);
    const auto tsv_format = b.binder.declared_format(tsv_name);
    if (tg_format != tsv_format) {
        throw artifact::ArtifactError(tsv_name + " and " + tg_name +
                                      " must share one storage format (tg=" +
                                      std::string(artifact::format_name(tg_format)) + ", tsv=" +
                                      std::string(artifact::format_name(tsv_format)) + ")");
    }
    return out;
}

} // namespace ninfer::models::dspine
