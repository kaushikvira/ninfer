#include "models/common/dspine/weights.h"
#include "models/common/bindings.h"

#include "artifact/formats.h"
#include "artifact/reader.h"

#include <array>
#include <span>
#include <string>
#include <vector>

namespace ninfer::models::dspine {
namespace {

// A vocabulary-indexed bridge table is BF16 in the reference artifact and NVFP4 block-scaled in
// the sized one. ops::dspine_normalize_codes and ops::dspine_serial_scores validate the
// representation they receive, so the binding follows the artifact's declaration within this set.
constexpr std::array kBridgeTableFormats{QType::BF16, QType::NVFP4};

// One projection of a packed draft-layer group: the manifest's binding name and the rows it
// contributes to the packed object, in the packed object's row order.
struct SplitProjection {
    std::string name;
    std::uint64_t rows = 0;
};

// The V3 recipe binds a draft layer's Q, K and V projections - and its gate and up projections -
// as ranges of ONE packed object, and the two context projections alias the K and V ranges. The
// loader follows that: it binds the named ranges, so the packed object is materialized once and
// the drafter runs the projections apart over it. A copy into a fused buffer is never taken.
//
// Binding the names is only half of following a packed group. The ranges are checked against the
// object they name: same object for the whole group, adjacent in the group's order, and covering
// every element of it. A manifest that names the split ranges of some other layout - a hole, a
// reordered group, a range that is not what the drafter projects - is refused by name, and the
// bytes the drafter reads are then known to be the packed tensor it expects.
void require_packed_group(const Bindings& b, std::span<const WeightId> ids,
                          std::span<const SplitProjection> projections, std::uint64_t columns) {
    if (ids.size() != projections.size()) {
        throw artifact::ArtifactError("dspine packed group: the bound weights and the packed "
                                      "layout describe different groups");
    }
    std::optional<artifact::ObjectHandle> packed;
    std::uint64_t covered = 0;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        const auto& name    = b.at(ids[i]).reference.name;
        const auto& binding = b.at(ids[i]).reference.binding;
        const auto rows     = projections[i].rows;
        if (binding.elements != artifact::checked_mul(rows, columns, "packed projection rows")) {
            throw artifact::ArtifactError(name + ": covers " +
                                          std::to_string(binding.elements) +
                                          " elements, the rows of a " + std::to_string(rows) +
                                          " by " + std::to_string(columns) + " projection");
        }
        for (const auto& part : binding.parts) {
            if (packed && part.object != *packed) {
                throw artifact::ArtifactError(
                    name + ": a packed group must share one object, and this range is in another");
            }
            if (part.begin != covered) {
                throw artifact::ArtifactError(name + ": a packed group covers its object in "
                                              "order, and this range begins at element " +
                                              std::to_string(part.begin) + ", not " +
                                              std::to_string(covered));
            }
            packed  = part.object;
            covered = part.end;
        }
    }
    if (!packed) {
        throw artifact::ArtifactError("dspine packed group: no projection names an object");
    }
    const auto total = b.binder.reader().geometry(*packed).elements;
    if (covered != total) {
        throw artifact::ArtifactError(b.at(ids.front()).reference.name +
                                      ": the packed group covers " + std::to_string(covered) +
                                      " of the " + std::to_string(total) +
                                      " elements of its object");
    }
}

// A draft layer bound as the packed names is a manifest from the recipe's fused naming, which
// this engine does not bind. Refusing it by name - the packed name and the split names it would
// need - is the defined other case: the loader never guesses which layout it is being handed.
void require_split_manifest(const Bindings& b, std::string_view packed,
                            std::span<const std::string_view> split_names) {
    if (!b.binder.contains(packed)) { return; }
    std::string names;
    for (const auto& name : split_names) {
        if (!names.empty()) { names += ", "; }
        names += name;
    }
    throw artifact::ArtifactError(std::string(packed) +
                                  ": a draft layer bound as one projection is not a layout this "
                                  "engine binds, which requires " +
                                  names);
}

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

        require_split_manifest(
            b, p + "attention/query_key_value",
            std::array<std::string_view, 3>{p + "attention/query", p + "attention/key",
                                           p + "attention/value"});
        layer.attention.query =
            b.parameter(p + "attention/query", {q, h}, {p + "query_projection_input"});
        layer.attention.key =
            b.parameter(p + "attention/key", {k, h}, {p + "query_projection_input"});
        layer.attention.value =
            b.parameter(p + "attention/value", {k, h}, {p + "query_projection_input"});
        const std::array qkv{SplitProjection{"query", q}, SplitProjection{"key", k},
                              SplitProjection{"value", k}};
        require_packed_group(b,
                              std::array{layer.attention.query, layer.attention.key,
                                         layer.attention.value},
                              qkv, h);

        // The context projections of a draft layer read the same rows as its K and V projections;
        // the artifact says so by binding both names to the same range, and the binding keeps
        // that promise rather than paying for it twice.
        layer.attention.context_key =
            b.parameter(p + "attention/context_key", {k, h}, {component + "/context_input"});
        require_shared_binding(b.at(layer.attention.context_key).reference,
                               b.at(layer.attention.key).reference);
        layer.attention.context_value =
            b.parameter(p + "attention/context_value", {k, h}, {component + "/context_input"});
        require_shared_binding(b.at(layer.attention.context_value).reference,
                               b.at(layer.attention.value).reference);

        layer.attention.query_norm = b.direct(p + "attention/query_norm", {d});
        layer.attention.key_norm   = b.direct(p + "attention/key_norm", {d});
        layer.attention.output =
            b.parameter(p + "attention/output", {h, q}, {p + "attention_output"});

        const auto intermediate = config.intermediate_size;
        require_split_manifest(b, p + "mlp/gate_up",
                               std::array<std::string_view, 2>{p + "mlp/gate", p + "mlp/up"});
        layer.mlp.gate = b.parameter(p + "mlp/gate", {intermediate, h}, {p + "mlp_input"});
        layer.mlp.up   = b.parameter(p + "mlp/up", {intermediate, h}, {p + "mlp_input"});
        const std::array gate_up{SplitProjection{"gate", intermediate},
                                  SplitProjection{"up", intermediate}};
        require_packed_group(b, std::array{layer.mlp.gate, layer.mlp.up}, gate_up, h);
        layer.mlp.down =
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
