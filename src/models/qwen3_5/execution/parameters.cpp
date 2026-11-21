#include "models/qwen3_5/execution/parameters.h"

#include "core/weight_view.h"
#include "ninfer/ops/linear_split.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::qwen3_5::execution {
namespace {

template <class Function>
auto with_context(const std::string& context, Function&& function) {
    try {
        return function();
    } catch (const std::invalid_argument& error) {
        throw std::invalid_argument(context + ": " + error.what());
    }
}

class Prepare {
public:
    Prepare(const ModelShard& model, const ModelProfile& profile)
        : model_(model), profile_(profile) {}

    LinearParameters linear(WeightId id) const {
        return with_context(model_.weight(id).name,
                            [&] { return ops::prepare_linear_weight(model_.input(id)); });
    }

    LinearParameters linear(WeightUseId id) const {
        return with_context(model_.weight(id.parameter).name,
                            [&] { return ops::prepare_linear_weight(model_.input(id)); });
    }

    Tensor tensor(WeightId id) const {
        const auto& bound = model_.weight(id);
        return with_context(bound.name, [&] { return parameter_tensor(bound); });
    }

    DenseParameters dense(const DenseWeights& w) const {
        return {with_context(model_.weight(w.gate).name,
                             [&] {
                                 return ops::prepare_linear_swiglu_weight(model_.input(w.gate),
                                                                          model_.input(w.up));
                             }),
                linear(w.down)};
    }

    FfnParameters ffn(const BlockWeights& w) const {
        if (const auto* d = std::get_if<DenseWeights>(&w.ffn)) { return dense(*d); }
        return moe(std::get<MoeWeights>(w.ffn));
    }

    moe::Parameters moe(const MoeWeights& w) const {
        return with_context(model_.weight(w.router).name, [&] {
            const auto& source = model_.expert_banks()[w.bank.index];
            ops::ExpertInputUsage gate_up, down;
            for (const auto& expert : w.experts) {
                gate_up.include(model_.input(expert.gate));
                gate_up.include(model_.input(expert.up));
                down.include(model_.input(expert.down));
            }
            moe::Parameters out;
            const std::array control{model_.input(w.router), model_.input(w.shared_score)};
            out.control = ops::prepare_linear_weight(control).weight;
            out.shared  = {ops::prepare_linear_swiglu_weight(model_.input(w.shared.gate),
                                                             model_.input(w.shared.up)),
                           linear(w.shared.down)};
            out.profile = {
                ops::routed_swiglu_profile(
                    source.gate_up, source.down,
                    dimension(std::get<MoeConfig>(profile_.config.text.ffn).num_experts_per_tok),
                    gate_up.policy(), down.policy()),
                ops::swiglu_profile(out.shared)};
            out.bank                   = w.bank;
            out.activation_calibration = {gate_up.calibration(source.gate_up.layout.format),
                                          down.calibration(source.down.layout.format)};
            if (source.gate_up.parent->device.type() == DeviceType::CUDA) {
                out.resident = device_expert_bank(source.gate_up, source.down);
            }
            return out;
        });
    }

    BlockParameters block(const BlockWeights& w) const {
        BlockParameters out;
        out.input_norm          = tensor(w.input_norm);
        out.post_attention_norm = tensor(w.post_attention_norm);
        out.ffn                 = ffn(w);
        if (const auto* a = std::get_if<AttentionWeights>(&w.mixer)) {
            const std::array inputs{model_.input(a->query), model_.input(a->key),
                                    model_.input(a->gate), model_.input(a->value)};
            const std::array rows{
                dimension(inputs[0].weight.shape[0]), dimension(inputs[1].weight.shape[0]),
                dimension(inputs[2].weight.shape[0]), dimension(inputs[3].weight.shape[0])};
            auto projection = ops::prepare_projection_weights(inputs);
            (void)ops::linear_split_workspace_capacity_bytes(projection, rows, 1, 1);
            out.mixer = AttentionParameters{std::move(projection), rows, tensor(a->query_norm),
                                            tensor(a->key_norm), linear(a->output)};
            out.projection_prefetch =
                prefetch(std::get<AttentionParameters>(out.mixer).projection, a->query);
        } else {
            const auto& g = std::get<GdnWeights>(w.mixer);
            const std::array inputs{model_.input(g.query), model_.input(g.key),
                                    model_.input(g.value), model_.input(g.z)};
            const std::array rows{dimension(inputs[0].weight.shape[0] + inputs[1].weight.shape[0] +
                                            inputs[2].weight.shape[0]),
                                  dimension(inputs[3].weight.shape[0])};
            auto projection = ops::prepare_projection_weights(inputs);
            (void)ops::linear_split_workspace_capacity_bytes(projection, rows, 1, 1);
            out.mixer =
                GdnParameters{std::move(projection),
                              rows,
                              ops::prepare_gdn_control_proj_weights(model_.input(g.a_projection),
                                                                    model_.input(g.b_projection)),
                              tensor(g.a_log),
                              tensor(g.dt_bias),
                              tensor(g.convolution),
                              tensor(g.norm),
                              linear(g.output)};
            out.projection_prefetch =
                prefetch(std::get<GdnParameters>(out.mixer).projection, g.query);
        }
        return out;
    }

    ops::MoeExecutionHints prefetch(const ops::ProjectionWeights& projection,
                                    WeightId query) const {
        const auto* single   = std::get_if<LinearParameters>(&projection);
        const auto& weight   = single
                                   ? single->weight
                                   : std::get<ops::PairedProjectionWeights>(projection).first.weight;
        const auto& geometry = model_.weight(query).view.parts.front().parent->geometry;
        const auto row_bytes = geometry.layout == QuantLayout::Contiguous
                                   ? std::uint64_t(weight.k) * dtype_size(DType::BF16)
                                   : geometry.code_bytes_per_row;
        return {weight.qdata, static_cast<std::size_t>(row_bytes * weight.n)};
    }

    MtpParameters mtp(const MtpWeights& w) const {
        const auto& a = std::get<AttentionWeights>(w.layer.mixer);
        const std::array inputs{model_.input(a.query), model_.input(a.key), model_.input(a.gate),
                                model_.input(a.value)};
        MtpParameters out;
        out.input_projection       = linear(w.input_projection);
        out.embedding_norm         = tensor(w.embedding_norm);
        out.hidden_norm            = tensor(w.hidden_norm);
        out.input_norm             = tensor(w.layer.input_norm);
        out.post_attention_norm    = tensor(w.layer.post_attention_norm);
        out.final_norm             = tensor(w.final_norm);
        out.projection.packed      = ops::prepare_linear_weight(inputs);
        out.projection.output_rows = {
            dimension(inputs[0].weight.shape[0]), dimension(inputs[1].weight.shape[0]),
            dimension(inputs[2].weight.shape[0]), dimension(inputs[3].weight.shape[0])};
        if (profile_.config.text.architecture == Architecture::Qwen3_5) {
            out.projection.rows = {linear(a.query), linear(a.key), linear(a.gate), linear(a.value)};
        } else {
            (void)ops::linear_split_workspace_capacity_bytes(out.projection.packed,
                                                             out.projection.output_rows, 1, 1);
        }
        out.query_norm  = tensor(a.query_norm);
        out.key_norm    = tensor(a.key_norm);
        out.output      = linear(a.output);
        out.ffn         = ffn(w.layer);
        out.output_head = linear(w.output_head_use);
        return out;
    }

    qwen::VisionParameters vision(const qwen::VisionWeights& weights) const {
        return qwen::prepare_vision(model_.weight_data(), weights);
    }

    DynamicConvParameters convolution(const dflash::DynamicConvWeights& w) const {
        return {tensor(w.base_kernel), linear(w.kernel_projection)};
    }

    // One matrix of a STACKED parameter: layer `index`'s [rows, columns] row range of the one
    // materialized [depth, rows, columns] object. The DSpine bridge's per-layer read and write
    // weights are stacked this way in the shipped artifacts, and a slice of the materialized
    // object is the only form that takes no copy: the same construction a weight bank uses for
    // one expert's matrix (core/weight_bank.cpp:matrix_weight).
    Weight stacked(WeightId id, std::uint64_t index, std::uint64_t rows,
                    std::uint64_t columns) const {
        const auto& bound = model_.weight(id);
        return with_context(bound.name, [&] {
            const auto& view = bound.view;
            if (view.shape.size() != 3 || view.parts.size() != 1) {
                throw std::invalid_argument(
                    "a stacked bridge weight is one contiguous [depth, rows, columns] object");
            }
            const auto* parent = view.parts.front().parent;
            if (parent == nullptr || parent->data == nullptr) {
                throw std::invalid_argument("its stacked object is not resident on the device");
            }
            const auto begin = index * rows * columns;
            if (view.shape[0] <= index || begin + rows * columns > view.parts.front().end) {
                throw std::invalid_argument("the requested layer leaves its stacked object");
            }
            return native_weight({{rows, columns}, {{parent, begin, begin + rows * columns}}},
                                 1.0F);
        });
    }

    DraftParameters draft(const dflash::Weights& w) const {
        DraftParameters out;
        out.feature_projection = linear(w.feature_projection);
        out.context_norm       = tensor(w.context_norm);
        out.final_norm         = tensor(w.final_norm);
        out.output_head        = linear(w.output_head_use);
        out.layers.reserve(w.layers.size());
        for (std::size_t i = 0; i < w.layers.size(); ++i) {
            out.layers.push_back(
                with_context(std::string(profile_.options.speculative_component()) + "/layers/" +
                                 std::to_string(i),
                             [&] {
                                 const auto& layer = w.layers[i];
                                 const auto& a     = layer.attention;
                                 DraftBlockParameters result;
                                 result.input_norm          = tensor(layer.input_norm);
                                 result.post_attention_norm = tensor(layer.post_attention_norm);
                                 const std::array inputs{model_.input(a.query), model_.input(a.key),
                                                         model_.input(a.value)};
                                 result.query_key_value = ops::prepare_linear_weight(inputs);
                                 const std::array rows{dimension(inputs[0].weight.shape[0]),
                                                       dimension(inputs[1].weight.shape[0]),
                                                       dimension(inputs[2].weight.shape[0])};
                                 (void)ops::linear_split_workspace_capacity_bytes(
                                     result.query_key_value, rows, 1, 1);
                                 result.context_key   = linear(a.context_key);
                                 result.context_value = linear(a.context_value);
                                 result.query_norm    = tensor(a.query_norm);
                                 result.key_norm      = tensor(a.key_norm);
                                 result.output        = linear(a.output);
                                 result.mlp           = dense(layer.mlp);
                                 if (layer.attention_conv) {
                                     result.attention_conv = convolution(*layer.attention_conv);
                                 }
                                 if (layer.mlp_conv) {
                                     result.mlp_conv = convolution(*layer.mlp_conv);
                                 }
                                 return result;
                             }));
        }
        if (w.selector) {
            out.selector = SelectorParameters{linear(w.selector->hidden_projection),
                                              tensor(w.selector->predecessor_codebook),
                                              tensor(w.selector->successor_codebook)};
        }
        return out;
    }

    // The DSpine drafter, in the form the executor's proposal round runs it in. The tail's four
    // streams (final norm, gate bias, tg, tsv) are what execution::dspine_propose_tail names; the
    // WALK half - the target-tap feature projection, the context norm, the five draft layers and
    // the code bridge - is what execution::dspine_propose walks, ported from the probe that walks
    // the same objects (tests/models/qwen3_5/test_dspine_forward_real.cpp).
    //
    // Nothing here copies a weight. A layer of a STACKED bridge object is one row range of the
    // one materialized object (`stacked` below, the core/weight_bank.cpp:matrix_weight form), and
    // a packed Q/K/V group is prepared as the split family's packed parent exactly as the DFlash2
    // draft arm above prepares its own. The token embedding is the tower's, aliased: the drafter
    // carries no copy of it (D-37), and the proposal head is `text.output_head`, which the round
    // reads through the tail's own view.
    DspineParameters dspine(const dspine::Weights& w) const {
        DspineParameters out;
        out.final_norm        = tensor(w.final_norm);
        out.gate_bias         = tensor(w.bridge.gate_bias);
        out.tg                = native_weight(model_.weight(w.bridge.tg).view);
        out.tsv               = native_weight(model_.weight(w.bridge.tsv).view);
        out.feature_projection = linear(w.feature_projection);
        out.context_norm       = tensor(w.context_norm);
        // The drafter embeds its block with the TOWER's own token embedding; the definition
        // aliases the id (qwen3_5/definition.cpp:98), so this names the tower's one materialized
        // object and the drafter carries no copy of it (D-37).
        out.token_embedding = native_weight(model_.weight(w.token_embedding).view);
        out.bridge.codebook = native_weight(model_.weight(w.bridge.codebook).view);
        out.bridge.gate             = linear(w.bridge.gate);
        out.bridge.predecessor_gate = linear(w.bridge.predecessor_gate);
        out.bridge.prev             = linear(w.bridge.prev);

        const auto layers = std::uint64_t(w.layers.size());
        const auto read   = model_.weight(w.bridge.read_weight).view.shape;          // [code, h]
        const auto write  = model_.weight(w.bridge.feature_write_weights).view.shape; // [L, h, r]
        if (write[0] != layers || read.size() != 2 || write.size() != 3) {
            throw std::invalid_argument("the DSpine bridge's stacked objects are outside the "
                                        "[layers, code, hidden] / [layers, hidden, rank] form");
        }
        // Layer i reads through row i of the one `early_read_weights` object, and the LAST
        // layer through the artifact's own `read_weight` - which is exactly why
        // models/common/dspine/weights.cpp binds that one [layers-1, code, hidden]: the vector
        // is indexed by layer, and every entry is one row range of an object that was
        // materialized once.
        for (std::size_t layer = 0; layer + 1 < layers; ++layer)
            out.bridge.read.push_back(
                stacked(w.bridge.early_read_weights, layer, read[0], read[1]));
        out.bridge.read.push_back(linear(w.bridge.read_weight).weight);
        for (std::size_t layer = 0; layer < layers; ++layer)
            out.bridge.write.push_back(
                stacked(w.bridge.feature_write_weights, layer, write[2], write[1]));

        out.layers.reserve(w.layers.size());
        for (std::size_t i = 0; i < w.layers.size(); ++i) {
            out.layers.push_back(with_context(
                std::string(profile_.options.speculative_component()) + "/layers/"
                    + std::to_string(i),
                [&] {
                    const auto& layer = w.layers[i];
                    const auto& a     = layer.attention;
                    DspineBlockParameters block;
                    block.input_norm          = tensor(layer.input_norm);
                    block.post_attention_norm = tensor(layer.post_attention_norm);
                    // One packed object carries the block's Q, K and V ranges, and the two
                    // context projections alias the K and V ranges (weights.cpp binds them with
                    // require_shared_binding), so the walk reaches every one of them - the block
                    // split AND the context split - through this one packed parent.
                    const std::array qkv{model_.input(a.query), model_.input(a.key),
                                         model_.input(a.value)};
                    block.query_key_value = ops::prepare_linear_weight(qkv);
                    const std::array rows{dimension(qkv[0].weight.shape[0]),
                                          dimension(qkv[1].weight.shape[0]),
                                          dimension(qkv[2].weight.shape[0])};
                    (void)ops::linear_split_workspace_capacity_bytes(block.query_key_value, rows,
                                                                     1, 1);
                    block.query_norm = tensor(a.query_norm);
                    block.key_norm   = tensor(a.key_norm);
                    block.output     = linear(a.output);
                    block.mlp        = dense(layer.mlp);
                    return block;
                }));
        }
        return out;
    }

private:
    const ModelShard& model_;
    const ModelProfile& profile_;
};

} // namespace

Parameters::Parameters(const ModelShard& source, std::shared_ptr<const ModelProfile> profile)
    : model(source), profile(std::move(profile)) {
    if (!this->profile) throw std::invalid_argument("Parameters require an execution profile");
    const Prepare prepare(model, *this->profile);
    const auto& w        = model.definition().weights();
    text.token_embedding = native_weight(model.weight(w.text.token_embedding).view);
    text.output_head     = prepare.linear(w.text.output_head_use);
    text.final_norm      = prepare.tensor(w.text.final_norm);
    text.layers.reserve(w.text.layers.size());
    for (std::size_t i = 0; i < w.text.layers.size(); ++i) {
        text.layers.push_back(with_context("text/layers/" + std::to_string(i),
                                           [&] { return prepare.block(w.text.layers[i]); }));
    }
    if (w.mtp) {
        mtp = with_context("mtp", [&] { return prepare.mtp(*w.mtp); });
    }
    if (w.vision) {
        vision = with_context("vision", [&] { return prepare.vision(*w.vision); });
    }
    if (w.draft) {
        draft = with_context(std::string(this->profile->options.speculative_component()),
                             [&] { return prepare.draft(*w.draft); });
    }
    if (w.dspine) {
        dspine = with_context(std::string(this->profile->options.speculative_component()),
                              [&] { return prepare.dspine(*w.dspine); });
    }
    if (w.proposal) {
        proposal =
            ProposalParameters{prepare.linear(w.proposal->head), std::nullopt, w.proposal->rows};
        if (w.proposal->token_ids) { proposal->token_ids = prepare.tensor(*w.proposal->token_ids); }
    }
}

} // namespace ninfer::models::qwen3_5::execution
