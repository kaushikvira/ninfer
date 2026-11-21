#include "models/qwen3_5/definition.h"

#include "models/qwen3_5/load/bindings.h"
#include "models/qwen3_5/expert_weights.h"

#include <unordered_set>

namespace ninfer::models::qwen3_5 {

struct ModelDefinition::Impl {
    artifact::PreparedResources prepared_resources;
    artifact::ArtifactId artifact_id;
    artifact::MaterializationStats resource_stats;
    Config config;
    LoadOptions options;
    ModelWeights weights;
    qwen::FrontendResources resources;
    InstanceInfo info;
    std::vector<artifact::ParameterReference> references;
    std::vector<moe::BankDescription> banks;
    std::vector<WeightParent> parents;
    std::vector<BoundWeight> descriptions;
    std::uint64_t dspine_bytes = 0;
};

ModelDefinition::ModelDefinition(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

ModelDefinition::~ModelDefinition() = default;

const Config& ModelDefinition::config() const { return impl_->config; }

const LoadOptions& ModelDefinition::options() const { return impl_->options; }

const ModelWeights& ModelDefinition::weights() const { return impl_->weights; }

const qwen::FrontendResources& ModelDefinition::resources() const { return impl_->resources; }

const InstanceInfo& ModelDefinition::info() const { return impl_->info; }

const artifact::ArtifactId& ModelDefinition::artifact_id() const { return impl_->artifact_id; }

const artifact::MaterializationStats& ModelDefinition::resource_stats() const {
    return impl_->resource_stats;
}

std::uint64_t ModelDefinition::dspine_bytes() const noexcept { return impl_->dspine_bytes; }

const artifact::ParameterReference& ModelDefinition::parameter(WeightId id) const {
    return impl_->references.at(id.index);
}

std::span<const WeightUse> ModelDefinition::uses(WeightId id) const {
    return impl_->descriptions.at(id.index).uses;
}

std::span<const BoundWeight> ModelDefinition::weight_data() const { return impl_->descriptions; }

std::span<const moe::BankDescription> ModelDefinition::expert_banks() const { return impl_->banks; }

ops::WeightInput ModelDefinition::input(WeightId id) const {
    return weight_input(impl_->descriptions, id);
}

ops::WeightInput ModelDefinition::input(WeightUseId id) const {
    return weight_input(impl_->descriptions, id);
}

std::shared_ptr<const ModelDefinition> read_definition(const artifact::Reader& reader,
                                                       LoadOptions options) {
    auto out         = std::make_unique<ModelDefinition::Impl>();
    out->artifact_id = reader.artifact_id();
    out->options     = options;
    out->config      = parse_config(reader.directory(), options);
    artifact::Binder binder(reader);
    out->resources = loading::bind_resources(binder, out->config);
    Bindings bindings(binder);
    const auto& text  = out->config.text;
    out->weights.text = loading::bind_text(bindings, text, options);
    if (out->config.vision) {
        out->weights.vision = qwen::bind_vision(bindings, *out->config.vision, text.hidden_size);
    }
    if (out->config.mtp) {
        out->weights.mtp = loading::bind_mtp(bindings, text, out->weights.text);
    }
    if (out->config.draft) {
        out->weights.draft =
            dflash::bind_weights(bindings, *out->config.draft,
                                 {text.hidden_size, text.vocab_size, text.num_hidden_layers},
                                 std::string(options.speculative_component()));
        out->weights.draft->token_embedding = out->weights.text.token_embedding;
        out->weights.draft->output_head     = out->weights.text.output_head;
    }
    if (out->config.dspine) {
        out->weights.dspine =
            dspine::bind_weights(bindings, *out->config.dspine,
                                 {text.hidden_size, text.vocab_size, text.num_hidden_layers},
                                 std::string(options.speculative_component()));
        out->weights.dspine->token_embedding = out->weights.text.token_embedding;
        out->weights.dspine->output_head     = out->weights.text.output_head;
    }
    if (options.proposal_enabled()) {
        const auto& proposal = reader.directory().component("text").proposal;
        if (!proposal) {
            throw artifact::ArtifactError("selected proposal head is absent from artifact");
        }
        out->weights.proposal = loading::bind_proposal(bindings, *proposal, text, options,
                                                       out->resources.text.public_token_count);
        if (out->config.draft && out->config.draft->dflash2) {
            const auto domain =
                proposal->indexed ? proposal->rows : out->resources.text.public_token_count;
            if (out->config.draft->dflash2->selector_top_k > domain) {
                throw artifact::ArtifactError(
                    "proposal domain is smaller than DFlash2 selector_top_k");
            }
        }
        if (out->weights.mtp) { out->weights.mtp->output_head = out->weights.proposal->head; }
        if (out->weights.draft) { out->weights.draft->output_head = out->weights.proposal->head; }
    }
    out->weights.text.output_head_use =
        bindings.use(out->weights.text.output_head, "text/final_hidden");
    if (out->weights.mtp) {
        out->weights.mtp->output_head_use =
            bindings.use(out->weights.mtp->output_head, "mtp/final_hidden");
    }
    if (out->weights.draft) {
        out->weights.draft->output_head_use =
            bindings.use(out->weights.draft->output_head,
                         std::string(options.speculative_component()) + "/final_hidden");
    }
    if (out->weights.dspine) {
        out->weights.dspine->output_head_use =
            bindings.use(out->weights.dspine->output_head,
                         std::string(options.speculative_component()) + "/final_hidden");
        // Byte-budget report: surface the DSpine draft's actual size at load so P2/P3 can size
        // against it. Sum the bytes of all objects referenced by the DSpine weights.
        std::unordered_set<std::size_t> counted;
        const auto& dspine_weights = out->weights.dspine;
        auto count_object = [&](artifact::ObjectHandle handle) {
            if (counted.insert(handle.index).second) {
                out->dspine_bytes += artifact::object_bytes(reader.directory().object(handle));
            }
        };
        auto count_weight = [&](WeightId id) {
            for (const auto& part : bindings.at(id).reference.binding.parts) {
                count_object(part.object);
            }
        };
        count_weight(dspine_weights->feature_projection);
        count_weight(dspine_weights->context_norm);
        count_weight(dspine_weights->final_norm);
        for (const auto& layer : dspine_weights->layers) {
            count_weight(layer.input_norm);
            count_weight(layer.post_attention_norm);
            count_weight(layer.attention.query_key_value);
            count_weight(layer.attention.query_norm);
            count_weight(layer.attention.key_norm);
            count_weight(layer.attention.output);
            count_weight(layer.gate_up);
            count_weight(layer.down);
        }
        count_weight(dspine_weights->bridge.codebook);
        count_weight(dspine_weights->bridge.early_read_weights);
        count_weight(dspine_weights->bridge.read_weight);
        count_weight(dspine_weights->bridge.feature_write_weights);
        count_weight(dspine_weights->bridge.gate);
        count_weight(dspine_weights->bridge.gate_bias);
        count_weight(dspine_weights->bridge.predecessor_gate);
        count_weight(dspine_weights->bridge.prev);
        count_weight(dspine_weights->bridge.tg);
        count_weight(dspine_weights->bridge.tsv);
    }
    auto pending            = std::move(bindings.weights);
    out->banks              = describe_expert_weights(reader, out->weights, pending);
    out->prepared_resources = std::move(binder).finish();
    out->info.name          = reader.directory().metadata.value(
        "name", std::string(architecture_name(text.architecture)));
    out->parents.resize(reader.directory().objects.size());
    out->references.reserve(pending.size());
    out->descriptions.reserve(pending.size());
    for (auto& parameter : pending) {
        WeightView view{parameter.reference.shape, {}};
        for (const auto& part : parameter.reference.binding.parts) {
            auto& parent = out->parents.at(part.object.index);
            if (!parent.geometry.elements) parent.geometry = reader.geometry(part.object);
            view.parts.push_back({&parent, part.begin, part.end});
        }
        out->descriptions.push_back({parameter.reference.name, std::move(parameter.source_objects),
                                     std::move(view), std::move(parameter.uses)});
        out->references.push_back(std::move(parameter.reference));
    }
    out->prepared_resources.source = nullptr;
    auto& stats                    = out->resource_stats;
    stats.file_bytes               = reader.file_bytes();
    stats.read_bytes               = out->prepared_resources.read_bytes;
    stats.owned_value_bytes        = out->prepared_resources.owned_value_bytes;
    stats.host_object_count        = out->prepared_resources.objects.size();
    for (const auto& resource : out->prepared_resources.objects)
        stats.retained_host_bytes += resource.data.size();
    return std::shared_ptr<const ModelDefinition>(new ModelDefinition(std::move(out)));
}

} // namespace ninfer::models::qwen3_5
