#include "artifact/binder.h"
#include "artifact/fixture.h"
#include "artifact/formats.h"
#include "artifact/materializer.h"
#include "artifact/views.h"
#include "core/weight_view.h"
#include "models/common/bindings.h"
#include "models/common/dspine/weights.h"
#include "models/qwen3_5/load.h"
#include "ops/common/host/operands.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::models;
namespace q35 = ninfer::models::qwen3_5;
using namespace ninfer::test::artifact_fixture;

Json added_token(int id, const std::string& content) {
    return {{"id", id},        {"content", content}, {"special", true},    {"single_word", false},
            {"lstrip", false}, {"rstrip", false},    {"normalized", false}};
}

// Minimal DSpine draft geometry for the load-side test. The target is the same small
// Qwen3.5 text backbone used by test_loading.cpp; the DSpine component is a 1-layer
// sliding-attention draft with a code bridge and TG/TSV tables.
struct DspineFixture {
    Fixture file;

    // Bridge geometry. The three vocabulary-indexed tables are [vocabulary, C],
    // [vocabulary, R] and [vocabulary, 2R]; an NVFP4 declaration also needs N%128=0 and K%64=0,
    // which kSized satisfies. kReference keeps the shipped BF16 table shape.
    struct BridgeGeometry {
        std::uint64_t vocabulary, code_size, rank;
    };
    static constexpr BridgeGeometry kReference{272, 16, 8};
    static constexpr BridgeGeometry kSized{512, 64, 64};

    BridgeGeometry bridge;
    // What the artifact declares for dspine/bridge/codebook, /tg and /tsv.
    QType table_format       = QType::BF16;
    QuantLayout table_layout = QuantLayout::Contiguous;
    // A single table declared apart from the rest: the mixed and unsupported declarations the
    // binding must reject.
    std::map<std::string, std::pair<QType, QuantLayout>> table_overrides;

    void tensor(const std::string& id, std::vector<std::uint64_t> shape, QType format = QType::BF16,
                QuantLayout layout = QuantLayout::Contiguous) {
        const auto geometry = weight_geometry(format, layout, shape);
        const auto offset   = (file.payload.size() + 255) / 256 * 256;
        file.payload.resize(offset + geometry.bytes);
        if (format == QType::NVFP4) {
            put_word(file.payload, offset + geometry.matrix_scale_offset, 0x3f800000, 4);
        }
        file.root["objects"].push_back({{"id", id},
                                        {"kind", "tensor"},
                                        {"shape", shape},
                                        {"format", artifact::format_name(format)},
                                        {"layout", artifact::layout_name(layout)},
                                        {"offset", offset},
                                        {"bytes", geometry.bytes}});
    }

    void resource(const std::string& role, const std::string& bytes) {
        const auto offset = file.payload.size();
        for (const auto byte : bytes) { file.payload.push_back(std::byte(byte)); }
        file.root["objects"].push_back({{"id", role},
                                        {"kind", "resource"},
                                        {"encoding", "raw_bytes_v1"},
                                        {"offset", offset},
                                        {"bytes", bytes.size()}});
        file.root["components"]["text"]["resources"][role] = role;
    }

    void use(const std::string& name, const std::string& input) {
        file.root["uses"].push_back(
            {{"parameter", name}, {"input", input}, {"activation_policy", "A16Only"}});
    }

    void parameter(const std::string& name, std::vector<std::uint64_t> shape,
                   const std::string& input = {}) {
        tensor(name, shape);
        file.root["bindings"][name] = {{"object", name}};
        if (!input.empty()) { use(name, input); }
    }

    // A bridge table: the same object/binding shape as parameter(), in the representation the
    // fixture declares (BF16 contiguous by default, block-scaled NVFP4 for the sized artifact).
    void table(const std::string& name, std::vector<std::uint64_t> shape) {
        const auto override = table_overrides.find(name);
        const auto format =
            override == table_overrides.end() ? table_format : override->second.first;
        const auto layout =
            override == table_overrides.end() ? table_layout : override->second.second;
        tensor(name, shape, format, layout);
        file.root["bindings"][name] = {{"object", name}};
    }

    // Adds the full DSpine component (config + tensors + bindings) to the fixture.
    void add_dspine_component() {
        const auto h = 128;
        const auto inter = 24;
        const auto heads = 2;
        const auto kv_heads = 1;
        const auto head_dim = 8;
        const auto q = heads * head_dim;
        const auto k = kv_heads * head_dim;
        const auto layers = 2;
        const auto vocab = bridge.vocabulary;
        const auto code_size = bridge.code_size;
        const auto rank = bridge.rank;
        const auto taps = 1; // target_layer_ids size (target has 1 layer)

        file.root["components"]["dspine"] = {
            {"config",
             {{"architectures", {"DSpineDraftModel"}},
              {"model_type", "qwen3"},
              {"hidden_size", h},
              {"intermediate_size", inter},
              {"num_attention_heads", heads},
              {"num_key_value_heads", kv_heads},
              {"head_dim", head_dim},
              {"num_hidden_layers", layers},
              {"max_position_embeddings", 128},
              {"rms_norm_eps", 1e-6},
              {"rope_parameters", {{"rope_theta", 10000}}},
              {"layer_types", {"sliding_attention", "sliding_attention"}},
              {"sliding_window", 64},
              {"vocab_size", vocab},
              {"num_target_layers", 1},
              {"block_size", 4},
              {"dflash_config",
               {{"target_layer_ids", {0}},
                {"mask_token_id", 256},
                {"selected_prefix_code_bridge",
                 {{"code_size", code_size}, {"rank", rank}, {"eps", 1e-6}}},
                {"selected_prefix_feature_layers", {0}},
                {"selected_prefix_last_write_refine", {{"head_topk", 4}}}}}}},
            {"target", "text"}};

        // The output head is shared with the text backbone; the DSpine draft uses it with a
        // dspine/final_hidden input.
        use("text/output_head", "dspine/final_hidden");
        // Backbone linears (NVFP4 in the real artifact; BF16 here for the synthetic stand-in).
        parameter("dspine/feature_projection", {h, taps * h}, "dspine/target_features");
        parameter("dspine/context_norm", {h});
        parameter("dspine/final_norm", {h});
        for (std::uint32_t i = 0; i < layers; ++i) {
            const auto p = "dspine/layers/" + std::to_string(i) + "/";
            parameter(p + "input_norm", {h});
            parameter(p + "post_attention_norm", {h});
            parameter(p + "attention/query_key_value", {q + 2 * k, h}, p + "query_projection_input");
            parameter(p + "attention/query_norm", {head_dim});
            parameter(p + "attention/key_norm", {head_dim});
            parameter(p + "attention/output", {h, q}, p + "attention_output");
            parameter(p + "mlp/gate_up", {2 * inter, h}, p + "mlp_input");
            parameter(p + "mlp/down", {h, inter}, p + "mlp_product");
        }
        // Bridge + TG/TSV tables. codebook/tg/tsv carry the varied representation; the rest of
        // the bridge stays BF16.
        table("dspine/bridge/codebook", {vocab, code_size});
        parameter("dspine/bridge/early_read_weights", {layers - 1, code_size, h});
        parameter("dspine/bridge/read_weight", {code_size, h});
        parameter("dspine/bridge/feature_write_weights", {layers, h, rank});
        parameter("dspine/bridge/gate", {rank, code_size});
        parameter("dspine/bridge/gate_bias", {rank});
        parameter("dspine/bridge/predecessor_gate", {rank, code_size});
        parameter("dspine/bridge/prev", {rank, code_size});
        table("dspine/bridge/tg", {vocab, rank});
        table("dspine/bridge/tsv", {vocab, 2 * rank});
        // Update the declared payload size to include the DSpine tensors.
        file.root["files"][0]["payload_bytes"] = file.payload.size();
    }

    explicit DspineFixture(
        BridgeGeometry bridge                                       = kReference,
        QType table_format                                          = QType::BF16,
        QuantLayout table_layout                                    = QuantLayout::Contiguous,
        std::map<std::string, std::pair<QType, QuantLayout>> table_overrides = {})
        : bridge(bridge), table_format(table_format), table_layout(table_layout),
          table_overrides(std::move(table_overrides)) {
        file.payload.clear();
        file.root = {{"components",
                      {{"text",
                        {{"config",
                          {{"architectures", {"Qwen3_5ForCausalLM"}},
                           {"model_type", "qwen3_5_text"},
                           {"hidden_size", 128},
                           {"vocab_size", bridge.vocabulary},
                           {"num_hidden_layers", 1},
                           {"max_position_embeddings", 128},
                           {"tie_word_embeddings", true},
                           {"rms_norm_eps", 1e-6},
                           {"layer_types", {"full_attention"}},
                           {"num_attention_heads", 2},
                           {"num_key_value_heads", 1},
                           {"head_dim", 8},
                           {"intermediate_size", 24},
                           {"rope_parameters",
                            {{"rope_theta", 10000},
                             {"partial_rotary_factor", 0.5},
                             {"mrope_section", {1, 1, 0}}}}}}}}}},
                     {"objects", Json::array()},
                     {"bindings", Json::object()},
                     {"uses", Json::array()},
                     {"metadata", {{"name", "dspine-load-test"}}}};
        tensor("embedding-storage", {bridge.vocabulary, 128});
        file.root["bindings"]["text/token_embedding"] = {{"object", "embedding-storage"}};
        file.root["bindings"]["text/output_head"]     = {{"object", "embedding-storage"}};
        use("text/output_head", "text/final_hidden");
        parameter("text/final_norm", {128});
        const std::string p = "text/layers/0/";
        parameter(p + "input_norm", {128});
        parameter(p + "post_attention_norm", {128});
        tensor("mixer-storage", {128, 128}, QType::NVFP4, QuantLayout::BlockScaleK16M128x4);
        tensor("calibration", {2}, QType::FP32);
        const auto calibration = file.root["objects"].back()["offset"].get<std::size_t>();
        put_word(file.payload, calibration, 0x40000000, 4);
        put_word(file.payload, calibration + 4, 0x40400000, 4);
        const std::array roles = {"query", "key", "gate", "value"};
        const std::array rows  = {0, 16, 24, 40, 48};
        for (std::size_t i = 0; i < roles.size(); ++i) {
            const auto name             = p + "attention/" + roles[i];
            file.root["bindings"][name] = {
                {"parts", Json::array({{{"object", "mixer-storage"},
                                        {"range", {rows[i] * 128, rows[i + 1] * 128}}}})}};
            use(name, p + "mixer_input");
        }
        parameter(p + "attention/query_norm", {8});
        parameter(p + "attention/key_norm", {8});
        parameter(p + "attention/output", {128, 16}, p + "attention/gated_output");
        parameter(p + "mlp/gate", {24, 128}, p + "ffn_input");
        parameter(p + "mlp/up", {24, 128}, p + "ffn_input");
        tensor("down-storage", {128, 24}, QType::Q6_G64_FP16, QuantLayout::RowSplit);
        file.root["bindings"][p + "mlp/down"] = {{"object", "down-storage"}};
        use(p + "mlp/down", p + "mlp/product");

        Json vocab    = Json::object();
        unsigned next = 256;
        for (unsigned byte = 0; byte < 256; ++byte) {
            const bool visible =
                (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) || byte >= 174;
            const unsigned cp = visible ? byte : next++;
            std::string token;
            if (cp < 128) {
                token.push_back(static_cast<char>(cp));
            } else {
                token.push_back(static_cast<char>(0xc0 | (cp >> 6)));
                token.push_back(static_cast<char>(0x80 | (cp & 63)));
            }
            vocab[token] = byte;
        }
        Json added                = Json::array();
        Json decoder              = Json::object();
        const std::array specials = {"<|endoftext|>",  "<|im_start|>",  "<|im_end|>",
                                     "<think>",        "</think>",      "<|vision_start|>",
                                     "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"};
        for (std::size_t i = 0; i < specials.size(); ++i) {
            auto token = added_token(static_cast<int>(256 + i), specials[i]);
            if (i == 3 || i == 4) token["special"] = false;
            if (i < 8) { added.push_back(token); }
            decoder[std::to_string(256 + i)] = token;
        }
        resource(
            "tokenizer.json",
            Json{
                {"normalizer", {{"type", "NFC"}}},
                {"pre_tokenizer",
                 Json::parse(
                     R"json({"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"use_regex":false}]})json")},
                {"decoder", {{"type", "ByteLevel"}}},
                {"model", {{"type", "BPE"}, {"vocab", vocab}, {"merges", Json::array()}}},
                {"added_tokens", added}}
                .dump());
        resource("tokenizer_config.json", Json{{"added_tokens_decoder", decoder}}.dump());
        resource("generation_config.json", Json{{"eos_token_id", {256, 258}}}.dump());
        resource("chat_template.jinja", "a user-provided template, preserved as data");
        file.root["files"] =
            Json::array({{{"path", nullptr}, {"payload_bytes", file.payload.size()}}});
    }
};

// A DSpine load whose codebook/tg/tsv carry `table_format`, bound with the loader's own config
// and target geometry, then materialized to the Host: the native Weight the bridge ops receive.
struct BridgeTables {
    BridgeTables(
        DspineFixture::BridgeGeometry geometry, QType table_format, QuantLayout layout,
        std::map<std::string, std::pair<QType, QuantLayout>> overrides = {})
        : fixture(geometry, table_format, layout, std::move(overrides)),
          reader(written(fixture)) {}

    // The Reader opens the file, so the fixture must be on disk before its member is built.
    static const std::filesystem::path& written(DspineFixture& source) {
        source.add_dspine_component();
        source.file.write();
        return source.file.entry;
    }

    // The Reader outlives every member below: Binder borrows it, Bindings borrows the Binder.
    template <class Body> void with_binding(Body&& body) {
        add_component();
        artifact::Binder local_binder(reader);
        Bindings local_bindings(local_binder);
        const auto text = target();
        const auto weights = dspine::bind_weights(local_bindings, *plan->config().dspine, text,
                                                   "dspine");
        body(local_bindings, weights);
    }

    dspine::TargetGeometry target() const {
        const auto& text = plan->config().text;
        return {text.hidden_size, text.vocab_size, text.num_hidden_layers};
    }

    void add_component() {
        if (plan) { return; }
        plan = q35::read_definition(reader, {.speculative = SpeculativeBackend::Dspine});
        require(plan->weights().dspine.has_value(),
                "the bridge-table artifact did not bind DSpine weights");
    }

    DspineFixture fixture;
    artifact::Reader reader;
    std::shared_ptr<const q35::ModelDefinition> plan;
};

// (a)+(c) The tables bind in the representation the artifact declares, and the Weight the ops
// get reports it: NVFP4 passes the NVFP4 operand validation, BF16 stays the plain BF16 table.
void bridge_tables_bind_as_declared(DspineFixture::BridgeGeometry geometry, QType table_format,
                                    QuantLayout layout) {
    BridgeTables tables(geometry, table_format, layout);
    tables.with_binding([&](Bindings& bindings, const dspine::Weights& w) {
        const std::array ids{w.bridge.codebook, w.bridge.tg, w.bridge.tsv};
        std::vector<artifact::TensorRequest> requests;
        for (const auto id : ids) {
            const auto& binding = bindings.at(id).reference.binding;
            require(binding.whole_object && binding.parts.size() == 1,
                    "a bridge table must be one whole tensor object");
            requests.push_back({artifact::whole_tensor(tables.reader, binding.parts[0].object),
                                Device::cpu()});
        }
        const auto materialized =
            artifact::materialize(tables.reader,
                                  artifact::plan_materialization(
                                      tables.reader,
                                      std::span<const artifact::TensorRequest>{requests}));
        for (const auto id : ids) {
            const auto name = bindings.at(id).reference.name;
            const auto view =
                artifact::bind_view(bindings.at(id).reference, materialized, Device::cpu());
            const auto weight = native_weight(view, 1.0F);
            require(weight.qtype == table_format,
                    (name + ": the bound Weight reports " +
                     std::string(artifact::format_name(weight.qtype)) + ", not " +
                     std::string(artifact::format_name(table_format))).c_str());
            if (table_format == QType::NVFP4) {
                (void)ops::common::validate_nvfp4_weight(weight, "dspine bridge table");
            }
        }
        std::cout << "bridge tables bound as " << artifact::format_name(table_format)
                  << ": codebook, tg, tsv reported\n";
    });
}

template <class Fn>
void rejects_with(Fn&& bind, std::string_view needle, const char* message) {
    try {
        bind();
    } catch (const artifact::ArtifactError& error) {
        const std::string reported = error.what();
        if (reported.find(needle) == std::string::npos) {
            throw std::runtime_error(std::string(message) + " (reported " + reported +
                                     ", expected to contain " + std::string(needle) + ")");
        }
        std::cout << "rejected: " << reported << '\n';
        return;
    }
    throw std::runtime_error(std::string(message) + " (no ArtifactError was raised)");
}

// (b) A declaration the bridge cannot execute is refused by name and format - the binding never
// falls back to BF16 for a table the artifact declared some other way.
void bridge_rejects_unsupported_declaration() {
    BridgeTables tables(DspineFixture::kSized, QType::NVFP4, QuantLayout::BlockScaleK16M128x4,
                        {{"dspine/bridge/codebook", {QType::MXFP8, QuantLayout::RowGroupScale}}});
    rejects_with([&] { tables.add_component(); },
                 "requires bf16 or nvfp4 storage, artifact declares mxfp8",
                 "an MXFP8 codebook declaration was bound");
}

// (d) ops::dspine_serial_scores indexes tg and tsv as one representation; a mixed pair is refused
// while binding, not at the first proposal.
void bridge_rejects_mixed_tg_tsv() {
    BridgeTables tables(DspineFixture::kSized, QType::NVFP4, QuantLayout::BlockScaleK16M128x4,
                        {{"dspine/bridge/tsv", {QType::BF16, QuantLayout::Contiguous}}});
    rejects_with([&] { tables.add_component(); }, "must share one storage format",
                 "a BF16 tsv with an NVFP4 tg was bound");
}

// (b) The Binder's own representation check is intact: asking for a BF16 table that the artifact
// declared NVFP4 is still the binder's rejection, so following the declaration cannot hide a
// mismatch.
void bridge_binder_check_unchanged() {
    BridgeTables tables(DspineFixture::kSized, QType::NVFP4, QuantLayout::BlockScaleK16M128x4);
    rejects_with(
        [&] {
            artifact::Binder binder(tables.reader);
            Bindings bindings(binder);
            (void)bindings.direct("dspine/bridge/tg", {DspineFixture::kSized.vocabulary,
                                                      DspineFixture::kSized.rank});
        },
        "representation does not match its mathematical value type",
        "a BF16 request for an NVFP4-declared table was bound");
}

void dspine_loads() {
    DspineFixture fixture;
    fixture.add_dspine_component();
    fixture.file.write();
    artifact::Reader reader(fixture.file.entry);
    const auto plan = q35::read_definition(reader, {.speculative = SpeculativeBackend::Dspine});
    require(plan->config().dspine.has_value(), "DSpine config was not parsed");
    require(plan->weights().dspine.has_value(), "DSpine weights were not bound");
    require(plan->config().dspine->num_hidden_layers == 2, "DSpine layer count mismatch");
    require(plan->config().dspine->bridge.code_size == 16, "DSpine code size mismatch");
    require(plan->config().dspine->bridge.rank == 8, "DSpine rank mismatch");
    require(plan->config().dspine->target_layer_ids.size() == 1, "DSpine target taps mismatch");
    require(plan->weights().dspine->layers.size() == 2, "DSpine weight layer count mismatch");
    require(plan->dspine_bytes() > 0, "DSpine byte budget was not reported");
    std::cout << "dspine load: " << plan->dspine_bytes() << " bytes reported\n";
}

void dspine_rejects_mismatch() {
    // A DFlash2 artifact must be rejected when --spec dspine is requested.
    DspineFixture fixture;
    // Add a dflash2 component instead of dspine.
    fixture.file.root["components"]["dflash2"] = {
        {"config",
         {{"architectures", {"DFlash2DraftModel"}},
          {"model_type", "qwen3"},
          {"intermediate_size", 24},
          {"num_attention_heads", 2},
          {"num_key_value_heads", 1},
          {"head_dim", 8},
          {"num_hidden_layers", 1},
          {"max_position_embeddings", 128},
          {"rms_norm_eps", 1e-6},
          {"rope_parameters", {{"rope_theta", 10000}}},
          {"layer_types", {"full_attention"}},
          {"dflash_config",
           {{"target_layer_ids", {0}},
            {"mask_token_id", 256},
            {"conv_kernel_size", 4},
            {"conv_group_size", 8},
            {"selector_rank", 8},
            {"selector_top_k", 4}}}}},
        {"target", "text"}};
    fixture.file.write();
    artifact::Reader reader(fixture.file.entry);
    rejects(
        [&] { (void)q35::read_definition(reader, {.speculative = SpeculativeBackend::Dspine}); },
        "a DFlash2 artifact was accepted as DSpine");
}

void dspine_rejects_missing_component() {
    // An artifact without a dspine component must be rejected when --spec dspine is requested.
    DspineFixture fixture;
    fixture.file.write();
    artifact::Reader reader(fixture.file.entry);
    rejects(
        [&] { (void)q35::read_definition(reader, {.speculative = SpeculativeBackend::Dspine}); },
        "an artifact without a dspine component was accepted");
}

void dflash2_path_unchanged() {
    // The DFlash2 path must still work: a dflash2 artifact loads with --spec dflash2.
    DspineFixture fixture;
    fixture.file.root["components"]["dflash2"] = {
        {"config",
         {{"architectures", {"DFlash2DraftModel"}},
          {"model_type", "qwen3"},
          {"intermediate_size", 24},
          {"num_attention_heads", 2},
          {"num_key_value_heads", 1},
          {"head_dim", 8},
          {"num_hidden_layers", 1},
          {"max_position_embeddings", 128},
          {"rms_norm_eps", 1e-6},
          {"rope_parameters", {{"rope_theta", 10000}}},
          {"layer_types", {"full_attention"}},
          {"dflash_config",
           {{"target_layer_ids", {0}},
            {"mask_token_id", 256},
            {"conv_kernel_size", 4},
            {"conv_group_size", 8},
            {"selector_rank", 8},
            {"selector_top_k", 4}}}}},
        {"target", "text"}};
    const auto h = 128;
    const auto q = 16;
    const auto k = 8;
    const auto d = 8;
    const auto inter = 24;
    const auto taps = 1;
    const std::string p = "dflash2/layers/0/";
    fixture.parameter("dflash2/feature_projection", {h, taps * h}, "dflash2/target_features");
    fixture.parameter("dflash2/context_norm", {h});
    fixture.parameter("dflash2/final_norm", {h});
    fixture.parameter(p + "input_norm", {h});
    fixture.parameter(p + "post_attention_norm", {h});
    fixture.parameter(p + "attention/query", {q, h}, p + "query_projection_input");
    fixture.parameter(p + "attention/key", {k, h}, p + "query_projection_input");
    fixture.parameter(p + "attention/value", {k, h}, p + "query_projection_input");
    fixture.parameter(p + "attention/context_key", {k, h}, "dflash2/context_input");
    fixture.parameter(p + "attention/context_value", {k, h}, "dflash2/context_input");
    fixture.parameter(p + "attention/query_norm", {d});
    fixture.parameter(p + "attention/key_norm", {d});
    fixture.parameter(p + "attention/output", {h, q}, p + "attention_output");
    fixture.parameter(p + "mlp/gate", {inter, h}, p + "mlp_input");
    fixture.parameter(p + "mlp/up", {inter, h}, p + "mlp_input");
    fixture.parameter(p + "mlp/down", {h, inter}, p + "mlp_product");
    // DFlash2 conv + selector.
    const auto conv_rows = 2 * 4 * (h / 8);
    fixture.parameter(p + "attention_conv/base_kernel", {2, 4, h});
    fixture.parameter(p + "attention_conv/kernel_projection", {conv_rows, h},
                      p + "attention_conv_input");
    fixture.parameter(p + "mlp_conv/base_kernel", {2, 4, h});
    fixture.parameter(p + "mlp_conv/kernel_projection", {conv_rows, h}, p + "mlp_conv_input");
    fixture.parameter("dflash2/candidate_selector/hidden_projection", {8, h}, "dflash2/final_hidden");
    fixture.parameter("dflash2/candidate_selector/predecessor_codebook", {272, 8});
    fixture.parameter("dflash2/candidate_selector/successor_codebook", {272, 8});
    fixture.use("text/output_head", "dflash2/final_hidden");
    fixture.file.root["files"][0]["payload_bytes"] = fixture.file.payload.size();
    fixture.file.write();
    artifact::Reader reader(fixture.file.entry);
    const auto plan = q35::read_definition(reader, {.speculative = SpeculativeBackend::DFlash2});
    require(plan->config().draft.has_value(), "DFlash2 config was not parsed");
    require(plan->weights().draft.has_value(), "DFlash2 weights were not bound");
    require(!plan->weights().dspine.has_value(), "DFlash2 load should not bind DSpine weights");
    require(plan->dspine_bytes() == 0, "DFlash2 load should report zero DSpine bytes");
}

} // namespace

int main() {
    try {
        dspine_loads();
        dspine_rejects_mismatch();
        dspine_rejects_missing_component();
        dflash2_path_unchanged();
        bridge_tables_bind_as_declared(DspineFixture::kSized, QType::NVFP4,
                                       QuantLayout::BlockScaleK16M128x4);
        bridge_tables_bind_as_declared(DspineFixture::kReference, QType::BF16,
                                       QuantLayout::Contiguous);
        bridge_rejects_unsupported_declaration();
        bridge_rejects_mixed_tg_tsv();
        bridge_binder_check_unchanged();
        std::cout << "DSpine load-side checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
