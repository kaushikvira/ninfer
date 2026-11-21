#include "artifact/binder.h"
#include "artifact/fixture.h"
#include "artifact/formats.h"
#include "models/common/bindings.h"
#include "models/qwen3_5/load.h"

#include <algorithm>
#include <array>
#include <iostream>

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
        const auto vocab = 272;
        const auto code_size = 16;
        const auto rank = 8;
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
        // Bridge + TG/TSV tables (BF16).
        parameter("dspine/bridge/codebook", {vocab, code_size});
        parameter("dspine/bridge/early_read_weights", {layers - 1, code_size, h});
        parameter("dspine/bridge/read_weight", {code_size, h});
        parameter("dspine/bridge/feature_write_weights", {layers, h, rank});
        parameter("dspine/bridge/gate", {rank, code_size});
        parameter("dspine/bridge/gate_bias", {rank});
        parameter("dspine/bridge/predecessor_gate", {rank, code_size});
        parameter("dspine/bridge/prev", {rank, code_size});
        parameter("dspine/bridge/tg", {vocab, rank});
        parameter("dspine/bridge/tsv", {vocab, 2 * rank});
        // Update the declared payload size to include the DSpine tensors.
        file.root["files"][0]["payload_bytes"] = file.payload.size();
    }

    DspineFixture() {
        file.payload.clear();
        file.root = {{"components",
                      {{"text",
                        {{"config",
                          {{"architectures", {"Qwen3_5ForCausalLM"}},
                           {"model_type", "qwen3_5_text"},
                           {"hidden_size", 128},
                           {"vocab_size", 272},
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
        tensor("embedding-storage", {272, 128});
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
        std::cout << "DSpine load-side checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
