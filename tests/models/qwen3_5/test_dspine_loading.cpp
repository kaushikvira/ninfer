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
    // The draft layers as one packed projection binding (`attention/query_key_value`,
    // `mlp/gate_up`) instead of the split ranges of a packed object that the V3 recipe emits.
    // The binding refuses this manifest; the case is defined, not untested.
    bool fused_draft_layers = false;
    // Mutations of the split ranges that describe no layout the drafter can read: a hole in a
    // packed object, and a context projection that does not share its K/V rows.
    bool short_packed_range = false;
    bool overlapping_range  = false;
    bool separate_context   = false;

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

    // One named range of a packed object, as the V3 manifest binds a draft layer's projections.
    void range(const std::string& name, const std::string& object, std::uint64_t begin,
               std::uint64_t end, const std::string& input) {
        if (short_packed_range && name.ends_with("/value")) { end -= 128; }
        if (overlapping_range && name.ends_with("/value")) {
            const auto width = end - begin;  // the value range becomes the key range
            begin -= width;
            end -= width;
        }
        file.root["bindings"][name] = {
            {"parts", Json::array({{{"object", object}, {"range", {begin, end}}}})}};
        use(name, input);
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
                 {
                    {"architectures", {"DSpineDraftModel"}},
                    {"model_type", "qwen3"},
                    {"hidden_size", h},
                    {"intermediate_size", inter},
                    {"num_attention_heads", heads},
                    {"num_key_value_heads", kv_heads},
                    {"head_dim", head_dim},
                    {"num_hidden_layers", layers},
                    {"max_position_embeddings", 128},
                    {"rms_norm_eps", 1e-6},
                    {"rope_parameters", { {"rope_theta", 10000}, {"rope_type", "default"} }},
                    {"layer_types", {"sliding_attention", "sliding_attention"}},
                    {"sliding_window", 64},
                    {"is_causal", true},
                    {"vocab_size", vocab},
                    {"num_target_layers", 1},
                    {"block_size", 4},
        // The member set is the converter's (tools/convert/qwen3_8_27b/dspine_recipe.py:
        // _ROOT_CONFIG + _SECOND_PASS_CONFIG + _DRAFT_CONFIG + _LAST_WRITE_REFINE) at this
        // geometry, so the allowlist's positive case carries every member the recipe emits.
        // first_exact counts the draft layers that run once (num_hidden_layers - 1) and head_topk
        // is the width the serial tail scores (kSerialTailSlots); the feature layer ids are
        // 1-based, exactly as the reference emits them.
                    {"code_bridge_second_pass", { {"first_exact", 1}, {"head_topk", 16}, {"mode", "serial_table"} }},
                    {"dflash_config",
                         {
                            {"draft_block_causal", true},
                            {"mask_token_id", 256},
                            {"selected_prefix_code_bridge",
                                { {"code_size", code_size}, {"rank", rank}, {"eps", 1e-6} } },
                            {"selected_prefix_decode_mode", "serial"},
                            {"selected_prefix_feature_layers", {1, 2}},
                            {"selected_prefix_feature_write_mode", "independent"},
                            {"selected_prefix_feature_writes", true},
                            {"selected_prefix_last_write_refine",
                                { {"head_topk", 16},
                                    {"kind", "exact_sender"},
                                    {"version", 1},
                                    {"weight", 1.0} } },
                            {"selected_prefix_terminal", false},
                            {"target_layer_ids", {0}}}},
                    },
                },
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
            if (fused_draft_layers) {
                parameter(p + "attention/query_key_value", {q + 2 * k, h},
                          p + "query_projection_input");
                parameter(p + "mlp/gate_up", {2 * inter, h}, p + "mlp_input");
            } else {
                // The V3 shape: two packed objects, named apart by ranges, with the two context
                // projections aliasing the K and V ranges. The `context_input` name is the
                // manifest's, not the loader's - it is what the recipe's Use declares.
                const std::string qkv = p + "attention/query_key_value";
                tensor(qkv, {q + 2 * k, h});
                const auto key_begin   = q * h;
                const auto key_end     = (q + k) * h;
                const auto value_begin = (q + k) * h;
                const auto value_end   = (q + 2 * k) * h;
                range(p + "attention/query", qkv, 0, key_begin, p + "query_projection_input");
                range(p + "attention/key", qkv, key_begin, key_end, p + "query_projection_input");
                range(p + "attention/value", qkv, value_begin, value_end,
                      p + "query_projection_input");
                // The context projections carry the K and V rows themselves: the manifest names
                // the same ranges under both names.
                const std::string context = [&] {
                    if (!separate_context) { return qkv; }
                    const auto id = p + "attention/context_storage";
                    tensor(id, {2 * k, h});
                    return id;
                }();
                const auto context_shift = separate_context ? key_begin : 0;
                range(p + "attention/context_key", context, key_begin - context_shift,
                      key_end - context_shift, "dspine/context_input");
                range(p + "attention/context_value", context, value_begin - context_shift,
                      value_end - context_shift, "dspine/context_input");
                parameter(p + "attention/query_norm", {head_dim});
                parameter(p + "attention/key_norm", {head_dim});
                parameter(p + "attention/output", {h, q}, p + "attention_output");
                const std::string gate_up = p + "mlp/gate_up";
                tensor(gate_up, {2 * inter, h});
                range(p + "mlp/gate", gate_up, 0, inter * h, p + "mlp_input");
                range(p + "mlp/up", gate_up, inter * h, 2 * inter * h, p + "mlp_input");
                parameter(p + "mlp/down", {h, inter}, p + "mlp_product");
                continue;
            }
            parameter(p + "attention/query_norm", {head_dim});
            parameter(p + "attention/key_norm", {head_dim});
            parameter(p + "attention/output", {h, q}, p + "attention_output");
            parameter(p + "mlp/down", {h, inter}, p + "mlp_product");
        }
        // Bridge + TG/TSV tables. codebook/tg/tsv carry the varied representation; the rest of
        // the bridge stays BF16.
        table("dspine/bridge/codebook", {vocab, code_size});
        // The six bridge MATMUL OPERANDS carry the Use the recipe declares for them, exactly as
        // all three shipped V3 manifests do (`@dspine/bridge/early_read_input`,
        // `@read_input`, `@message_input`, and `@packed_input` for the three gate/value
        // projections; each one A16Only). The loader binds them as that use - a parameter with no
        // declared use cannot reach `weight_input`, which is the rule the DSpine round now runs
        // on - so a fixture that declared none would describe an artifact no V3 drafter is.
        // `gate_bias` keeps no use: it is added by `dspine_message`, not multiplied, and the
        // tables below are gathered by row, and no manifest declares a use for either kind.
        parameter("dspine/bridge/early_read_weights", {layers - 1, code_size, h},
                  "dspine/bridge/early_read_input");
        parameter("dspine/bridge/read_weight", {code_size, h}, "dspine/bridge/read_input");
        parameter("dspine/bridge/feature_write_weights", {layers, h, rank},
                  "dspine/bridge/message_input");
        parameter("dspine/bridge/gate", {rank, code_size}, "dspine/bridge/packed_input");
        parameter("dspine/bridge/gate_bias", {rank});
        parameter("dspine/bridge/predecessor_gate", {rank, code_size},
                  "dspine/bridge/packed_input");
        parameter("dspine/bridge/prev", {rank, code_size}, "dspine/bridge/packed_input");
        table("dspine/bridge/tg", {vocab, rank});
        table("dspine/bridge/tsv", {vocab, 2 * rank});
        // Update the declared payload size to include the DSpine tensors.
        file.root["files"][0]["payload_bytes"] = file.payload.size();
    }

    explicit DspineFixture(
        BridgeGeometry bridge                                       = kReference,
        QType table_format                                          = QType::BF16,
        QuantLayout table_layout                                    = QuantLayout::Contiguous,
        std::map<std::string, std::pair<QType, QuantLayout>> table_overrides = {},
        bool fused_draft_layers = false, bool short_packed_range = false,
        bool overlapping_range = false, bool separate_context = false)
        : bridge(bridge), table_format(table_format), table_layout(table_layout),
          table_overrides(std::move(table_overrides)), fused_draft_layers(fused_draft_layers),
          short_packed_range(short_packed_range), overlapping_range(overlapping_range),
          separate_context(separate_context) {
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

// The reported footprint is the bytes of the objects the drafter reads, so a draft layer bound
// as ranges of a packed object costs the packed object once. This counts what the fixture puts
// under `dspine/` and must equal what the loader reports.
std::uint64_t packed_drafter_bytes(const artifact::Reader& reader) {
    std::uint64_t bytes = 0;
    for (const auto& object : reader.directory().objects) {
        const auto& id = std::visit(
            [](const auto& typed) -> const std::string& { return typed.id; }, object);
        if (id.starts_with("dspine/")) { bytes += artifact::object_bytes(object); }
    }
    return bytes;
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
    require(plan->dspine_bytes() == packed_drafter_bytes(reader),
            "the reported DSpine footprint is not the objects the drafter reads");
    std::cout << "dspine load: " << plan->dspine_bytes() << " bytes reported ("
              << packed_drafter_bytes(reader) << " in the packed objects)\n";
}

// The draft layer is bound apart over two packed objects, and the two context projections carry
// the K and V rows themselves. This is the layout both shipped V3 artifacts carry.
void dspine_binds_the_split_draft_layers() {
    DspineFixture fixture;
    fixture.add_dspine_component();
    fixture.file.write();
    const artifact::Reader reader(fixture.file.entry);
    const auto plan = q35::read_definition(reader, {.speculative = SpeculativeBackend::Dspine});
    const auto& w = plan->weights().dspine.value();
    const std::array names{std::string_view("query"), std::string_view("key"),
                           std::string_view("value"), std::string_view("context_key"),
                           std::string_view("context_value")};
    const std::array ids{w.layers[0].attention.query, w.layers[0].attention.key,
                         w.layers[0].attention.value, w.layers[0].attention.context_key,
                         w.layers[0].attention.context_value};
    for (std::size_t i = 0; i < ids.size(); ++i) {
        const auto& binding = plan->parameter(ids[i]).binding;
        require(!binding.whole_object && binding.parts.size() == 1,
                (std::string(names[i]) + ": a split projection is one range of a packed object")
                    .c_str());
        std::cout << "  dspine/layers/0/attention/" << names[i] << " "
                  << plan->parameter(ids[i]).name << " over the packed object at element "
                  << binding.parts[0].begin << '-' << binding.parts[0].end << '\n';
    }
    // Q, K, V, context K and context V are five names over the same object; gate and up over one.
    const auto packed_of = [&](WeightId id) {
        return plan->parameter(id).binding.parts.at(0).object;
    };
    require(packed_of(w.layers[0].attention.query) == packed_of(w.layers[0].attention.key) &&
                packed_of(w.layers[0].attention.key) == packed_of(w.layers[0].attention.value),
            "the split Q, K and V projections do not share one packed object");
    require(packed_of(w.layers[0].attention.context_key) == packed_of(w.layers[0].attention.key),
            "the context key projection is not the key projection's rows");
    require(packed_of(w.layers[0].attention.context_value) ==
                packed_of(w.layers[0].attention.value),
            "the context value projection is not the value projection's rows");
    require(packed_of(w.layers[0].mlp.gate) == packed_of(w.layers[0].mlp.up),
            "the split gate and up projections do not share one packed object");
    require(packed_of(w.layers[0].mlp.gate) != packed_of(w.layers[0].attention.query),
            "the attention and MLP packs are the same object");
    std::cout << "dspine draft layers bound apart: " << names.size()
              << " attention names over one pack, gate and up over another\n";
    require(plan->dspine_bytes() == packed_drafter_bytes(reader),
            "the split layout's reported footprint counts a packed object twice");
}

// The two defined manifest shapes, both refused by name: the fused naming this engine does not
// bind, and the split names arranged so that they describe no packed tensor.
void dspine_rejects_manifests_it_does_not_bind() {
    DspineFixture fused;
    fused.fused_draft_layers = true;
    rejects_with([&] {
        fused.add_dspine_component();
        fused.file.write();
        const artifact::Reader reader(fused.file.entry);
        (void)q35::read_definition(reader, {.speculative = SpeculativeBackend::Dspine});
    }, "a draft layer bound as one projection is not a layout this engine binds",
        "a fused-only manifest was bound");

    DspineFixture hole;
    hole.short_packed_range = true;
    rejects_with([&] {
        hole.add_dspine_component();
        hole.file.write();
        const artifact::Reader reader(hole.file.entry);
        (void)q35::read_definition(reader, {.speculative = SpeculativeBackend::Dspine});
    }, "logical shape differs from Binding coverage",
        "a split group that leaves rows of its pack unread was bound");

    DspineFixture overlap;
    overlap.overlapping_range = true;
    rejects_with([&] {
        overlap.add_dspine_component();
        overlap.file.write();
        const artifact::Reader reader(overlap.file.entry);
        (void)q35::read_definition(reader, {.speculative = SpeculativeBackend::Dspine});
    }, "a packed group covers its object in order",
        "two split projections that read the same rows of a pack were bound");

    DspineFixture detached;
    detached.separate_context = true;
    // The context names are the K and V names' own bytes, and the loader keeps them in order.
    rejects_with([&] {
        detached.add_dspine_component();
        detached.file.write();
        const artifact::Reader reader(detached.file.entry);
        (void)q35::read_definition(reader, {.speculative = SpeculativeBackend::Dspine});
    }, "must share storage with",
        "context projections that do not carry the K and V rows were bound");
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

// Applies a mutation to the fixture's dspine config and requires the parser to refuse it with a
// diagnostic containing `needle`.
template <class Mutation>
void rejects_config(Mutation&& mutation, std::string_view needle, const char* message) {
    DspineFixture fixture;
    fixture.add_dspine_component();
    mutation(fixture.file.root["components"]["dspine"]["config"]);
    rejects_with(
        [&] {
            fixture.file.write();
            const artifact::Reader reader(fixture.file.entry);
            (void)q35::read_definition(reader, {.speculative = SpeculativeBackend::Dspine});
        },
        needle, message);
}

// The positive half of the allowlist: every member the V3 recipe emits (the converter's
// _ROOT_CONFIG + _SECOND_PASS_CONFIG + _DRAFT_CONFIG + _LAST_WRITE_REFINE, which is what both
// shipped artifacts carry) parses, and the schedule the serial tail is built from reaches the
// config struct. This is the regression the real artifacts exposed: the parser's vocabulary was
// narrower than the recipe's, so neither artifact loaded.
void dspine_recipe_vocabulary_accepted() {
    DspineFixture fixture;
    fixture.add_dspine_component();
    fixture.file.write();
    const artifact::Reader reader(fixture.file.entry);
    const auto plan = q35::read_definition(reader, {.speculative = SpeculativeBackend::Dspine});
    require(plan->config().dspine.has_value(), "DSpine config was not parsed");
    const auto& config = *plan->config().dspine;
    require(config.second_pass_first_exact == 1, "second-pass first_exact was not parsed");
    require(config.second_pass_head_topk == 16, "second-pass head_topk was not parsed");
    require(config.second_pass_mode == "serial_table", "second-pass mode was not parsed");
    require(config.decode_mode == "serial", "bridge decode mode was not parsed");
    require(config.head_topk == 16, "refinement head_topk was not parsed");
    require(config.feature_layers == (std::vector<std::uint32_t>{0, 1}),
            "the 1-based feature layers were not converted to the draft's numbering");
    require(config.bridge.kappa == 0, "an absent bridge kappa was invented");
    std::cout << "dspine recipe vocabulary accepted: second pass first_exact="
              << config.second_pass_first_exact << " head_topk=" << config.second_pass_head_topk
              << " mode=" << config.second_pass_mode << " decode=" << config.decode_mode << "\n";
}

// The allowlist is the recipe's member set, not a permissive parse: an unknown member is refused
// at every level it can appear at.
void dspine_rejects_unknown_members() {
    rejects_config([](Json& config) { config["selected_prefix_terminal"] = false; },
                   "dspine config: unknown member selected_prefix_terminal",
                   "a dflash_config member hoisted to the config root was accepted");
    rejects_config([](Json& config) { config["rope_parameters"]["rope_scale"] = 1.0; },
                   "dspine RoPE: unknown member rope_scale",
                   "an unrecognised RoPE member was accepted");
    rejects_config([](Json& config) { config["code_bridge_second_pass"]["tau"] = 0.5; },
                   "DSpine code-bridge second pass: unknown member tau",
                   "an unrecognised second-pass member was accepted");
    rejects_config([](Json& config) { config["dflash_config"]["is_causal"] = true; },
                   "DSpine dflash_config: unknown member is_causal",
                   "a config-root member dropped into dflash_config was accepted");
    rejects_config(
        [](Json& config) {
            config["dflash_config"]["selected_prefix_last_write_refine"]["schedule"] = "linear";
        },
        "DSpine last-write refine: unknown member schedule",
        "an unrecognised last-write refine member was accepted");
}

// A member the allowlist admits is not accepted at any value: the pins are the shapes of the
// path this engine binds, so an unimplemented one is a rejection naming the supported set.
void dspine_rejects_unimplemented_values() {
    rejects_config(
        [](Json& config) { config["code_bridge_second_pass"]["mode"] = "parallel_table"; },
        "mode is \"parallel_table\", which this engine does not implement (supported: "
        "serial_table)",
        "a second-pass mode with no tail implementation was accepted");
    rejects_config(
        [](Json& config) { config["dflash_config"]["selected_prefix_decode_mode"] = "parallel"; },
        "selected_prefix_decode_mode is \"parallel\"",
        "a bridge decode mode with no implementation was accepted");
    rejects_config(
        [](Json& config) { config["dflash_config"]["selected_prefix_terminal"] = true; },
        "selected_prefix_terminal must be false",
        "a terminal-code artifact was accepted");
    rejects_config([](Json& config) { config["is_causal"] = false; },
                   "is_causal must be true", "a non-causal draft was accepted");
    rejects_config([](Json& config) { config["code_bridge_second_pass"]["first_exact"] = 2; },
                   "serial_table redoes the last layer only",
                   "a second pass that redoes more than the last layer was accepted");
    rejects_config([](Json& config) { config["code_bridge_second_pass"]["first_exact"] = 0; },
                   "serial_table redoes the last layer only",
                   "a second pass that redoes no layer was accepted");
    rejects_config(
        [](Json& config) {
            config["code_bridge_second_pass"]["head_topk"]                 = 8;
            config["dflash_config"]["selected_prefix_last_write_refine"]["head_topk"] = 8;
        },
        "the serial tail scores 16 candidates per slot",
        "a refinement width with no tail kernel was accepted");
    rejects_config(
        [](Json& config) {
            config["dflash_config"]["selected_prefix_last_write_refine"]["head_topk"] = 8;
        },
        "the bridge's refinement widths disagree",
        "a refine width that contradicts the second pass was accepted");
    rejects_config(
        [](Json& config) {
            config["dflash_config"]["selected_prefix_last_write_refine"]["version"] = 2;
        },
        "version must be 1", "an unknown last-write refine version was accepted");
    rejects_config(
        [](Json& config) {
            config["dflash_config"]["selected_prefix_feature_layers"] = Json::array({1});
        },
        "a bridge write on every draft layer",
        "a draft that writes features on only some layers was accepted");
    rejects_config(
        [](Json& config) {
            config["dflash_config"]["selected_prefix_feature_layers"] = Json::array({0, 1});
        },
        "feature layer index must be positive",
        "a zero-based feature layer list was accepted");
    rejects_config([](Json& config) { config["rope_parameters"]["rope_type"] = "yarn"; },
                   "rope_type is \"yarn\"", "a RoPE type with no draft path was accepted");
    rejects_config(
        [](Json& config) {
            config["dflash_config"]["selected_prefix_last_write_refine"]["kind"] = "topk_union";
        },
        "kind is \"topk_union\"", "a refine kind with no bridge path was accepted");
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
        dspine_recipe_vocabulary_accepted();
        dspine_rejects_unknown_members();
        dspine_rejects_unimplemented_values();
        dspine_binds_the_split_draft_layers();
        dspine_rejects_manifests_it_does_not_bind();
        std::cout << "DSpine load-side checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
