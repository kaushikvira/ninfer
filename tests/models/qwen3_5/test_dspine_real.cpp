// The DSpine load path on a real V3-tailored artifact: manifest -> config -> binding -> the
// footprint the drafter costs.
//
// This is the probe the config-allowlist blocker (D-28: "dspine config: unknown member
// code_bridge_second_pass") and the split-vs-fused naming blocker (D-29) needed. Both shipped
// artifacts (…_dspine-nvfp4tables.ninfer, the 1.3157 GiB drafter of D-14/D-20, and
// …_dspine.ninfer, the 2.1667 GiB BF16-table fallback) carry the converter's full recipe
// vocabulary and the converter's split draft-layer naming.
//
// Stages, in the order the load path hits them:
//   1. dspine::parse_config on the artifact's own dspine component (the allowlist).
//   2. The three bridge tables bound through Bindings::direct_as_declared, the call
//      dspine::bind_weights makes, each in the representation the manifest declares.
//   3. The dspine tensor inventory, per name.
//   4. The whole drafter, through the loader's own binder: the split names the manifest carries
//      (attention/query|key|value and mlp/gate|up as ranges of one packed projection object, the
//      context projections as the K and V ranges), then the footprint - the bytes of the objects
//      the bound weights read, one count per object, the accounting dspine_bytes() reports.
//      A manifest that binds a draft layer fused, under the packed name, is refused by name: the
//      case is defined, not unbound.
//   5. The three bridge tables as the operands the bridge ops receive: materialized one at a
//      time on the Host and run through validate_nvfp4_weight, the N%128/K%64, plane-geometry
//      and positive-finite-scale checks ops::dspine_normalize_codes and ops::dspine_serial_scores
//      apply. This is the only stage that reads payload, and it reads one table at a time.
//
// Stages (4) and (5) are the load contract, and the tripwire is NINFER_TEST_DSPINE_FULL_BIND=1:
// set, a drafter that does not bind end-to-end - or a table that fails the operand geometry -
// fails the test. Unset, the blocker is reported and the run stops, so a partial load is never
// mistaken for a loadable drafter.
//
// NINFER_TEST_ARTIFACT names the artifact. NINFER_TEST_DSPINE_TABLE_FORMAT pins the
// representation the three tables must declare ("nvfp4" for the sized artifact, "bf16" for the
// fallback) - the only difference between them.

#include "artifact/binder.h"
#include "artifact/encoded_slice.h"
#include "artifact/fixture.h"
#include "artifact/formats.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "artifact/schema.h"
#include "artifact/views.h"
#include "core/weight_view.h"
#include "models/common/bindings.h"
#include "models/common/dspine/config.h"
#include "models/common/dspine/weights.h"
#include "ops/common/host/operands.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::models;
namespace dspine = ninfer::models::dspine;
using ninfer::test::artifact_fixture::require;

std::string setting(std::string_view name, std::string_view fallback = {}) {
    const char* value = std::getenv(std::string(name).c_str());
    return value && *value ? std::string(value) : std::string(fallback);
}

std::uint32_t dimension(const artifact::Json& config, const char* name) {
    return static_cast<std::uint32_t>(artifact::require_u64(config.at(name), name));
}

// The three bridge tables are BF16 in the fallback artifact and NVFP4 block-scaled in the sized
// one; this is dspine/weights.cpp's accepted set, repeated so the probe follows the loader.
constexpr std::array kTableFormats{QType::BF16, QType::NVFP4};

constexpr double kMiB = 1024.0 * 1024.0;
constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

// Which part of the drafter one packed object belongs to, for the footprint report.
std::string section_of(const artifact::Directory& directory, artifact::ObjectHandle object) {
    const auto& id = directory.tensor(object).id;
    const auto holds = [&](std::string_view needle) { return id.find(needle) != std::string::npos; };
    if (holds("/bridge/codebook") || holds("/bridge/tg") || holds("/bridge/tsv")) {
        return "bridge gather tables (codebook, tg, tsv)";
    }
    if (holds("/bridge/")) { return "bridge gates and reads"; }
    if (holds("/layers/")) {
        return "draft layers (5 blocks of packed projections)";
    }
    return "feature projection and norms";
}

void probe() {
    const artifact::Reader reader(setting("NINFER_TEST_ARTIFACT"));
    const auto& directory = reader.directory();

    // (1) The config allowlist, on the artifact's own component.
    const auto& text = directory.component("text").config;
    const dspine::TargetGeometry target{dimension(text, "hidden_size"),
                                        dimension(text, "vocab_size"),
                                        dimension(text, "num_hidden_layers")};
    const auto config = dspine::parse_config(directory.component("dspine"), target);
    std::cout << "dspine config parsed: hidden=" << config.hidden_size
              << " layers=" << config.num_hidden_layers << " vocab=" << config.vocab_size
              << " block_size=" << config.block_size << " taps=" << config.target_layer_ids.size()
              << " feature_layers=" << config.feature_layers.size()
              << " bridge code_size=" << config.bridge.code_size << " rank=" << config.bridge.rank
              << " kappa=" << config.bridge.kappa << " eps=" << config.bridge.eps << '\n';
    std::cout << "dspine second pass: first_exact=" << config.second_pass_first_exact
              << " head_topk=" << config.second_pass_head_topk
              << " mode=" << config.second_pass_mode << " decode=" << config.decode_mode
              << " refine head_topk=" << config.head_topk << '\n';
    require(config.second_pass_mode == "serial_table", "the recipe's second-pass mode changed");
    require(config.decode_mode == "serial", "the recipe's bridge decode mode changed");
    require(config.second_pass_first_exact + 1 == config.num_hidden_layers,
            "serial_table does not redo the last draft layer only");
    require(config.head_topk == 16, "the recipe's refinement width changed");
    require(config.bridge.kappa > 0, "the recipe's bridge kappa was not parsed");
    require(config.feature_layers.size() == config.num_hidden_layers,
            "the feature layers were not converted out of the artifact's 1-based numbering");

    artifact::Binder binder(reader);

    // (2) The tables bind in the representation the manifest declares. The probe binds them once
    // here, on its own Bindings, and once again through bind_weights below: the loader's own
    // binder refuses a parameter bound twice, so the two bindings are kept apart.
    Bindings tables_bindings(binder);
    const auto v = target.vocabulary;
    const auto c = config.bridge.code_size;
    const auto r = config.bridge.rank;
    const std::vector tables{std::pair{std::string("dspine/bridge/codebook"),
                                       artifact::Shape{v, c}},
                              std::pair{std::string("dspine/bridge/tg"), artifact::Shape{v, r}},
                              std::pair{std::string("dspine/bridge/tsv"),
                                        artifact::Shape{v, 2 * r}}};
    std::string declared;
    for (const auto& [name, shape] : tables) {
        (void)tables_bindings.direct_as_declared(name, shape, kTableFormats);
        const auto format = artifact::format_name(binder.declared_format(name));
        std::cout << "table " << name << " bound as " << format << '\n';
        if (declared.empty()) {
            declared = format;
        } else {
            require(declared == format, "the three bridge tables did not declare one format");
        }
    }
    const auto expected = setting("NINFER_TEST_DSPINE_TABLE_FORMAT");
    if (!expected.empty()) {
        require(artifact::format_name(artifact::parse_format(expected)) == declared,
                ((("the bridge tables are bound as " + declared + ", the caller expected ")
                   + std::string(artifact::format_name(artifact::parse_format(expected))))).c_str());
    }

    // (3) The inventory the loader worked from, per name. The bytes here are per name too, and
    // the split draft-layer names share two packed objects per block, so this total is larger
    // than what the drafter costs: (4b) is the footprint.
    std::uint64_t named_bytes = 0;
    std::size_t names        = 0;
    for (const auto& [name, binding] : directory.bindings) {
        if (!name.starts_with("dspine/")) { continue; }
        ++names;
        const auto& tensor = directory.tensor(binding.parts.at(0).object);
        named_bytes += tensor.bytes;
        if (names <= 3 || name.starts_with("dspine/bridge/")) {
            std::cout << "  " << name << ' ' << tensor.format << ' ' << tensor.bytes << "B ";
            for (const auto dim : tensor.shape) { std::cout << 'x' << dim; }
            std::cout << '\n';
        }
    }
    std::cout << "dspine inventory: " << names << " bindings, " << named_bytes << " bytes ("
              << (named_bytes / kGiB) << " GiB) named\n";

    // (4) The whole drafter, through the loader's own binder.
    Bindings bindings(binder);
    dspine::Weights weights;
    try {
        weights = dspine::bind_weights(bindings, config, target, "dspine");
        std::cout << "dspine bind_weights: " << weights.layers.size()
                  << " draft layers and the bridge bound\n";
        // The mapping, as the loader's binding records it: the rows each split name contributes
        // to the packed projection object of draft layer 0.
        const auto rows = [&](WeightId id) {
            const auto& part = bindings.at(id).reference.binding.parts.at(0);
            return std::to_string(part.begin / target.hidden_size) + '-' +
                   std::to_string(part.end / target.hidden_size);
        };
        const auto packed = [&](WeightId id) {
            return directory.tensor(bindings.at(id).reference.binding.parts.at(0).object).id;
        };
        const auto& a = weights.layers.front().attention;
        std::cout << "  attention/query      rows " << rows(a.query) << " of " << packed(a.query)
                  << '\n'
                  << "  attention/key        rows " << rows(a.key) << ", attention/value rows "
                  << rows(a.value) << " of the same object\n"
                  << "  attention/context_key rows " << rows(a.context_key)
                  << ", attention/context_value rows " << rows(a.context_value)
                  << " (the K and V rows themselves)\n";
        const auto& m = weights.layers.front().mlp;
        std::cout << "  mlp/gate             rows " << rows(m.gate) << ", mlp/up rows "
                  << rows(m.up) << " of " << packed(m.gate) << '\n';
        require(packed(a.query) == packed(a.key) && packed(a.key) == packed(a.value) &&
                    packed(a.query) != packed(m.gate),
                "the split projections of a draft layer do not share one packed object");
        const auto same_storage = [](const artifact::Binding& one, const artifact::Binding& two) {
            return std::equal(one.parts.begin(), one.parts.end(), two.parts.begin(),
                              two.parts.end(), [](const auto& a, const auto& b) {
                                  return a.object == b.object && a.begin == b.begin &&
                                         a.end == b.end;
                              });
        };
        require(same_storage(bindings.at(a.context_key).reference.binding,
                              bindings.at(a.key).reference.binding) &&
                    same_storage(bindings.at(a.context_value).reference.binding,
                                 bindings.at(a.value).reference.binding),
                "the context projections are not the K and V projections' bytes");
    } catch (const artifact::ArtifactError& error) {
        std::cout << "BLOCKED dspine::bind_weights: " << error.what() << '\n';
        if (setting("NINFER_TEST_DSPINE_FULL_BIND") != "1") {
            std::cout << "(a drafter that does not bind end-to-end is not loadable; set "
                         "NINFER_TEST_DSPINE_FULL_BIND=1 to fail here)\n";
            return;
        }
        throw;
    }

    // (4b) The footprint, one count per object: the bytes the drafter costs, which is what the
    // D-14 sizing is priced against.
    std::set<std::size_t> counted;
    std::uint64_t footprint = 0;
    std::map<std::string, std::uint64_t> sections;
    const auto count_weight = [&](WeightId id) {
        for (const auto& part : bindings.at(id).reference.binding.parts) {
            if (!counted.insert(part.object.index).second) { continue; }
            const auto size = artifact::object_bytes(directory.object(part.object));
            footprint += size;
            sections[section_of(directory, part.object)] += size;
        }
    };
    for (const auto& layer : weights.layers) {
        count_weight(layer.input_norm);
        count_weight(layer.post_attention_norm);
        count_weight(layer.attention.query);
        count_weight(layer.attention.key);
        count_weight(layer.attention.value);
        count_weight(layer.attention.context_key);
        count_weight(layer.attention.context_value);
        count_weight(layer.attention.query_norm);
        count_weight(layer.attention.key_norm);
        count_weight(layer.attention.output);
        count_weight(layer.mlp.gate);
        count_weight(layer.mlp.up);
        count_weight(layer.mlp.down);
    }
    count_weight(weights.feature_projection);
    count_weight(weights.context_norm);
    count_weight(weights.final_norm);
    for (const auto& table :
         std::array{weights.bridge.codebook, weights.bridge.tg, weights.bridge.tsv,
                    weights.bridge.early_read_weights, weights.bridge.read_weight,
                    weights.bridge.feature_write_weights, weights.bridge.gate,
                    weights.bridge.gate_bias, weights.bridge.predecessor_gate,
                    weights.bridge.prev}) {
        count_weight(table);
    }
    for (const auto& [section, size] : sections) {
        std::cout << "  footprint " << section << ": " << size << "B (" << (size / kMiB)
                  << " MiB)\n";
    }
    std::cout << "dspine footprint: " << counted.size() << " objects, " << footprint << " bytes ("
              << (footprint / kGiB) << " GiB); the " << names << " names total " << named_bytes
              << " bytes (" << (named_bytes / kGiB)
              << " GiB), which reads the packed objects of a draft layer several times over\n"
              << "D-14 drafter sizing: 1.35 GiB, the artifact costs " << (footprint / kGiB)
              << " GiB (" << ((footprint / kGiB) - 1.35) / 1.35 * 100.0 << " %)\n";

    // (5) The bridge tables as the operands the bridge ops receive. One table is materialized at
    // a time on the Host - the only payload this probe reads.
    if (setting("NINFER_TEST_DSPINE_FULL_BIND") == "1") {
        const std::vector check{std::tuple{std::string("dspine/bridge/codebook"),
                                           weights.bridge.codebook, std::uint64_t(c)},
                                std::tuple{std::string("dspine/bridge/tg"), weights.bridge.tg,
                                           std::uint64_t(r)},
                                std::tuple{std::string("dspine/bridge/tsv"), weights.bridge.tsv,
                                           std::uint64_t(2 * r)}};
        for (const auto& [name, id, width] : check) {
            const auto& reference = bindings.at(id).reference;
            const auto request    = artifact::TensorRequest{
                artifact::whole_tensor(reader, reference.binding.parts.at(0).object),
                Device::cpu()};
            const auto materialized = artifact::materialize(
                reader, artifact::plan_materialization(reader, std::span{&request, 1}));
            const auto view   = artifact::bind_view(reference, materialized, Device::cpu());
            const auto weight = native_weight(view, 1.0F);
            require(weight.n == std::int64_t(v) && weight.k == std::int64_t(width),
                    "a real bridge table is not [vocabulary, width] as bound");
            if (weight.qtype != QType::NVFP4) {
                std::cout << "  " << name << ": a " << declared << " table " << weight.n << 'x'
                          << weight.k << ", the NVFP4 operand geometry does not apply\n";
                continue;
            }
            (void)ops::common::validate_nvfp4_weight(weight, name.c_str());
            std::cout << "  " << name << ": NVFP4 operand validated, " << weight.n << 'x'
                      << weight.k << " N%128=" << weight.n % 128 << " K%64=" << weight.k % 64
                      << " matrix scale " << weight.weight_scale << " input scale "
                      << weight.input_scale << '\n';
        }
    }
}

} // namespace

int main() {
    if (setting("NINFER_TEST_ARTIFACT").empty()) {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    try {
        probe();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
