// The DSpine load path on a real V3-tailored artifact: manifest -> config -> bridge binding.
//
// This is the probe the config-allowlist blocker (D-28: "dspine config: unknown member
// code_bridge_second_pass") needed. Both shipped artifacts
// (…_dspine-nvfp4tables.ninfer, the 1.3157 GiB drafter of D-14/D-20, and …_dspine.ninfer, the
// 2.1667 GiB BF16-table fallback) carry the converter's full recipe vocabulary, and the parser's
// allowlist was narrower than it. Only the manifest is read - no tensor is materialized - so the
// ~25 GiB payload never becomes resident.
//
// Stages, in the order the load path hits them:
//   1. dspine::parse_config on the artifact's own dspine component (the allowlist).
//   2. The three bridge tables bound through Bindings::direct_as_declared, the call
//      dspine::bind_weights makes, each in the representation the manifest declares.
//   3. The dspine tensor inventory (declared format, shape, bytes per binding).
//   4. The whole-drafter dspine::bind_weights. `⚠` This is the NEXT blocker and it is open: the
//      V3 draft layers are bound apart (`attention/query|key|value`,
//      `attention/context_key|context_value`, `mlp/gate|up`) while bind_weights asks for the fused
//      `attention/query_key_value` and `mlp/gate_up`. By default the blocker is reported; with
//      NINFER_TEST_DSPINE_FULL_BIND=1 it fails the test, so the day the naming/fusion lands the
//      run goes green and the tables inventory is checked through the real loader.
//
// NINFER_TEST_ARTIFACT names the artifact. NINFER_TEST_DSPINE_TABLE_FORMAT pins the
// representation the three tables must declare ("nvfp4" for the sized artifact, "bf16" for the
// fallback) - the only difference between them.

#include "artifact/binder.h"
#include "artifact/fixture.h"
#include "artifact/formats.h"
#include "artifact/reader.h"
#include "artifact/schema.h"
#include "models/common/bindings.h"
#include "models/common/dspine/config.h"
#include "models/common/dspine/weights.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
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
    Bindings bindings(binder);

    // (2) The tables bind in the representation the manifest declares.
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
        (void)bindings.direct_as_declared(name, shape, kTableFormats);
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

    // (3) The inventory the loader worked from.
    std::uint64_t bytes = 0;
    std::size_t count  = 0;
    for (const auto& [name, binding] : directory.bindings) {
        if (!name.starts_with("dspine/")) { continue; }
        ++count;
        const auto& tensor = directory.tensor(binding.parts.at(0).object);
        bytes += tensor.bytes;
        if (count <= 3 || name.starts_with("dspine/bridge/")) {
            std::cout << "  " << name << ' ' << tensor.format << ' ' << tensor.bytes << "B ";
            for (const auto dim : tensor.shape) { std::cout << 'x' << dim; }
            std::cout << '\n';
        }
    }
    std::cout << "dspine inventory: " << count << " bindings, " << bytes << " bytes ("
              << (double(bytes) / (1024.0 * 1024.0 * 1024.0)) << " GiB) declared\n";

    // (4) The whole drafter, through the loader's own binder.
    try {
        const auto weights = dspine::bind_weights(bindings, config, target, "dspine");
        std::cout << "dspine bind_weights: " << weights.layers.size()
                  << " draft layers and the bridge bound\n";
    } catch (const artifact::ArtifactError& error) {
        std::cout << "BLOCKED dspine::bind_weights: " << error.what() << '\n';
        if (setting("NINFER_TEST_DSPINE_FULL_BIND") != "1") {
            std::cout << "(set NINFER_TEST_DSPINE_FULL_BIND=1 to fail on the open naming/fusion "
                         "blocker)\n";
            return;
        }
        throw;
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
