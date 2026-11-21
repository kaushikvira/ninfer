#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ninfer::artifact {
struct Component;
}

namespace ninfer::models::dspine {

struct TargetGeometry {
    std::uint32_t hidden_size, vocabulary, layers;
};

struct BridgeConfig {
    std::uint32_t code_size = 0;
    std::uint32_t rank      = 0;
    float eps               = 0;
    float kappa             = 0;
};

struct Config {
    std::uint32_t hidden_size             = 0;
    std::uint32_t intermediate_size       = 0;
    std::uint32_t num_hidden_layers       = 0;
    std::uint32_t num_attention_heads     = 0;
    std::uint32_t num_key_value_heads     = 0;
    std::uint32_t head_dim                = 0;
    std::uint32_t vocab_size              = 0;
    std::uint32_t max_position_embeddings = 0;
    float rms_norm_eps                    = 0;
    float rope_theta                      = 0;
    std::uint32_t block_size              = 0;
    std::uint32_t mask_token_id           = 0;
    std::vector<std::uint32_t> target_layer_ids;
    std::uint32_t num_target_layers       = 0;
    BridgeConfig bridge;
    std::vector<std::uint32_t> feature_layers;
    std::uint32_t head_topk               = 0;
    std::uint32_t trained_block_size      = 0;
    std::uint32_t sliding_window          = 0;
    // code_bridge_second_pass: the serial tail's own schedule. The first
    // `second_pass_first_exact` block positions are taken exact, the rest are drawn through the
    // head's top-k; `second_pass_mode` is the validated mode string the tail is built for.
    std::uint32_t second_pass_first_exact = 0;
    std::uint32_t second_pass_head_topk   = 0;
    std::string second_pass_mode;
    // selected_prefix_decode_mode: the validated bridge decode mode ("serial").
    std::string decode_mode;

    [[nodiscard]] std::uint64_t query_width() const noexcept {
        return std::uint64_t(num_attention_heads) * head_dim;
    }

    [[nodiscard]] std::uint64_t key_width() const noexcept {
        return std::uint64_t(num_key_value_heads) * head_dim;
    }

    [[nodiscard]] std::uint64_t concat_size() const noexcept {
        return std::uint64_t(target_layer_ids.size()) * hidden_size;
    }
};

[[nodiscard]] Config parse_config(const artifact::Component&, TargetGeometry);

} // namespace ninfer::models::dspine
