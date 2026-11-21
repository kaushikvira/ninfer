#pragma once

#include "models/common/weights.h"
#include "models/common/dspine/config.h"

namespace ninfer::models {
class Bindings;
}

namespace ninfer::models::dspine {

struct AttentionWeights {
    WeightId query_key_value;
    WeightId query_norm, key_norm, output;
};

struct BlockWeights {
    WeightId input_norm, post_attention_norm;
    AttentionWeights attention;
    WeightId gate_up, down;
};

struct BridgeWeights {
    WeightId codebook;
    WeightId early_read_weights, read_weight;
    WeightId feature_write_weights;
    WeightId gate, gate_bias, predecessor_gate, prev;
    WeightId tg, tsv;
};

struct Weights {
    WeightId feature_projection, context_norm, final_norm;
    std::vector<BlockWeights> layers;
    BridgeWeights bridge;
    WeightId token_embedding, output_head;
    WeightUseId output_head_use;
};

[[nodiscard]] Weights bind_weights(Bindings&, const Config&, TargetGeometry,
                                   const std::string& component);

} // namespace ninfer::models::dspine
