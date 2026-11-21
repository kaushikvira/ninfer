#pragma once

#include "models/common/weights.h"
#include "models/common/dspine/config.h"

namespace ninfer::models {
class Bindings;
}

namespace ninfer::models::dspine {

// A DSpine draft layer is bound apart. The shipped artifacts carry one packed projection object
// per block and name its ranges - `attention/query|key|value`, the two context ranges that alias
// the K and V ranges, and `mlp/gate|up` - so the drafter runs the projections apart over the one
// materialized object. No fused copy of a draft weight exists in memory or in this struct.
struct AttentionWeights {
    WeightId query, key, value, context_key, context_value;
    WeightId query_norm, key_norm, output;
};

struct BlockWeights {
    WeightId input_norm, post_attention_norm;
    AttentionWeights attention;
    DenseWeights mlp;
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
