#pragma once

#include "models/common/qwen/vision/parameters.h"

#include "models/qwen3_5/execution/dense_mlp.h"
#include "models/common/moe/execution.h"
#include "models/qwen3_5/model.h"
#include "ninfer/ops/weight_input.h"

#include <array>
#include <memory>
#include <limits>
#include <optional>
#include <stdexcept>
#include <variant>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

using LinearParameters = ops::SingleProjectionWeight;

[[nodiscard]] inline std::int32_t dimension(std::uint64_t value) {
    if (value > std::uint64_t(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("model dimension exceeds the Tensor integer domain");
    }
    return static_cast<std::int32_t>(value);
}

using FfnParameters = std::variant<DenseParameters, moe::Parameters>;

struct AttentionParameters {
    ops::ProjectionWeights projection;
    std::array<std::int32_t, 4> projection_rows;
    Tensor query_norm, key_norm;
    LinearParameters output;
};

struct GdnParameters {
    ops::ProjectionWeights projection;
    std::array<std::int32_t, 2> projection_rows;
    ops::ProjectionWeights control;
    Tensor a_log, dt_bias, convolution, norm;
    LinearParameters output;
};

struct BlockParameters {
    Tensor input_norm, post_attention_norm;
    std::variant<AttentionParameters, GdnParameters> mixer;
    FfnParameters ffn;
    ops::MoeExecutionHints projection_prefetch;
};

struct TextParameters {
    Weight token_embedding;
    LinearParameters output_head;
    Tensor final_norm;
    std::vector<BlockParameters> layers;
};

struct MtpProjectionParameters {
    LinearParameters packed;
    std::array<std::int32_t, 4> output_rows;
    // Dense MTP projects K/V and Q/gate independently in its incremental path.
    // MoE MTP uses a complete-parent LinearSplit.
    std::optional<std::array<LinearParameters, 4>> rows;
};

struct MtpParameters {
    LinearParameters input_projection;
    Tensor embedding_norm, hidden_norm, input_norm, post_attention_norm, final_norm;
    MtpProjectionParameters projection;
    Tensor query_norm, key_norm;
    LinearParameters output;
    FfnParameters ffn;
    LinearParameters output_head;
};

struct DynamicConvParameters {
    Tensor base_kernel;
    LinearParameters kernel_projection;
};

struct DraftBlockParameters {
    Tensor input_norm, post_attention_norm;
    LinearParameters query_key_value, context_key, context_value;
    Tensor query_norm, key_norm;
    LinearParameters output;
    DenseParameters mlp;
    std::optional<DynamicConvParameters> attention_conv, mlp_conv;
};

struct SelectorParameters {
    LinearParameters hidden_projection;
    Tensor predecessor_codebook, successor_codebook;
};

struct DraftParameters {
    LinearParameters feature_projection;
    Tensor context_norm, final_norm;
    std::vector<DraftBlockParameters> layers;
    std::optional<SelectorParameters> selector;
    LinearParameters output_head;
};

struct ProposalParameters {
    LinearParameters head;
    std::optional<Tensor> token_ids;
    std::uint32_t rows = 0;
};

// One DSpine draft layer, in the form the executor's walk runs it in. The drafter's projections
// are bound APART over one packed object (models/common/dspine/weights.h: the shipped manifests
// name `attention/query|key|value` and `mlp/gate|up` as ranges of one object, and the two context
// projections alias the K and V ranges), so the walk owns no fused copy: it issues the split
// family over `query_key_value` (the packed parent) for the block's Q/K/V AND for the context
// K/V, dropping the rows it does not need. That is the same packed-parent split the DFlash2 draft
// path runs (execution/draft.cpp:391), and the DSpine probe's, and it is the only compiled route
// to those bytes at this artifact's geometries (D-35).
struct DspineBlockParameters {
    Tensor input_norm, post_attention_norm;
    LinearParameters query_key_value;
    Tensor query_norm, key_norm;
    LinearParameters output;
    DenseParameters mlp;
};

// The code bridge, as the walk runs it: per-layer read and write slices out of the two STACKED
// bridge objects (layer `i` of [layers, code, hidden] / [layers, hidden, rank] is one row range of
// the one materialized object, never a copy), the three [rank, code] gate projections over the
// codes, the gate bias, and the codebook the anchor rows are drawn from.
struct DspineBridgeParameters {
    Weight codebook;
    std::vector<Weight> read;         // [layers]: layer 0..L-2 slice the stacked early-read object
    std::vector<Weight> write;        // [layers]: slices of the stacked feature-write object
    LinearParameters gate, predecessor_gate, prev;
    Tensor gate_bias;
};

// The DSpine streams a proposal round names by weight. The TAIL half (execution::
// dspine_propose_tail) is the final norm, the bridge's gate bias, and the bridge's two serial
// tables, tg [vocabulary, rank] and tsv [vocabulary, 2 rank]; the WALK half (execution::
// dspine_propose) adds the target-tap feature projection, the context norm, the five draft
// layers, the code bridge, and the tower's own token embedding and output head.
//
// They are the drafter's own materialized objects: the tail gathers tg/tsv/the head by candidate
// id inside its kernel and the walk reads the packed parents in place, so no copy of a table is
// taken and no second representation exists (D-31, D-35, D-37). The two objects the walk shares
// with the tower - the token embedding and the output head - are the tower's, aliased, not
// duplicated.
struct DspineParameters {
    Tensor final_norm, gate_bias;
    Weight tg, tsv;
    // The walk half: empty for an artifact that carries no drafter layers.
    LinearParameters feature_projection;
    Tensor context_norm;
    std::vector<DspineBlockParameters> layers;
    DspineBridgeParameters bridge;
    Weight token_embedding;
    // The proposal head, bound through the head's DSpine USE. `text/output_head` is one object
    // with four declared uses in every V3 manifest (text, mtp, dflash2, dspine), so it is a
    // multi-use parameter and the single-use rule at models/common/weights.cpp:17 refuses it by
    // id: the round must name which use it is multiplying. It names `dspine/final_hidden`, which
    // the manifest gives the policy `AllowA8` where the text tower's own use carries `A16Only`;
    // the definition resolves that use into `dspine::Weights::output_head_use`
    // (qwen3_5/definition.cpp:126-129) and nothing read it until now. The bytes are the tower's
    // one materialized head either way - the two ids are aliased - so this is the declared use
    // being selected, not a second weight.
    LinearParameters output_head;
};

// Cold native preparation for the fixed model implementation. This owner is stable before
// startup sizing, execution, or Graph capture; all weight addresses borrow the source ModelShard.
// Shape-dependent kernel selection and scratch remain with the calling implementation and Op.
class Parameters {
public:
    explicit Parameters(const ModelShard& source, std::shared_ptr<const ModelProfile> profile);
    Parameters(const Parameters&)            = delete;
    Parameters& operator=(const Parameters&) = delete;
    Parameters(Parameters&&)                 = delete;
    Parameters& operator=(Parameters&&)      = delete;

    const ModelShard& model;
    const std::shared_ptr<const ModelProfile> profile;
    TextParameters text;
    std::optional<MtpParameters> mtp;
    std::optional<qwen::VisionParameters> vision;
    std::optional<DraftParameters> draft;
    std::optional<DspineParameters> dspine;
    std::optional<ProposalParameters> proposal;
};

} // namespace ninfer::models::qwen3_5::execution
