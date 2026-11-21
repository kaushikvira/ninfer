#include "models/common/dspine/dspine_serial_tail.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ninfer::models::common::dspine {
namespace {

// Reference filter_probs ranks by descending probability with the lower slot index winning
// ties (propose.py:77-80). Ranking the softmax output would collapse near-equal probabilities
// into an order the reference never sees, so rank on the distribution exactly as it does.
bool ahead(const float* p, std::int32_t j, std::int32_t k) {
    return p[j] > p[k] || (p[j] == p[k] && j < k);
}

void require_slots(std::int32_t slots) {
    if (slots < 1 || slots > kSerialTailSlots) {
        throw std::invalid_argument("dspine serial tail: slot count is outside [1,16]");
    }
}

void require_geometry(std::int32_t slots, std::int32_t drafts, std::int32_t batch) {
    require_slots(slots);
    if (drafts < kSerialTailMinDrafts || drafts > kSerialTailMaxDrafts) {
        throw std::invalid_argument("dspine serial tail: draft count is outside [1,15]");
    }
    if (batch < 1) { throw std::invalid_argument("dspine serial tail: batch must be positive"); }
}

// One (slot, sender) row: filter the softmax, then either take the argmax (greedy) or draw.
// Returns the chosen candidate slot.
std::int32_t resolve_row(const float* scores, std::int32_t slots, const SerialTailPolicy& policy,
                         std::uint64_t seed, std::int32_t position, std::uint32_t sub,
                         float* q, bool* keep) {
    serial_tail_filter(scores, slots, policy, q, keep);
    if (policy.greedy()) {
        return static_cast<std::int32_t>(std::max_element(keep, keep + slots) - keep);
    }
    return serial_tail_draw(q, keep, slots,
                            serial_tail_uniform(seed, position, kSerialTailPurpose, sub));
}

} // namespace

SerialTailPolicy serial_tail_policy(float temperature, std::int32_t top_k, float top_p,
                                    float min_p, bool has_penalties, bool has_token_mask) {
    if (has_penalties) {
        throw std::invalid_argument(
            "dspine serial tail: presence/frequency penalties cannot be expressed by the "
            "reference tail; refusing to sample a proposal the acceptor would verify under a "
            "different distribution");
    }
    if (has_token_mask) {
        throw std::invalid_argument(
            "dspine serial tail: a token mask cannot be expressed by the reference tail; "
            "constrained decoding over a DSpine proposal is unsupported");
    }
    if (std::isnan(temperature)) {
        throw std::invalid_argument("dspine serial tail: temperature is NaN");
    }
    if (!(temperature <= 0.0f) && !std::isfinite(temperature)) {
        throw std::invalid_argument("dspine serial tail: temperature is not finite");
    }
    if (top_k < 1 || top_k > 20) {
        throw std::invalid_argument("dspine serial tail: top_k is outside [1,20]");
    }
    if (std::isnan(top_p) || top_p > 1.0f) {
        throw std::invalid_argument("dspine serial tail: top_p must be in [0,1]");
    }
    if (std::isnan(min_p) || min_p < 0.0f || min_p >= 1.0f) {
        throw std::invalid_argument("dspine serial tail: min_p must be in [0,1)");
    }
    SerialTailPolicy policy;
    policy.temperature = temperature;
    policy.top_k       = top_k;
    policy.top_p       = top_p;
    policy.min_p       = min_p;
    return policy;
}

void serial_tail_filter(const float* scores, std::int32_t slots, const SerialTailPolicy& policy,
                        float* q, bool* keep) {
    if (scores == nullptr || q == nullptr || keep == nullptr) {
        throw std::invalid_argument("dspine serial tail: null filter operand");
    }
    require_slots(slots);
    if (policy.greedy()) {
        // Temperature <= 0 is the greedy degenerate: the reference tail is never invoked (it
        // takes inv_temp), so the documented behaviour is the argmax chain. It is expressed
        // through the same filter surface as a one-hot distribution so callers have one path,
        // and so the sparse acceptor's "positive mass at the chosen id" contract holds.
        std::int32_t best = 0;
        for (std::int32_t k = 1; k < slots; ++k) {
            if (scores[k] > scores[best]) best = k;
        }
        for (std::int32_t k = 0; k < slots; ++k) {
            keep[k] = k == best;
            q[k]    = keep[k] ? 1.0f : 0.0f;
        }
        return;
    }
    // softmax(row * inv_temp) with the max subtraction a softmax performs.
    const float inv_temp = 1.0f / policy.temperature;
    float peak           = scores[0] * inv_temp;
    for (std::int32_t k = 1; k < slots; ++k) {
        peak = std::max(peak, scores[k] * inv_temp);
    }
    float p[kSerialTailSlots];
    float total = 0.0f;
    for (std::int32_t k = 0; k < slots; ++k) {
        p[k] = std::exp(scores[k] * inv_temp - peak);
        total += p[k];
    }
    if (!(total > 0.0f) || !std::isfinite(total)) {
        throw std::invalid_argument("dspine serial tail: non-finite proposal distribution");
    }
    for (std::int32_t k = 0; k < slots; ++k) p[k] /= total;

    float maxima = p[0];
    for (std::int32_t k = 1; k < slots; ++k) maxima = std::max(maxima, p[k]);
    float kept_mass = 0.0f;
    for (std::int32_t k = 0; k < slots; ++k) {
        std::int32_t rank   = 0;
        float mass_before   = 0.0f;
        for (std::int32_t j = 0; j < slots; ++j) {
            if (!ahead(p, j, k)) continue;
            ++rank;
            mass_before += p[j];
        }
        bool keep_slot = rank < policy.top_k && mass_before <= policy.top_p;
        if (policy.min_p > 0.0f) keep_slot = keep_slot && p[k] >= policy.min_p * maxima;
        // The reference relies on the top candidate always surviving (rank 0, mass 0), so the
        // support can never be empty; the check below proves that invariant still holds.
        keep[k] = keep_slot;
        q[k]    = keep_slot ? p[k] : 0.0f;
        kept_mass += q[k];
    }
    if (!(kept_mass > 0.0f) || !std::isfinite(kept_mass)) {
        throw std::invalid_argument("dspine serial tail: filtered proposal has empty support");
    }
    for (std::int32_t k = 0; k < slots; ++k) q[k] /= kept_mass;
}

std::int32_t serial_tail_draw(const float* q, const bool* keep, std::int32_t slots, float u) {
    if (q == nullptr || keep == nullptr) {
        throw std::invalid_argument("dspine serial tail: null draw operand");
    }
    require_slots(slots);
    if (!(u >= 0.0f) || u >= 1.0f) {
        throw std::invalid_argument("dspine serial tail: draw uniform is outside [0,1)");
    }
    // cdf[-1] is the kept mass (1 after renormalisation); keeping the multiply mirrors
    // propose.py:149 including its fp32 rounding.
    float cdf[kSerialTailSlots];
    float running = 0.0f;
    for (std::int32_t k = 0; k < slots; ++k) {
        running += q[k];
        cdf[k] = running;
    }
    const float threshold = u * cdf[slots - 1];
    std::int32_t choice   = 0;
    for (std::int32_t k = 0; k < slots; ++k) {
        if (cdf[k] <= threshold) ++choice;
    }
    std::int32_t last = 0;
    for (std::int32_t k = 0; k < slots; ++k) {
        if (keep[k]) last = k;
    }
    return std::min(choice, last);
}

float serial_tail_uniform(std::uint64_t seed, std::int32_t position, std::int32_t purpose,
                          std::uint32_t sub) {
    // Mirrors ops/sampling/kernels/random.cuh:5-28 (splitmix64 mixing, 24-bit mantissa).
    auto splitmix = [](std::uint64_t x) {
        x += 0x9E3779B97F4A7C15ull;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
        return x ^ (x >> 31);
    };
    std::uint64_t key = seed;
    key = splitmix(key ^ (static_cast<std::uint64_t>(static_cast<std::uint32_t>(position)) *
                          0xD1B54A32D192ED03ull));
    key = splitmix(key ^ (static_cast<std::uint64_t>(static_cast<std::uint32_t>(purpose)) << 21) ^
                   (static_cast<std::uint64_t>(sub) * 0x2545F4914F6CDD1Dull));
    const std::uint32_t bits = static_cast<std::uint32_t>(key >> 40);
    return static_cast<float>(bits) * (1.0f / 16777216.0f);
}

void serial_tail_table_from_logits(const float* logits, std::int32_t vocab, std::int32_t width,
                                   const std::int32_t* cand, std::int32_t slots,
                                   std::int32_t drafts, std::int32_t batch, float* table) {
    if (logits == nullptr || cand == nullptr || table == nullptr) {
        throw std::invalid_argument("dspine serial tail: null logits operand");
    }
    if (vocab < 1 || width != drafts + 1) {
        throw std::invalid_argument("dspine serial tail: logits geometry is invalid");
    }
    require_geometry(slots, drafts, batch);
    // A plain per-slot proposal has no sender dependence: a candidate's score is that slot's
    // own logit, so the chain only narrows the support to the slot's candidate set.
    for (std::int32_t b = 0; b < batch; ++b) {
        for (std::int32_t i = 0; i < drafts; ++i) {
            for (std::int32_t k = 0; k < slots; ++k) {
                const std::int32_t id =
                    cand[(static_cast<std::int64_t>(k) * drafts + i) * batch + b];
                if (id < 0 || id >= vocab) {
                    throw std::invalid_argument(
                        "dspine serial tail: candidate id is outside the vocabulary");
                }
                const float value =
                    logits[(static_cast<std::int64_t>(id) * width + 1 + i) * batch + b];
                for (std::int32_t sender = 0; sender < slots; ++sender) {
                    table[((static_cast<std::int64_t>(i) * slots + sender) * slots + k) * batch +
                          b] = value;
                }
            }
        }
    }
}

void serial_tail_run(const float* table, const std::int32_t* cand, std::int32_t slots,
                     std::int32_t drafts, std::int32_t batch, const SerialTailPolicy* policies,
                     const std::uint64_t* seeds, const std::int32_t* positions,
                     std::int32_t* drafts_out, float* proposal_q, std::int32_t* idx_out) {
    if (table == nullptr || cand == nullptr || policies == nullptr || seeds == nullptr ||
        positions == nullptr || drafts_out == nullptr || proposal_q == nullptr ||
        idx_out == nullptr) {
        throw std::invalid_argument("dspine serial tail: null chain operand");
    }
    require_geometry(slots, drafts, batch);
    for (std::int32_t b = 0; b < batch; ++b) {
        std::int32_t sender = 0;
        for (std::int32_t i = 0; i < drafts; ++i) {
            float q[kSerialTailSlots];
            bool keep[kSerialTailSlots];
            std::int32_t chosen = 0;
            for (std::int32_t s = 0; s < slots; ++s) {
                float scores[kSerialTailSlots];
                // table entry (slot i, sender s, candidate k) is at ((i*slots+s)*slots+k)*batch+b.
                const std::int64_t base =
                    (static_cast<std::int64_t>(i) * slots + s) * slots * batch + b;
                for (std::int32_t k = 0; k < slots; ++k) {
                    scores[k] = table[base + static_cast<std::int64_t>(k) * batch];
                }
                const std::int32_t index = resolve_row(
                    scores, slots, policies[b], seeds[b], positions[b],
                    static_cast<std::uint32_t>(i), q, keep);
                idx_out[(static_cast<std::int64_t>(s) * drafts + i) * batch + b] = index;
                // Only the sender this walk actually reaches picks the draft id, and only its
                // distribution is the one that generated that id - so only its row is the q the
                // acceptor must verify against (the other senders' rows are never reached).
                if (s == sender) {
                    chosen = index;
                    for (std::int32_t k = 0; k < slots; ++k) {
                        proposal_q[(static_cast<std::int64_t>(k) * drafts + i) * batch + b] = q[k];
                    }
                }
            }
            const std::int32_t id =
                cand[(static_cast<std::int64_t>(chosen) * drafts + i) * batch + b];
            if (id < 0) {
                throw std::invalid_argument("dspine serial tail: negative candidate id");
            }
            drafts_out[static_cast<std::int64_t>(i) * batch + b] = id;
            sender                                              = chosen;
        }
    }
}

const char* serial_tail_accept_geometry_error(std::int32_t slots, std::int32_t drafts,
                                              std::int32_t batch) {
    if (slots != kSerialTailSlots) {
        return "dspine serial tail: the sparse acceptor needs exactly 16 candidate ids per row";
    }
    if (drafts < kSerialTailMinDrafts || drafts > kSerialTailMaxDrafts) {
        return "dspine serial tail: draft count is outside [1,15]";
    }
    if (batch < 1) { return "dspine serial tail: batch must be positive"; }
    return nullptr;
}

const char* serial_tail_accept_q_error(const float* proposal_q, const std::int32_t* cand,
                                       const std::int32_t* drafts_out, std::int32_t slots,
                                       std::int32_t drafts, std::int32_t batch, float tolerance) {
    if (proposal_q == nullptr || cand == nullptr || drafts_out == nullptr) {
        return "dspine serial tail: null accept operand";
    }
    if (const char* reason = serial_tail_accept_geometry_error(slots, drafts, batch)) {
        return reason;
    }
    for (std::int32_t b = 0; b < batch; ++b) {
        for (std::int32_t i = 0; i < drafts; ++i) {
            float sum = 0.0f;
            for (std::int32_t k = 0; k < slots; ++k) {
                const float value =
                    proposal_q[(static_cast<std::int64_t>(k) * drafts + i) * batch + b];
                if (!std::isfinite(value) || value < 0.0f) {
                    return "dspine serial tail: proposal q is not a non-negative finite mass";
                }
                sum += value;
            }
            if (std::fabs(sum - 1.0f) > tolerance) {
                return "dspine serial tail: proposal q does not normalise to one";
            }
            // speculative_accept_sparse_drafts requires each live candidate list to hold
            // distinct ids; the walked draft id then resolves to exactly one candidate slot.
            const std::int32_t draft = drafts_out[static_cast<std::int64_t>(i) * batch + b];
            std::int32_t chosen      = -1;
            for (std::int32_t k = 0; k < slots; ++k) {
                const std::int32_t id =
                    cand[(static_cast<std::int64_t>(k) * drafts + i) * batch + b];
                for (std::int32_t other = k + 1; other < slots; ++other) {
                    if (cand[(static_cast<std::int64_t>(other) * drafts + i) * batch + b] == id) {
                        return "dspine serial tail: candidate ids repeat inside a slot";
                    }
                }
                if (id == draft) chosen = k;
            }
            if (chosen < 0) {
                return "dspine serial tail: the walked draft id is not one of its candidates";
            }
            if (!(proposal_q[(static_cast<std::int64_t>(chosen) * drafts + i) * batch + b] >
                  0.0f)) {
                return "dspine serial tail: chosen candidate carries no proposal mass";
            }
        }
    }
    return nullptr;
}

} // namespace ninfer::models::common::dspine
