#pragma once

#include "ninfer/types.h"

#include <string_view>

namespace ninfer::models {

struct LoadOptions {
    EnginePurpose purpose          = EnginePurpose::Generation;
    bool vision                    = false;
    SpeculativeBackend speculative = SpeculativeBackend::None;
    ProposalHead proposal_head     = ProposalHead::Full;

    bool operator==(const LoadOptions&) const = default;

    [[nodiscard]] bool mtp() const noexcept { return speculative == SpeculativeBackend::Mtp; }

    [[nodiscard]] bool dflash() const noexcept { return speculative == SpeculativeBackend::DFlash; }

    [[nodiscard]] bool dflash2() const noexcept {
        return speculative == SpeculativeBackend::DFlash2;
    }

    [[nodiscard]] bool dspine() const noexcept { return speculative == SpeculativeBackend::Dspine; }

    // The masked-block drafter family: a draft that arranges one (K+1)-wide masked block, owns a
    // context key/value history of its own, and has its rows' `backend_frontier` advanced by the
    // committed prefix rather than by one token per step. DSpine is that shape - block 16, one
    // anchor column plus K in [1,15] drafts, target taps in, a paged context per draft layer - so
    // it joins the gates that give a drafter its frame, its context KV and its (K+1)-wide verify
    // block (frames.cpp:165, planning/startup.cpp:94-108, session/sequence.cpp:74). DFlash and
    // DFlash2 answer exactly as they did: the set they were in is unchanged.
    [[nodiscard]] bool masked_draft() const noexcept {
        return dflash() || dflash2() || dspine();
    }

    [[nodiscard]] bool proposal_enabled() const noexcept {
        return purpose == EnginePurpose::Generation && speculative != SpeculativeBackend::None &&
               proposal_head == ProposalHead::Optimized;
    }

    [[nodiscard]] std::string_view speculative_component() const noexcept {
        switch (speculative) {
        case SpeculativeBackend::None:
            return {};
        case SpeculativeBackend::Mtp:
            return "mtp";
        case SpeculativeBackend::DFlash:
            return "dflash";
        case SpeculativeBackend::DFlash2:
            return "dflash2";
        case SpeculativeBackend::Dspine:
            return "dspine";
        }
        return {};
    }
};

[[nodiscard]] constexpr bool is_masked_draft_backend(SpeculativeBackend backend) noexcept {
    return backend == SpeculativeBackend::DFlash || backend == SpeculativeBackend::DFlash2
        || backend == SpeculativeBackend::Dspine;
}

[[nodiscard]] inline LoadOptions load_options(const EngineOptions& options) noexcept {
    return {.purpose       = options.purpose,
            .vision        = options.enable_vision,
            .speculative   = options.speculative.backend,
            .proposal_head = options.speculative.proposal_head};
}

} // namespace ninfer::models
