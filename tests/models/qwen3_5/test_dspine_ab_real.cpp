// P2 criterion ⑤: the first ACCEPTED/ROUND number for DSpine, verified against a REAL target.
//
// THE NUMBER THAT DECIDES THE PROJECT. D-40 put the bar at DFlash2's measured 2.998 accepted
// drafts/round (prod's own /metrics, 214,153 live rounds at K=7), and D-48's additive bound says
// the K=15 window has to buy 3.250 to pay for itself. D-42 closed the forward probe with the
// reason no probe could answer it: the accept op verifies against the TARGET's logits, the probe
// has no target, and on synthetic taps the prod policy is degenerate (1 of 16 candidates survives
// the filter, so T=0.9 and T=0 are the same arm). D-42 moved criterion ⑤ to the a5000-2 lane A/B
// on the premise that "the drafter cannot co-reside with the target tower on GPU 1".
//
// D-51 `DISPROVED THAT PREMISE`. The fidelity probe loaded the EXECUTOR's own shard - 712 objects,
// 20.43905 GiB - on GPU 1 (`device free 25113 -> 4183 MiB`), i.e. tower AND drafter resident, and
// then walked the drafter 32 rounds bit-exact. The artifact carries both. So a real round can be
// run and verified against the real target with no lane, no prod downtime and no image build.
// This probe is that round.
//
// WHY IT RUNS THE ENGINE RATHER THAN A HAND-ROLLED ROUND. The quantity is accepted tokens per
// round, and the engine is the only code that owns all four parts of a round: the target forward
// that produces the logits the verifier reads, the drafter (`run_dspine_draft` ->
// `execution::dspine_walk` + `dspine_propose_tail`, program/decode.cpp:148-207), the ONE verifier
// (`ops::speculative_accept_sparse_drafts`, program/decision.cpp:84-99 - the call DFlash2 makes,
// argument for argument, one op at one call site), and the accounting that turns its counts into
// `stats.speculative` (session/commit.cpp:294-308). A hand-rolled round would re-implement the
// 64-layer V3 tower forward, and every re-implementation is a place the number could be wrong.
// Running the engine means the accepted count IS prod's count: the same counters that produced
// 2.998, read through the same public `RequestStats`.
//
// WHAT IS REAL HERE. The target's own weights, the drafter's own weights, the tower's own logits,
// the engine's own accept op, the engine's own accepted/round accounting, the prod sampling policy
// (T=0.9, top_k 20, top_p 1.0, min_p 0.05) plus a T=0 arm, the real tokenizer, real text prompts.
//
// WHAT IT IS NOT. It is NOT the lane A/B: one card (GPU 1, RTX PRO 5000, pci c1:00) with basalt
// live, batch 1, EAGER (DSpine has no CUDA graph profile - program/graph_bodies.cpp:19 refuses
// loudly), a few prompts. So the ACCEPTED column is prod's own arithmetic and transfers; the
// ms/round and tok/s columns are a contended card's and do not (D-48's own caveat, same shape).
//
// THE SELF-REFERENCE TEETH (D-42's named failure: a drafter verified against its own distribution
// scores ~15/round). The accepted count is produced by the tower's argmax over the tower's head
// (`project_decode`, decision.cpp:53-66), computed before `decide_decode` runs; the drafter never
// writes that tensor. Two arms ride with it: the T=0 arm (greedy - an accepted draft must equal the
// tower's argmax token, so a self-referential chain scores ~0 there), and the per-position decay,
// which a self-referential chain does not have. If the number lands >10/round, this line is the
// first suspect list.
//
// THE ROW IT PUBLISHED (GPU 1, RTX PRO 5000 pci c1:00, CUDA_VISIBLE_DEVICES=1, the V3
// `…_dspine-nvfp4tables.ninfer` shard, K=15, batch 1, eager, 3 prompts, every count the engine's
// own). The artifact carries BOTH drafters - DFlash2 is the V3 tower's own - so the control arm of
// the SAME card, SAME prompts, SAME policy, SAME verifier exists in the same binary:
//
//   backend    K    arm      accepted drafts/round   prod-def (+1)   ms/round   rounds   p0   depth
//   DFlash2    7    T=0.9    1.6809                  2.6809            25.47       94     .72   p6
//   DFlash2    15   T=0.9    1.7802  (0..9)          2.7802            27.40      273     .70   p9
//   DFlash2    15   T=0      2.2308  (0..11)         3.2308            27.35      117     .67   p11
//   DSpine     15   T=0.9    0.8467  (0..5)          1.8467            27.60      411     .55   p5
//   DSpine     15   T=0      0.7746  (0..5)          1.7746            27.46      213     .45   p5
//
// THE HEADLINE: at the SAME window (K=15) on the SAME card and prompts, DSpine drafts 0.47x what
// the artifact's own DFlash2 drafter drafts (0.8467 against 1.7802) for the same round cost (27.6
// against 27.4 ms), and its per-position curve is BOTH lower at p0 (.55 against .70) AND steeper -
// it dies at p5 where DFlash2 reaches p9. The wide window is not buying DSpine what it buys
// DFlash2. The absolutes are under prod's 2.998 (that row is the 5090, graphs on, batch 8, prod
// traffic, D-48's own caveat), so the ratio at equal window is the reading that transfers.
//
// THE CAVEAT THAT NARROWS IT (a tooth, not a fix). `NINFER_AB_PROMPT` re-runs every arm on one
// prompt. On a 316-token prompt DFlash2 moves 1.78 -> 1.52 (-15 %) while DSpine moves 0.85 -> 0.22
// (-74 %), and its p0 falls .55 -> .17. A drafter whose drafts get WORSE as the context grows is
// the signature of a context feed that is not landing, and the three DSpine context arms this
// route had to add (prefill, commit, plan) are new and unreviewed. So 0.8467 is a LOWER BOUND on
// DSpine whose first suspect is named; the 0.47x deficit at equal window is what the harness
// supports today.
//
// THE ROUND DISTRIBUTION. The public counters publish, per request, `rounds`, `accepted_tokens`
// and the depth census `accepted_per_position[j]` = the number of rounds whose j-th draft was
// accepted. That census IS the distribution: rounds that accepted at least d drafts =
// accepted_per_position[d-1], so rounds that accepted exactly d = census[d-1] - census[d] and
// rounds that accepted none = rounds - census[0]. The probe prints the histogram, the min/max and
// the mean out of it, so the range over rounds is the engine's own count, not a probe estimate.
//
// NINFER_TEST_ARTIFACT  the V3-tailored DSpine artifact (skip, exit 77, when unset)
// NINFER_TEST_DEVICE    the visible CUDA ordinal to run on; required, no default (D-29)
// NINFER_AB_DRAFTS      the DSpine draft window K (default 15, the shipped window)
// NINFER_AB_OUTPUTS     output tokens per request (default 64)
// NINFER_AB_RUNS        requests per arm (default 3)
// NINFER_AB_MAX_CONTEXT context ceiling (default 1024; prefill_chunk is the same, 8 x 128)
// NINFER_AB_ARM         "both" (default) | "sampled" | "greedy"
// NINFER_AB_BAR         the bar, accepted drafts/round (default 2.998, D-40)
// NINFER_AB_BREAK_EVEN  the K=15 break-even (default 3.250, D-48's additive bound)
// NINFER_AB_ROUND_MS    prod's measured ms/round (default 19.137)
//
// Run it pinned to the free card:
//   CUDA_VISIBLE_DEVICES=1 NINFER_TEST_DEVICE=0 \
//   NINFER_TEST_ARTIFACT=/home/kv/models/dspine-v3/qwen3_8_27b_swift15_v3_dspine-nvfp4tables.ninfer \
//   ./tests/ninfer_qwen3_5_dspine_ab_real_test

#include "ninfer/engine.h"
#include "ninfer/types.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace ninfer;

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
constexpr std::size_t kPositions = 15;  // kDFlashDecodeMaximumDrafts

void note(const std::string& line) {
    std::printf("%s\n", line.c_str());
    std::fflush(stdout);
}

[[noreturn]] void stop(const std::string& message) {
    std::printf("STOP: %s\n", message.c_str());
    std::fflush(stdout);
    std::quick_exit(1);
}

void report_fault(const int signal) {
    std::puts("SIGNAL: the A/B harness died in its own process");
    std::fflush(stdout);
    std::quick_exit(90 + signal);
}

std::string setting(std::string_view name, std::string_view fallback = {}) {
    const char* value = std::getenv(std::string(name).c_str());
    return value && *value ? std::string(value) : std::string(fallback);
}

std::int32_t setting_integer(std::string_view name, std::int32_t fallback) {
    const auto text = setting(name);
    return text.empty() ? fallback : static_cast<std::int32_t>(std::stol(text));
}

double setting_real(std::string_view name, double fallback) {
    const auto text = setting(name);
    return text.empty() ? fallback : std::stod(text);
}

void cuda_ok(const cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        stop(std::string(what) + ": " + cudaGetErrorString(error));
    }
}

std::pair<std::uint64_t, std::uint64_t> card_memory(const int ordinal) {
    cuda_ok(cudaSetDevice(ordinal), "set device");
    std::uint64_t free_bytes = 0, total_bytes = 0;
    cuda_ok(cudaMemGetInfo(&free_bytes, &total_bytes), "memory info");
    return {free_bytes, total_bytes};
}

std::uint64_t at(const std::vector<std::uint64_t>& values, const std::size_t index) {
    return index < values.size() ? values[index] : 0;
}

// ---------------------------------------------------------------- the arms -----------------------

struct Arm {
    std::string key;    // "sampled" | "greedy"
    std::string label;
    std::string prompt;
    std::uint32_t outputs = 64;
    std::optional<float> temperature;
    std::optional<std::int32_t> top_k;
    std::optional<float> top_p;
    std::optional<float> min_p;
    std::uint64_t seed    = 0x50371153ULL;
};

struct Measure {
    std::int32_t runs = 0;
    std::uint64_t rounds = 0, accepted = 0, drafted = 0, tokens = 0, fallback = 0;
    std::vector<std::uint64_t> census;  // accepted_per_position, pooled over the runs
    std::vector<double> per_run;        // accepted/round of each run
    std::vector<double> round_ms;       // the run's decode-step exposure per round
    double decode_seconds = 0.0, wall_seconds = 0.0;
};

RequestOptions request_for(const Arm& arm) {
    RequestOptions request;
    request.execution.requested_output_tokens = arm.outputs;
    request.execution.sampling.temperature    = arm.temperature;
    request.execution.sampling.top_k          = arm.top_k;
    request.execution.sampling.top_p          = arm.top_p;
    request.execution.sampling.min_p          = arm.min_p;
    request.execution.sampling.seed           = arm.seed;
    request.stop.include_model_defaults       = false;
    return request;
}

void run(Engine& engine, const Arm& arm, Measure& out, SpeculativeBackend route) {
    const auto prompt = engine.tokenize_text(arm.prompt);
    for (std::int32_t index = 0; index < setting_integer("NINFER_AB_RUNS", 3); ++index) {
        const auto began  = std::chrono::steady_clock::now();
        const auto result = engine.generate(engine.prepare_tokens(prompt), request_for(arm));
        const auto wall   = std::chrono::duration_cast<
            std::chrono::duration<double, std::ratio<1, 1000>>>(
                std::chrono::steady_clock::now() - began);
        const auto& spec = result.stats.speculative;
        // A request's SpeculativeStats are request-owned counters that start at zero
        // (session/binding.cpp:196-198, accumulated per commit in session/commit.cpp:294-308), so
        // the result is this request's own delta: no engine-wide subtraction, no shared state.
        if (!spec.enabled || spec.backend != route) {
            stop("the request did not run on the named backend");
        }
        if (spec.draft_window != arm.outputs && spec.draft_window == 0) {
            stop("the DSpine round published no draft window");
        }
        ++out.runs;
        out.rounds   += spec.rounds;
        out.accepted += spec.accepted_tokens;
        out.drafted  += spec.drafted_tokens;
        out.fallback += spec.fallback_steps;
        out.tokens   += result.generated_token_ids.size();
        out.decode_seconds += result.stats.timings.decode_seconds;
        out.wall_seconds   += wall.count() / 1000.0;
        for (std::size_t position = 0; position < spec.accepted_per_position.size(); ++position) {
            if (position >= out.census.size()) {
                out.census.resize(spec.accepted_per_position.size(), 0);
            }
            out.census[position] += spec.accepted_per_position[position];
        }
        const auto per_round = spec.rounds ? double(spec.accepted_tokens) / double(spec.rounds)
                                            : 0.0;
        const auto ms_round  = spec.rounds
                                 ? 1000.0 * result.stats.timings.decode_seconds
                                       / double(spec.rounds)
                                 : 0.0;
        if (spec.rounds) {
            out.per_run.push_back(per_round);
            out.round_ms.push_back(ms_round);
        }
        std::printf("   run %d: prompt %4zu tokens %4zu rounds %4llu drafted %6llu accepted "
                    "%5llu  %7.4f acc/round  %7.3f ms/round  fallback %llu  K=%u  finish %d\n",
                    index + 1, prompt.size(), result.generated_token_ids.size(),
                    static_cast<unsigned long long>(spec.rounds),
                    static_cast<unsigned long long>(spec.drafted_tokens),
                    static_cast<unsigned long long>(spec.accepted_tokens), per_round, ms_round,
                    static_cast<unsigned long long>(spec.fallback_steps), spec.draft_window,
                    int(result.finish_reason));
        std::fflush(stdout);
    }
}

// The round histogram out of the depth census: rounds that accepted AT LEAST d drafts is
// census[d-1], so exactly d is census[d-1] - census[d].
std::vector<std::uint64_t> histogram_of(const Measure& measure) {
    std::vector<std::uint64_t> histogram(measure.census.size() + 1, 0);
    std::uint64_t reached = measure.rounds;
    for (std::size_t depth = 0; depth < measure.census.size(); ++depth) {
        const auto next = measure.census[depth];
        if (next > reached) {
            stop("the depth census is not monotone: more rounds reached a deeper position");
        }
        histogram[depth] = reached - next;  // rounds that accepted exactly `depth` drafts
        reached          = next;
    }
    histogram[measure.census.size()] = reached;  // accepted the whole window
    return histogram;
}

void report(const char* label, const Measure& measure, const double bar,
             const double break_even, const double prod_round_ms) {
    if (measure.rounds == 0) {
        std::printf("\n== %s == no drafted round\n", label);
        std::fflush(stdout);
        return;
    }
    const auto mean = double(measure.accepted) / double(measure.rounds);
    std::printf("\n== %s ==\n", label);
    std::printf("   runs %d, output tokens %llu, rounds %llu, drafted %llu, fallback %llu\n",
                measure.runs, static_cast<unsigned long long>(measure.tokens),
                static_cast<unsigned long long>(measure.rounds),
                static_cast<unsigned long long>(measure.drafted),
                static_cast<unsigned long long>(measure.fallback));
    std::printf("ACCEPTED/ROUND %10.4f   accepted drafts / rounds (N = %llu rounds) - the ratio "
                "DFlash2's 2.998 is on\n",
                mean, static_cast<unsigned long long>(measure.rounds));
    std::printf("PROD-DEFINITION %9.4f   (+1 for the verified target token: tokens / round)\n",
                mean + 1.0);
    std::printf("   bar DFlash2 K=7 = %.3f  |  K=15 break-even = %.3f  ->  %s\n", bar, break_even,
                mean >= break_even ? "BEATS the break-even"
                : mean >= bar      ? "between the bar and the break-even"
                                   : "BELOW the bar");
    if (!measure.per_run.empty()) {
        auto sorted = measure.per_run;
        std::sort(sorted.begin(), sorted.end());
        std::printf("   per-run accepted/round: min %.4f  p50 %.4f  max %.4f  (n=%zu runs)\n",
                    sorted.front(), sorted[sorted.size() / 2], sorted.back(), sorted.size());
    }
    const auto histogram = histogram_of(measure);
    std::size_t low = histogram.size(), high = 0;
    for (std::size_t depth = 0; depth < histogram.size(); ++depth) {
        if (histogram[depth] == 0) continue;
        low  = std::min(low, depth);
        high = std::max(high, depth);
    }
    std::printf("   per-ROUND range: %zu .. %zu accepted drafts (min/max over the %llu rounds)\n",
                low == histogram.size() ? 0 : low, high,
                static_cast<unsigned long long>(measure.rounds));
    std::printf("   histogram (accepted drafts -> rounds):\n       ");
    for (std::size_t depth = 0; depth < histogram.size(); ++depth) {
        if (histogram[depth] == 0) continue;
        std::printf("%zu:%llu ", depth, static_cast<unsigned long long>(histogram[depth]));
    }
    std::printf("\n");
    std::printf("   per-position p0..p%zu (the rounds whose j-th draft was accepted, over the "
                "rounds that reached it):\n       ",
                kPositions - 1);
    for (std::size_t position = 0; position < measure.census.size(); ++position) {
        const auto reached = position == 0 ? measure.rounds : at(measure.census, position - 1);
        if (reached == 0) break;
        std::printf("p%zu %.3f ", position, double(at(measure.census, position)) / double(reached));
    }
    std::printf("\n");
    if (!measure.round_ms.empty()) {
        auto sorted = measure.round_ms;
        std::sort(sorted.begin(), sorted.end());
        double total = 0.0;
        for (const auto value : sorted) { total += value; }
        const auto ms = total / double(sorted.size());
        std::printf("MS/ROUND %12.3f ms (mean over runs; prod's round is %.3f ms -> %.2fx)\n", ms,
                    prod_round_ms, ms / prod_round_ms);
        std::printf("   run ms/round min %.3f p50 %.3f max %.3f; decode tok/s %.1f\n",
                    sorted.front(), sorted[sorted.size() / 2], sorted.back(),
                    measure.decode_seconds > 0.0
                        ? double(measure.accepted + measure.rounds) / measure.decode_seconds
                        : 0.0);
    }
    std::fflush(stdout);
}
}  // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) {
        std::printf("skip: NINFER_TEST_ARTIFACT is not set\n");
        return 77;
    }
    const auto asked = setting("NINFER_TEST_DEVICE");
    if (asked.empty()) {
        std::printf("skip: NINFER_TEST_DEVICE is not set (a device probe does not guess a card)\n");
        return 77;
    }
    const int device = std::atoi(asked.c_str());
    int visible      = 0;
    if (cudaGetDeviceCount(&visible) != cudaSuccess || visible <= 0 || device >= visible) {
        std::printf("skip: NINFER_TEST_DEVICE=%d, %d visible\n", device, visible);
        return 77;
    }
    std::signal(SIGSEGV, report_fault);
    std::signal(SIGABRT, report_fault);

    const auto drafts        = static_cast<std::uint32_t>(setting_integer("NINFER_AB_DRAFTS", 15));
    const auto outputs       = static_cast<std::uint32_t>(setting_integer("NINFER_AB_OUTPUTS", 64));
    const auto max_context   = static_cast<std::uint32_t>(setting_integer("NINFER_AB_MAX_CONTEXT", 1024));
    const auto arm           = setting("NINFER_AB_ARM", "both");
    const auto bar           = setting_real("NINFER_AB_BAR", 2.998);
    const auto break_even    = setting_real("NINFER_AB_BREAK_EVEN", 3.250);
    const auto prod_round_ms = setting_real("NINFER_AB_ROUND_MS", 19.137);

    cuda_ok(cudaSetDevice(device), "set device");
    cudaDeviceProp properties{};
    cuda_ok(cudaGetDeviceProperties(&properties, device), "device properties");
    const auto [free_before, total] = card_memory(device);
    std::printf("device: visible ordinal %d, name \"%s\", pci %02x:%02x, %llu MiB total, %llu MiB "
                "free\n",
                device, properties.name, properties.pciBusID, properties.pciDeviceID,
                static_cast<unsigned long long>(total / 1024 / 1024),
                static_cast<unsigned long long>(free_before / 1024 / 1024));
    std::printf("        CUDA_VISIBLE_DEVICES=\"%s\"\n",
                setting("CUDA_VISIBLE_DEVICES", "<unset>").c_str());
    note("route: ninfer::Engine, --spec dspine, EAGER (DSpine has no CUDA graph profile, "
         "graph_bodies.cpp:19), batch 1, the engine's own accept op");
    std::fflush(stdout);

    EngineOptions options;
    options.artifact_path                 = artifact;
    options.device                        = device;
    options.purpose                       = EnginePurpose::Generation;
    options.max_context                   = max_context;
    options.prefill_chunk                 = max_context;
    options.max_concurrency               = 1;
    options.use_cuda_graph                = false;
    options.enable_vision                 = false;
    options.kv_cache                      = KvCacheStorage::BFloat16;
    // The control-of-the-same-shape probe. The artifact family names one drafter; DFlash2's plan
    // gate (planning/startup.cpp:676-684) asks exactly "does this target carry the drafter the
    // option names", so naming dflash2 here is the read-only answer to "is there a DFlash2 arm of
    // this artifact to measure against at 2.998". `none` is the no-speculative decoding cost.
    const auto backend_text = setting("NINFER_AB_BACKEND", "dspine");
    options.speculative.backend = backend_text == "dflash2"
                                       ? SpeculativeBackend::DFlash2
                                   : backend_text == "none" ? SpeculativeBackend::None
                                                            : SpeculativeBackend::Dspine;
    options.speculative.draft_tokens = backend_text == "none" ? 0U : drafts;
    options.speculative.proposal_head     = ProposalHead::Full;

    const auto load_began = std::chrono::steady_clock::now();
    const auto options_route = options.speculative.backend;
    Engine engine(options);
    const auto load_ms    = std::chrono::duration_cast<std::chrono::duration<double,
            std::ratio<1, 1000>>>(std::chrono::steady_clock::now() - load_began).count();
    const auto [free_after, total_after] = card_memory(device);
    (void)total_after;
    std::printf("load: the shard (the target tower AND the DSpine drafter) opened in %.0f ms; "
                "device free %llu -> %llu MiB (the shard costs %.5f GiB)\n",
                load_ms, static_cast<unsigned long long>(free_before / 1024 / 1024),
                static_cast<unsigned long long>(free_after / 1024 / 1024),
                double(free_before - free_after) / kGiB);
    std::fflush(stdout);

    const std::string sky = "Explain, in plain language, why the sky is blue and why sunsets are "
                             "red. Give the shortest answer first, then one example.";
    const std::string move = "Write a short, practical checklist for migrating a self-hosted web "
                             "service to a new machine without downtime. Number the steps.";
    // The context-sensitivity tooth. The three drafter-context arms this route had to add
    // (program/execution.cpp, program/draft_commit.cpp, program/planning/startup.cpp) are what put
    // the prompt's target-tap field into the drafter's five paged contexts. If they did not work,
    // the drafter would draft from an empty history and the accepted count would be INDEPENDENT of
    // the prompt - so a prompt long enough to carry real context must accept measurably better
    // than a 12-token one. Named by NINFER_AB_PROMPT so the same binary can be re-run on any text.
    const std::string long_prompt = setting(
        "NINFER_AB_PROMPT",
        "The following is a maintenance manual for a self-hosted web service on a single Linux "
        "machine with one Nginx front door, one Postgres database and a Python API behind Uvicorn. "
        "Read it carefully and then answer the single question at the end. Section one, the boot "
        "order: the database opens first, because the API reads its connection string out of the "
        "database at startup, and the front door opens last, because it reads its upstream list "
        "out of the service registry. Section two, the log files: the front door writes an access "
        "log and an error log, the API writes a structured event log, the database writes a slow "
        "query log, and all four are rotated nightly at ten minutes past two by a timer, not a "
        "cron job, so that a missed run is visible as a failed unit. Section three, the backup "
        "recipe: every night at ten minutes past three a dump runs, it is written to a second disk "
        "and then to object storage, and a restore is exercised on the first Sunday of the month "
        "against a scratch database so that a dump that cannot be played back is found before it "
        "is needed. Section four, the migration: a new machine joins the load balancer as a drain "
        "target first, takes a copy of the database by streaming replication, is promoted by "
        "flipping one pointer in the registry, and the old machine stays as a drain target for one "
        "week so that a rollback is a pointer flip in the other direction. Section five, the two "
        "things that break most often: a certificate that expires while the renewal timer is "
        "masked by a failed unit, and a schema migration that was applied to the primary and never "
        "to the replica, which is why the migration is a part of the deploy and not a manual step. "
        "Question: in the order the manual gives, what is the very first thing that must be true "
        "before the front door can open? Explain the reason for that order.");
    const auto custom = setting("NINFER_AB_PROMPT");
    const std::vector<Arm> arms{
        Arm{"sampled", "arm A - prod policy T=0.9, top_k 20, top_p 1.0, min_p 0.05 (prompt 1)",
            custom.empty() ? sky : long_prompt, outputs, 0.9f, 20, 1.0f, 0.05f},
        Arm{"greedy", "arm B - the T=0 arm (greedy argmax, no filter), prompt 1",
            custom.empty() ? sky : long_prompt, outputs, 0.0f, std::nullopt, std::nullopt,
            std::nullopt},
        Arm{"sampled", "arm C - prod policy, prompt 2 (the number is not one prompt's)",
            custom.empty() ? move : long_prompt, outputs, 0.9f, 20, 1.0f, 0.05f},
    };

    std::vector<std::pair<std::string, Measure>> per_arm;
    Measure sampled, greedy;
    for (const auto& item : arms) {
        if (item.key == "sampled" && arm == "greedy") continue;
        if (item.key == "greedy" && arm == "sampled") continue;
        std::printf("\n--- %s ---\n", item.label.c_str());
        std::fflush(stdout);
        Measure measure;
        try {
            run(engine, item, measure, options_route);
        } catch (const std::exception& failure) {
            note(std::string("STOP: the DSpine round stopped at: ") + failure.what());
            return 1;
        }
        per_arm.emplace_back(item.label, measure);
        auto& pooled = item.key == "greedy" ? greedy : sampled;
        pooled.runs += measure.runs;
        pooled.rounds += measure.rounds;
        pooled.accepted += measure.accepted;
        pooled.drafted += measure.drafted;
        pooled.tokens += measure.tokens;
        pooled.fallback += measure.fallback;
        pooled.decode_seconds += measure.decode_seconds;
        pooled.wall_seconds += measure.wall_seconds;
        pooled.per_run = measure.per_run;
        pooled.round_ms = measure.round_ms;
        for (std::size_t position = 0; position < measure.census.size(); ++position) {
            if (position >= pooled.census.size()) {
                pooled.census.resize(measure.census.size(), 0);
            }
            pooled.census[position] += measure.census[position];
        }
    }

    for (const auto& [label, measure] : per_arm) {
        report(label.c_str(), measure, bar, break_even, prod_round_ms);
    }
    if (arm != "greedy") {
        report("POOLED prod-policy arm (T=0.9)", sampled, bar, break_even, prod_round_ms);
    }
    if (arm != "sampled") {
        report("POOLED T=0 arm", greedy, bar, break_even, prod_round_ms);
    }

    const auto [free_end, total_end] = card_memory(device);
    (void)total_end;
    std::printf("\nmemory: device free at the end %llu MiB of %llu MiB total\n",
                static_cast<unsigned long long>(free_end / 1024 / 1024),
                static_cast<unsigned long long>(total / 1024 / 1024));
    note("NOTE: harness-level. One card (GPU 1, basalt live), batch 1, eager, two prompts. The "
         "ACCEPTED column is prod's own accounting and transfers; the ms/round and tok/s columns "
         "are a contended card's, not the 5090 lane A/B's.");
    std::fflush(stdout);
    return 0;
}
