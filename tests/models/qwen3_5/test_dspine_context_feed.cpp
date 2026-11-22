// H1 of the D-53 context-feed diagnostic: IS THE DSpine DRAFT PATH ACTUALLY FED THE TARGET'S
// STATE? (D-53's named suspect: the three context arms - program/execution.cpp prefill,
// program/draft_commit.cpp commit, program/planning/startup.cpp local-state - are new,
// unreviewed code, and a drafter that gets worse as the context grows is the signature of a
// context feed that is not landing.)
//
// THE DECISIVE TEST, in the two forms the task names.
//   (1) PERTURB THE PAGES. Append the drafter's context, then overwrite the bytes of a CONTEXT
//       PAGE on the device, then walk. If the published streams do not move, the walk does not
//       read what the append wrote - the feed is decorative. Perturb ONE page at a time over a
//       context that spans TWO physical pages: a stream that moves for page 0 and not for page 1
//       is a page-index defect, and a stream that moves for both is a consumed context.
//   (2) THE PAGE-INDEX EVIDENCE. Print the logical-to-physical page table the round publishes,
//       the page indices `dspine_append_context` writes (positions / kPagedKVPageSize), the page
//       indices `context_softmax_attention` reads (context length / kPagedKVPageSize), and the
//       byte census (non-zero bytes) of every page, per layer, before and after the append.
//   (3) POSITIVE CONTROL. Re-seed the TARGET-TAP FIELD alone and re-run the whole round: the
//       published streams must move. If they do not, the tap field is not the walk's input and
//       the page test above is reading the wrong thing.
//
// WHY THE DRAFTER ALONE IS ENOUGH FOR THIS, AND WHAT IT IS NOT. The quantity under test is a
// WIRING property: does the byte stream the append writes reach the bytes the attention reads.
// It is a property of the executor's own `dspine_append_context` + `dspine_walk`, over the
// engine's own `PagedKVCache`, the engine's own page geometry and the engine's own
// `execution::Parameters` binding of the real V3 drafter. It is NOT an acceptance number -
// accepted/round needs the real target (D-42, and `test_dspine_ab_real.cpp` is that probe), and
// the activation streams here are synthetic carriers. Nothing here widens a tolerance, edits a
// kernel, a numeric, a shape gate or an acceptance policy.
//
// THE SKELETON IS NOT INVENTED HERE. The device bring-up - the `Probe` struct, the `Device`/
// `DeviceContext`/`CudaDeviceScope` use, the artifact `Reader` open, the `buffer` allocation, the
// `upload`/`download` transfer boundary - and the executor's own load path
// (read_definition -> describe_model -> resolve_load -> materialize_model -> execution::Parameters)
// are copied from the probe that runs this artifact and this route
// (tests/models/qwen3_5/test_dspine_exec_fidelity.cpp, D-51: 319 checks / 0 reds), which copies
// them from test_dspine_tail_cost.cpp (D-47: a hand-rolled device-bind setup has hung this harness
// twice; the working path is the only accepted setup). The drafter's paged context is the engine's
// own `PagedKVCache` over `plan_paged_kv_cache`, built the same way.
//
// NINFER_TEST_ARTIFACT  the V3-tailored DSpine artifact (skip, exit 77, when unset)
// NINFER_TEST_DEVICE    the visible CUDA ordinal to run on; required, no default (D-29)
// NINFER_CTX_CONTEXT    context tokens the round appends and attends (default 128 = 2 pages)
// NINFER_CTX_SLACK      MiB of device headroom the load insists on (default 768)
//
// Run it pinned to the free card:
//   CUDA_VISIBLE_DEVICES=1 NINFER_TEST_DEVICE=0 \
//   NINFER_TEST_ARTIFACT=/home/kv/models/dspine-v3/qwen3_8_27b_swift15_v3_dspine-nvfp4tables.ninfer \
//   ./tests/ninfer_qwen3_5_dspine_context_feed_real_test

#include "artifact/reader.h"
#include "artifact/views.h"
#include "core/arena.h"
#include "core/buffer.h"
#include "core/context_pool.h"
#include "core/device.h"
#include "core/layout.h"
#include "core/paged_kv_cache.h"
#include "core/paged_kv_geometry.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "core/weight_view.h"
#include "models/common/dspine/config.h"
#include "models/common/dspine/dspine_serial_tail.h"
#include "models/qwen3_5/execution/dspine.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/execution/profile.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/linear_topk.h"
#include "ninfer/ops/softmax_attention.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
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
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::models;
namespace dspine    = ninfer::models::dspine;
namespace q35       = ninfer::models::qwen3_5;
namespace execution = q35::execution;

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
constexpr std::int32_t kSlots = 16;
constexpr std::int32_t kBatch = 1;
constexpr std::int32_t kTokenDomain = 248077;

void report_fault(const int signal) {
    std::puts("SIGNAL: a device-side fault in the harness itself");
    std::fflush(stdout);
    std::quick_exit(90 + signal);
}

std::string setting(std::string_view name, std::string_view fallback = {}) {
    const char* value = std::getenv(std::string(name).c_str());
    return value && *value ? std::string(value) : std::string(fallback);
}

void cuda_ok(const cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(error));
    }
}

[[nodiscard]] std::int32_t dimension(const std::uint64_t value) {
    return static_cast<std::int32_t>(value);
}
[[nodiscard]] std::int32_t dimension(const std::int32_t value) { return value; }
template <class Value>
[[nodiscard]] std::int32_t dim(const Value value) {
    return static_cast<std::int32_t>(value);
}

// ---------------------------------------------------------------- the probe (the working probe's) ---

struct Probe {
    Probe(const std::string& path, const int ordinal)
        : reader(path), device(Device::cuda(ordinal)), context(ordinal) {
        cuda_ok(cudaSetDevice(ordinal), "set device");
        cudaDeviceProp properties{};
        cuda_ok(cudaGetDeviceProperties(&properties, ordinal), "device properties");
        std::printf("device: visible ordinal %d, name \"%s\", pci %02x:%02x, %llu MiB total\n",
                    ordinal, properties.name, properties.pciBusID, properties.pciDeviceID,
                    static_cast<unsigned long long>(properties.totalGlobalMem / 1024 / 1024));
        std::printf("        CUDA_VISIBLE_DEVICES=\"%s\"\n",
                    setting("CUDA_VISIBLE_DEVICES", "<unset>").c_str());
    }

    [[nodiscard]] std::pair<std::uint64_t, std::uint64_t> memory() const {
        std::uint64_t free_bytes = 0, total_bytes = 0;
        cuda_ok(cudaMemGetInfo(&free_bytes, &total_bytes), "memory info");
        return {free_bytes, total_bytes};
    }

    [[nodiscard]] Tensor buffer(const std::size_t bytes, const DType dtype,
                                 const std::initializer_list<std::int32_t> shape) {
        storage.push_back(std::make_unique<DeviceBuffer>(device, bytes));
        auto& slot = *storage.back();
        const CudaDeviceScope clearing(device);
        cuda_ok(cudaMemset(slot.data(), 0, slot.size()), "clear buffer");
        return Tensor(slot.span(), dtype, shape);
    }

    CudaExecutionView cuda() const { return context.execution_view(); }

    artifact::Reader reader;
    Device device;
    DeviceContext context;
    std::vector<std::unique_ptr<DeviceBuffer>> storage;
};

namespace {
Probe* owner = nullptr;
}  // namespace

DeviceBuffer* owning_buffer(void* pointer) {
    for (const auto& slot : owner->storage) {
        const auto base      = static_cast<std::byte*>(slot->data());
        const auto target    = static_cast<std::byte*>(pointer);
        if (target >= base && target < base + slot->size()) {
            return slot.get();
        }
    }
    return nullptr;
}

void upload(const void* destination, const void* source, const std::size_t bytes,
             const char* what) {
    if (destination == nullptr || bytes == 0) {
        throw std::runtime_error(std::string("the probe uploaded ") + what +
                                 " to a destination it does not own");
    }
    const auto target = static_cast<std::byte*>(const_cast<void*>(destination));
    auto* buffer      = owning_buffer(const_cast<void*>(destination));
    if (buffer == nullptr) {
        throw std::runtime_error(std::string("the probe uploaded ") + what +
                                 " to a buffer it does not own");
    }
    buffer->copy_from_host(source, bytes, std::size_t(target - static_cast<std::byte*>(
                                                        buffer->data())));
}

void download(void* destination, const void* source, const std::size_t bytes, const char* what) {
    const auto target = static_cast<std::byte*>(const_cast<void*>(source));
    auto* buffer      = owning_buffer(const_cast<void*>(source));
    if (buffer == nullptr) {
        throw std::runtime_error(std::string("the probe read back ") + what +
                                 " from a buffer it does not own");
    }
    buffer->copy_to_host(destination, bytes,
                          std::size_t(target - static_cast<std::byte*>(buffer->data())));
}

// A deterministic BF16 activation field, column-major. `phase` shifts the whole field: the
// positive control (3) re-seeds the tap field with a phase and nothing else changes.
void seed(const Tensor& stream, const float phase = 0.0f) {
    const auto columns = std::int64_t(stream.ne[1]) * stream.ne[2];
    const auto rows    = std::int64_t(stream.ne[0]);
    std::vector<std::uint16_t> host(std::size_t(rows * columns));
    for (std::int64_t column = 0; column < columns; ++column) {
        for (std::int64_t row = 0; row < rows; ++row) {
            const float value = 1.7f * std::sin(0.00390621f * float(row + 1)
                                               + 0.217747f * float(column + 3) + phase);
            const auto bits = std::bit_cast<std::uint32_t>(value);
            host[std::size_t(column) * rows + row] =
                static_cast<std::uint16_t>((bits >> 16) + ((bits >> 16) & 1u));
        }
    }
    upload(stream.data, host.data(), host.size() * 2, "seed stream");
}

void upload_i32(const Tensor& stream, const std::vector<std::int32_t>& values, const char* what) {
    upload(stream.data, values.data(), values.size() * sizeof(std::int32_t), what);
}

[[nodiscard]] std::vector<std::int32_t> download_i32(const Tensor& stream) {
    std::vector<std::int32_t> values(std::size_t(stream.numel()));
    download(values.data(), stream.data, values.size() * sizeof(std::int32_t), "read i32 stream");
    return values;
}

[[nodiscard]] std::vector<std::uint16_t> download_bf16(const Tensor& stream) {
    std::vector<std::uint16_t> values(std::size_t(stream.numel()));
    download(values.data(), stream.data, values.size() * 2, "read bf16 stream");
    return values;
}

[[nodiscard]] std::vector<float> download_f32(const Tensor& stream) {
    std::vector<float> values(std::size_t(stream.numel()));
    download(values.data(), stream.data, values.size() * sizeof(float), "read fp32 stream");
    return values;
}

[[nodiscard]] std::uint64_t digest(const void* bytes, const std::size_t count) {
    const auto* data = static_cast<const std::uint8_t*>(bytes);
    std::uint64_t hash = 14695981039346656037ULL;
    for (std::size_t index = 0; index < count; ++index) {
        hash ^= static_cast<std::uint64_t>(data[index]);
        hash *= 1099511628211ULL;
    }
    return hash;
}

[[nodiscard]] std::uint64_t digest_of(const Tensor& stream) {
    std::vector<std::byte> bytes(std::size_t(stream.bytes()));
    download(bytes.data(), stream.data, bytes.size(), "fingerprint a round stream");
    return digest(bytes.data(), bytes.size());
}

template <class Function>
void step(const char* label, Function&& body) {
    try {
        body();
        cuda_ok(cudaGetLastError(), "launch");
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string("GATE at ") + label + ": " + error.what());
    }
}

// ---------------------------------------------------------------- the drafter's paged context ----
//
// The engine's own PagedKVCache over a device page arena and a fixed execution table, at the
// drafter's own head_dim / kv_heads / layer count, with an IDENTITY logical-to-physical page
// mapping (logical page p -> byte offset p * stride) so the page a write lands in and the page a
// read selects are named by the same integer the printouts name.
struct DrafterContext {
    DrafterContext(const std::uint32_t layers, const std::int32_t head_dim,
                   const std::int32_t kv_heads, const std::int32_t capacity) {
        LayoutBuilder builder;
        const auto layout = q35::plan_paged_kv_cache(builder, layers, dimension(capacity), kv_heads,
                                                     head_dim, KvCacheStorage::BFloat16, kBatch);
        pages_      = layout.pages.page_bytes;
        alignment_  = layout.pages.alignment;
        page_tokens_ = layout.pages.geometry.page_tokens;
        stride_     = ((pages_ + alignment_ - 1) / alignment_) * alignment_;
        logical_    = layout.execution_tables.spec.logical_page_capacity;
        records_.emplace(owner->device, stride_ * std::size_t(logical_));
        records_->fill(0);
        tables_.emplace(owner->device, builder.finish(256, "dspine context page tables"));
        cache_ = std::make_unique<q35::PagedKVCache>(records_->span(), tables_->span(), layout);
        staging_.emplace(sizeof(std::int64_t) * std::size_t(logical_));
        offsets_.resize(std::size_t(logical_));
        for (std::size_t page = 0; page < offsets_.size(); ++page) {
            offsets_[page] = std::int64_t(page) * std::int64_t(stride_);
        }
        publish();
    }

    void publish() {
        cache_->tables().publish(0, 0, std::span<std::int64_t>(offsets_), staging_->span(),
                                  owner->cuda());
        owner->context.synchronize();
    }

    void clear() { records_->fill(0); }

    // Overwrite one LOGICAL page of one layer, both planes, with a byte pattern: the perturbation.
    // A page record carries every layer's planes at a fixed inner offset, so the record byte
    // offset the page table names plus the plane's own region offset is the page a read selects.
    void perturb(const std::size_t layer, const std::size_t page, const int pattern) {
        const auto view = cache_->batch_layer_view(std::uint32_t(layer));
        for (const auto& plane : {std::pair<const void*, std::size_t>{
                 view.k_pages.data(), view.k_pages.plane.tensor.region.bytes},
             {view.v_pages.data(), view.v_pages.plane.tensor.region.bytes}}) {
            cuda_ok(cudaMemsetAsync(static_cast<std::byte*>(const_cast<void*>(plane.first))
                                        + std::size_t(offsets_[page]),
                                    pattern, plane.second, owner->cuda().stream()),
                    "perturb a context page");
        }
        owner->context.synchronize();
    }

    [[nodiscard]] std::size_t page_bytes_of(const std::size_t layer) const {
        const auto view = cache_->batch_layer_view(std::uint32_t(layer));
        return view.k_pages.plane.tensor.region.bytes + view.v_pages.plane.tensor.region.bytes;
    }

    [[nodiscard]] std::size_t nonzero_bytes(const std::size_t layer, const std::size_t page) const {
        const auto view = cache_->batch_layer_view(std::uint32_t(layer));
        std::size_t total = 0;
        for (const auto& plane : {std::pair<const void*, std::size_t>{
                 view.k_pages.data(), view.k_pages.plane.tensor.region.bytes},
             {view.v_pages.data(), view.v_pages.plane.tensor.region.bytes}}) {
            std::vector<std::uint8_t> host(std::size_t(plane.second));
            cuda_ok(cudaMemcpy(host.data(),
                               static_cast<const std::byte*>(plane.first)
                                   + std::size_t(offsets_[page]),
                               host.size(), cudaMemcpyDeviceToHost),
                    "read back a context page");
            for (const auto byte : host) {
                total += byte != 0;
            }
        }
        return total;
    }

    [[nodiscard]] PagedKVBatchLayerView layer(const std::size_t index) const {
        return cache_->batch_layer_view(std::uint32_t(index));
    }

    std::uint64_t pages_ = 0, alignment_ = 0, stride_ = 0, page_tokens_ = 0;
    std::uint32_t logical_ = 0;
    std::optional<DeviceBuffer> records_, tables_;
    std::optional<PinnedHostBuffer> staging_;
    std::vector<std::int64_t> offsets_;
    std::unique_ptr<q35::PagedKVCache> cache_;
};

// ---------------------------------------------------------------- the round's streams ----------
struct Round {
    Tensor ids, positions, anchors, lengths, valid, table_rows, context_positions, context_lengths,
        context_counts, taps;
    Tensor hidden, residual, normalized, rms, codes, codes_out, packed, message, delta, normed,
        stash, intermediate, query, key, value, attended, query_raw, key_raw;
    Tensor proposal, candidates, scores, idx_scores, table, tail_cand, tail_drafts, tail_q,
        tail_index, index, configs, draw_positions;
    Tensor drafts, candidate_ids, proposal_q;
};

[[nodiscard]] Round allocate(const std::int32_t h, const std::int32_t w, const std::int32_t b,
                              const std::int32_t ctx, const std::int32_t c, const std::int32_t r,
                              const std::int32_t taps, const std::int32_t q, const std::int32_t k,
                              const std::int32_t d, const std::int32_t inter,
                              const std::int32_t drafts) {
    Round out;
    const auto integers = [&](const std::int64_t count,
                               const std::initializer_list<std::int32_t> shape) {
        return owner->buffer(std::size_t(4) * count, DType::I32, shape);
    };
    const auto stream = [&](const std::int64_t count,
                             const std::initializer_list<std::int32_t> shape) {
        return owner->buffer(std::size_t(2) * count, DType::BF16, shape);
    };
    const auto reals = [&](const std::int64_t count,
                            const std::initializer_list<std::int32_t> shape) {
        return owner->buffer(std::size_t(4) * count, DType::FP32, shape);
    };
    out.ids               = integers(w * b, {w, b, 1, 1});
    out.positions         = integers(w * b, {w, b, 1, 1});
    out.context_positions = integers(ctx * b, {ctx, b, 1, 1});
    out.context_lengths   = integers(b, {b, 1, 1, 1});
    out.context_counts    = integers(b, {b, 1, 1, 1});
    out.lengths           = integers(b, {b, 1, 1, 1});
    out.valid             = integers(b, {b, 1, 1, 1});
    out.table_rows        = integers(b, {b, 1, 1, 1});
    out.anchors           = integers(b, {b, 1, 1, 1});
    out.taps              = stream(std::int64_t(h) * taps * ctx * b, {h, taps, ctx, b});
    out.hidden            = stream(std::int64_t(h) * w * b, {h, w, b, 1});
    out.residual          = stream(std::int64_t(h) * w * b, {h, w, b, 1});
    out.normalized         = stream(std::int64_t(h) * w * b, {h, w, b, 1});
    out.delta             = stream(std::int64_t(h) * w * b, {h, w, b, 1});
    out.normed            = stream(std::int64_t(h) * w * b, {h, w, b, 1});
    out.stash             = stream(std::int64_t(h) * w * b, {h, w, b, 1});
    out.proposal          = stream(std::int64_t(h) * drafts * b, {h, drafts, b, 1});
    out.intermediate      = stream(std::int64_t(inter) * w * b, {inter, w, b, 1});
    out.rms               = reals(std::int64_t(w) * b, {w * b, 1, 1, 1});
    out.codes             = stream(std::int64_t(c) * w * b, {c, w, b, 1});
    out.codes_out         = stream(std::int64_t(c) * w * b, {c, w, b, 1});
    out.packed            = stream(std::int64_t(3 * r) * w * b, {3 * r, w, b, 1});
    out.message           = stream(std::int64_t(r) * w * b, {r, w, b, 1});
    out.query             = stream(std::int64_t(q) * w * b, {d, q / d, w, b});
    out.query_raw         = stream(std::int64_t(q) * w * b, {d, q / d, w, b});
    out.key               = stream(std::int64_t(k) * w * b, {d, k / d, w, b});
    out.key_raw           = stream(std::int64_t(k) * w * b, {d, k / d, w, b});
    out.value             = stream(std::int64_t(k) * w * b, {d, k / d, w, b});
    out.attended          = stream(std::int64_t(q) * w * b, {d, q / d, w, b});
    out.candidates        = integers(std::int64_t(kSlots) * drafts * b, {kSlots, drafts, b, 1});
    out.scores            = reals(std::int64_t(kSlots) * drafts * b, {kSlots, drafts, b, 1});
    out.idx_scores        = integers(std::int64_t(kSlots) * drafts * b, {kSlots, drafts, b, 1});
    out.table             = reals(std::int64_t(b) * kSlots * kSlots * drafts,
                                  {b, kSlots, kSlots, drafts});
    out.tail_cand         = integers(std::int64_t(b) * drafts * kSlots, {b, drafts, kSlots, 1});
    out.tail_drafts       = integers(std::int64_t(b) * drafts, {b, drafts, 1, 1});
    out.tail_q            = reals(std::int64_t(b) * drafts * kSlots, {b, drafts, kSlots, 1});
    out.tail_index        = integers(std::int64_t(b) * drafts * kSlots, {b, drafts, kSlots, 1});
    out.index             = integers(std::int64_t(b) * drafts * kSlots, {b, drafts, kSlots, 1});
    out.draw_positions    = integers(b, {b, 1, 1, 1});
    out.configs = owner->buffer(sizeof(ops::SamplingConfig) * std::size_t(b), DType::FP32,
                                 {b, 1, 1, 1});
    out.drafts        = integers(std::int64_t(drafts) * b, {drafts, b, 1, 1});
    out.candidate_ids = integers(std::int64_t(kSlots) * drafts * b, {kSlots, drafts, b, 1});
    out.proposal_q    = reals(std::int64_t(kSlots) * drafts * b, {kSlots, drafts, b, 1});
    return out;
}

struct Geometry {
    std::int32_t h, w, b, ctx, c, r, taps, q, k, d, inter, drafts, qh, kvh;
    float rms_eps, rope_theta, bridge_eps;
    std::int32_t mask_token;
};

// The round reset (D-42) and the round's inputs. `context` is the number of appended context
// tokens; the block is `w` wide on top of it.
void reset_round(const Geometry& g, Round& round, DrafterContext& pages, MemoryArena& arena,
                 const std::int32_t context, const float tap_phase) {
    const CudaDeviceScope settling(owner->device);
    const auto scratch = arena.storage();
    cuda_ok(cudaMemset(scratch.data, 0, scratch.bytes), "reset the round's shared workspace");
    arena.reset();
    const std::vector<Tensor*> streams{
        &round.ids,           &round.positions,       &round.context_positions, &round.lengths,
        &round.valid,         &round.context_lengths, &round.context_counts,    &round.table_rows,
        &round.anchors,       &round.taps,            &round.hidden,             &round.residual,
        &round.normalized,    &round.rms,             &round.codes,              &round.codes_out,
        &round.packed,        &round.message,         &round.delta,              &round.normed,
        &round.stash,         &round.intermediate,    &round.query,              &round.key,
        &round.value,         &round.attended,        &round.query_raw,          &round.key_raw,
        &round.proposal,      &round.candidates,      &round.scores,             &round.idx_scores,
        &round.table,         &round.tail_cand,       &round.tail_drafts,        &round.tail_q,
        &round.tail_index,    &round.index,           &round.draw_positions,     &round.configs,
        &round.drafts,        &round.candidate_ids,   &round.proposal_q};
    for (auto* stream : streams) {
        cuda_ok(cudaMemset(stream->data, 0, stream->bytes()), "reset a round stream");
    }
    pages.clear();
    const std::vector<std::int32_t> anchors{40901}, lengths{context}, valid{kSlots},
        context_lengths{context}, context_counts{context}, table_rows{0};
    upload_i32(round.anchors, anchors, "anchors");
    upload_i32(round.lengths, lengths, "block lengths");
    upload_i32(round.valid, valid, "valid columns");
    upload_i32(round.context_lengths, context_lengths, "context lengths");
    upload_i32(round.context_counts, context_counts, "context counts");
    upload_i32(round.table_rows, table_rows, "page table rows");
    std::vector<std::int32_t> context_positions(std::size_t(context * g.b));
    for (std::int32_t index = 0; index < context * g.b; ++index) {
        context_positions[std::size_t(index)] = index;
    }
    upload_i32(round.context_positions, context_positions, "context positions");
    std::vector<std::int32_t> draws(std::size_t(g.b));
    for (std::int32_t index = 0; index < g.b; ++index) {
        draws[std::size_t(index)] = 1024 + index;
    }
    upload_i32(round.draw_positions, draws, "tail draw positions");
    std::vector<ops::SamplingConfig> configs(std::size_t(g.b), ops::SamplingConfig{});
    for (auto& config : configs) {
        config.temperature = 0.9f;
        config.top_k       = 20;
        config.top_p       = 1.0f;
        config.min_p       = 0.05f;
        config.seed        = 0x50371153ULL;
    }
    upload(round.configs.data, configs.data(), configs.size() * sizeof(ops::SamplingConfig),
           "tail configs");
    seed(round.taps, tap_phase);
    pages.publish();
}

// The published streams of a round, fingerprinted.
struct Finger {
    std::vector<std::pair<std::string, std::uint64_t>> parts;

    [[nodiscard]] std::vector<std::string> parted_against(const Finger& other) const {
        std::vector<std::string> out;
        for (std::size_t index = 0; index < parts.size(); ++index) {
            if (parts[index].second != other.parts[index].second) {
                out.push_back(parts[index].first);
            }
        }
        return out;
    }
};

[[nodiscard]] Finger fingerprint(Round& round) {
    Finger out;
    for (const auto& [name, stream] : std::vector<std::pair<std::string, Tensor*>>{
             {"proposal_hidden", &round.normed},
             {"bridge_raw", &round.stash},
             {"bridge_message", &round.message},
             {"bridge_rms", &round.rms},
             {"block ids", &round.ids},
             {"block positions", &round.positions},
             {"head candidates", &round.candidate_ids},
             {"drafts", &round.drafts},
             {"proposal_q", &round.proposal_q}}) {
        out.parts.emplace_back(name, digest_of(*stream));
    }
    return out;
}

void print_fingers(const char* label, const Finger& base, const Finger& arm) {
    const auto parted = arm.parted_against(base);
    std::printf("ARM   %-26s %2zu of %2zu published streams MOVED", label, parted.size(),
                base.parts.size());
    for (const auto& name : parted) {
        std::printf("  %s", name.c_str());
    }
    std::printf("\n");
    std::fflush(stdout);
}

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::signal(SIGSEGV, report_fault);
    std::signal(SIGABRT, report_fault);
    std::signal(SIGFPE, report_fault);
    const auto artifact_path = setting("NINFER_TEST_ARTIFACT");
    if (artifact_path.empty()) {
        std::printf("skip: NINFER_TEST_ARTIFACT is not set\n");
        return 77;
    }
    const auto device_setting = setting("NINFER_TEST_DEVICE");
    if (device_setting.empty()) {
        std::printf("skip: NINFER_TEST_DEVICE is not set\n");
        return 77;
    }
    const int device = std::stoi(device_setting);
    int visible      = 0;
    if (cudaGetDeviceCount(&visible) != cudaSuccess || visible <= 0) {
        std::printf("skip: no CUDA device\n");
        return 77;
    }
    if (device >= visible) {
        std::printf("skip: NINFER_TEST_DEVICE=%d, %d visible\n", device, visible);
        return 77;
    }
    const auto context = std::max(std::int32_t(64), std::stoi(setting("NINFER_CTX_CONTEXT", "128")));
    const auto slack   = std::uint64_t(std::stoull(setting("NINFER_CTX_SLACK", "768"))) * 1024u
                       * 1024u;

    try {
        Probe probe(artifact_path, device);
        owner = &probe;

        // ---- the executor's own load (verbatim from the fidelity probe) ----------------------
        models::LoadOptions options;
        options.purpose       = EnginePurpose::Generation;
        options.speculative   = SpeculativeBackend::Dspine;
        options.proposal_head = ProposalHead::Full;
        const auto definition = q35::read_definition(probe.reader, options);
        const auto profile    = execution::describe_model(*definition);
        if (!definition->weights().dspine || !profile->config.dspine) {
            std::printf("STOP: the artifact carries no dspine drafter\n");
            return 1;
        }
        const auto& draft = *profile->config.dspine;
        auto plan         = q35::resolve_load(probe.reader, definition, probe.device);
        const auto [free, total] = probe.memory();
        const auto need   = plan.materialization().device_capacity_bytes;
        std::printf("load: the executor's shard needs %.5f GiB of the %.5f GiB free on the card\n",
                    need / kGiB, free / kGiB);
        std::fflush(stdout);
        if (need + slack > free) {
            std::printf("REFUSAL: the load plus the %.5f GiB headroom is wider than the card's free "
                        "memory; refused on the card's own numbers before a byte is allocated "
                        "(D-47's stop is not to be chased)\n",
                        slack / kGiB);
            return 3;
        }
        const auto began = std::chrono::steady_clock::now();
        auto model       = q35::materialize_model(std::move(plan), probe.context);
        const auto elapsed = std::chrono::duration_cast<
            std::chrono::duration<double, std::ratio<1, 1000>>>(std::chrono::steady_clock::now()
                                                                - began);
        const auto [after, unused] = probe.memory();
        (void)unused;
        const auto& stats = model->storage_stats();
        std::printf("load: %zu device objects, %.5f GiB of backing, h2d %.5f GiB in %.2f s; "
                    "device free %llu -> %llu MiB\n",
                    stats.device_object_count, stats.device_capacity_bytes / kGiB,
                    stats.h2d_bytes / kGiB, elapsed.count() / 1000.0,
                    static_cast<unsigned long long>(free / 1024 / 1024),
                    static_cast<unsigned long long>(after / 1024 / 1024));
        std::fflush(stdout);
        std::printf("bind: building execution::Parameters (the executor's DspineParameters)\n");
        std::fflush(stdout);
        const execution::Parameters parameters(*model, profile);
        if (!parameters.dspine || parameters.dspine->layers.empty()) {
            std::printf("STOP: the executor bound no dspine layers\n");
            return 1;
        }

        Geometry g;
        g.h      = dim(draft.hidden_size);
        g.w      = dim(draft.block_size);
        g.b      = kBatch;
        g.ctx    = context;
        g.c      = dim(draft.bridge.code_size);
        g.r      = dim(draft.bridge.rank);
        g.taps   = dim(draft.target_layer_ids.size());
        g.q      = dim(draft.query_width());
        g.k      = dim(draft.key_width());
        g.d      = dim(draft.head_dim);
        g.inter  = dim(draft.intermediate_size);
        g.drafts = dim(draft.block_size) - 1;
        g.rms_eps = draft.rms_norm_eps;
        g.rope_theta = draft.rope_theta;
        g.bridge_eps = draft.bridge.eps;
        g.mask_token = dim(draft.mask_token_id);
        g.qh     = dim(draft.num_attention_heads);
        g.kvh    = dim(draft.num_key_value_heads);
        std::printf("drafter: hidden=%d layers=%zu block=%d taps=%d code=%d rank=%d q=%d k=%d "
                    "heads=%d/%d head_dim=%d inter=%d mask=%d\n",
                    g.h, parameters.dspine->layers.size(), g.w, g.taps, g.c, g.r, g.q, g.k, g.qh,
                    g.kvh, g.d, g.inter, g.mask_token);
        std::printf("tap list (target_layer_ids):");
        for (const auto layer : draft.target_layer_ids) {
            std::printf(" %u", static_cast<unsigned>(layer));
        }
        std::printf("  (the tower has %llu layers)\n",
                    static_cast<unsigned long long>(profile->config.text.num_hidden_layers));
        std::printf("round: batch=%d width=%d drafts=%d context=%d\n", g.b, g.w, g.drafts, g.ctx);
        std::fflush(stdout);

        // ---- the round, its pages and its arena ---------------------------------------------
        auto round  = allocate(g.h, g.w, g.b, g.ctx, g.c, g.r, g.taps, g.q, g.k, g.d, g.inter,
                               g.drafts);
        const std::uint32_t layers = std::uint32_t(parameters.dspine->layers.size());
        DrafterContext pages(layers, g.d, g.kvh, g.ctx);
        std::printf("pages: page_tokens=%u record_bytes=%llu alignment=%llu stride=%llu, "
                    "one layer's K+V bytes in a record=%zu, logical pages=%u; context=%d tokens "
                    "span %d page(s)\n",
                    static_cast<unsigned>(pages.page_tokens_),
                    static_cast<unsigned long long>(pages.pages_),
                    static_cast<unsigned long long>(pages.alignment_),
                    static_cast<unsigned long long>(pages.stride_),
                    pages.page_bytes_of(0), pages.logical_, g.ctx,
                    (g.ctx + int(pages.page_tokens_) - 1) / int(pages.page_tokens_));
        std::fflush(stdout);
        const auto envelope = ops::ContextAttentionExecutionEnvelope{
            static_cast<std::uint32_t>(g.ctx), static_cast<std::uint32_t>(g.ctx)};
        const auto walk_bytes = execution::dspine_round_workspace_bytes(draft, g.b, g.w, envelope, 1);
        const auto topk_bytes = ops::linear_topk_workspace_capacity_bytes(
            parameters.dspine->output_head.weight.qtype, parameters.dspine->output_head.weight.n,
            parameters.dspine->output_head.weight.k, g.drafts * g.b, g.drafts * g.b);
        const auto tail_bytes = topk_bytes
            + std::size_t(g.h) * g.drafts * g.b * 2 + std::size_t(kSlots) * g.drafts * g.b * 4
            + std::size_t(kSlots) * g.drafts * g.b * 4
            + std::size_t(g.b) * kSlots * kSlots * g.drafts * 4
            + std::size_t(g.b) * g.drafts * kSlots * (4 + 4 + 4) + std::size_t(g.b) * g.drafts * 4;
        const auto arena_bytes = std::max(walk_bytes, tail_bytes) + (32u << 20);
        DeviceBuffer scratch(probe.device, arena_bytes);
        MemoryArena arena(scratch.span());
        std::printf("arena: the walk's own sizing is %zu B, the round arena is %zu B\n",
                    walk_bytes, arena_bytes);
        std::fflush(stdout);

        const auto cuda     = probe.cuda();
        const auto sampling = static_cast<const ops::SamplingConfig*>(round.configs.data);

        // One full round: the reset, the context append, the walk, the tail.
        const auto run = [&](const std::int32_t ctx, const float tap_phase,
                             const bool with_append) {
            reset_round(g, round, pages, arena, ctx, tap_phase);
            const execution::DspineExecutionView view{parameters, arena, cuda, pages.cache_.get()};
            if (with_append) {
                step("dspine_append_context", [&] {
                    execution::dspine_append_context(
                        view, round.taps.slice(2, 0, ctx).view({g.h * g.taps, ctx, g.b, 1}),
                        round.context_positions.slice(0, 0, ctx),
                        round.context_counts, round.table_rows,
                        ops::KVCacheAppendPrefixExecutionEnvelope{
                            static_cast<std::uint32_t>(ctx), static_cast<std::uint32_t>(ctx)});
                    cuda_ok(cudaGetLastError(), "dspine_append_context launch");
                });
            }
            const execution::DspineWalkInputs walk_inputs{round.anchors, round.lengths,
                                                          round.valid, round.table_rows};
            execution::DspineWalkOutputs walk_outputs{
                round.ids, round.positions, round.stash, round.rms, round.message, round.normed};
            step("dspine_walk", [&] {
                execution::dspine_walk(
                    view, walk_inputs, walk_outputs,
                    ops::ContextAttentionExecutionEnvelope{static_cast<std::uint32_t>(ctx),
                                                           static_cast<std::uint32_t>(ctx)});
                cuda_ok(cudaGetLastError(), "dspine_walk launch");
            });
            const execution::DspineTailView tail_view{
                parameters.dspine->output_head.weight, parameters.dspine->final_norm,
                parameters.dspine->gate_bias, parameters.dspine->tg, parameters.dspine->tsv, g.r};
            const execution::DspineTailInputs tail_inputs{
                round.stash, round.rms, round.message, round.normed, round.anchors,
                round.draw_positions, sampling};
            execution::DspineTailOutputs tail_outputs{
                round.drafts, round.candidate_ids, round.proposal_q};
            step("dspine_propose_tail", [&] {
                execution::dspine_propose_tail(tail_view, tail_inputs, tail_outputs, arena, cuda);
                cuda_ok(cudaGetLastError(), "dspine_propose_tail launch");
            });
        };

        // The warmup pass: the first launch of a module of a process is not a measurement.
        run(g.ctx, 0.0f, true);
        std::printf("warmup pass complete\n");
        std::fflush(stdout);

        // ---- the page-index printout ---------------------------------------------------------
        std::printf("pages: the page table row 0 the round publishes (logical -> byte offset):");
        const auto table = download_i32(round.table_rows);
        std::printf(" table_row=%d,", table[0]);
        for (std::size_t page = 0; page < pages.logical_; ++page) {
            std::printf(" p%zu->%lld", page, static_cast<long long>(pages.offsets_[page]));
        }
        std::printf("\n");
        (void)pages.page_bytes_of(0);
        std::printf("pages: a round that appends %d tokens writes logical pages 0..%d (position p "
                    "-> page p/%u) and attends context_lengths=%d, i.e. the SAME pages 0..%d\n",
                    g.ctx, (g.ctx - 1) / int(pages.page_tokens_),
                    static_cast<unsigned>(pages.page_tokens_),
                    g.ctx, (g.ctx - 1) / int(pages.page_tokens_));
        std::fflush(stdout);

        // ---- ARM BASE -------------------------------------------------------------------------
        run(g.ctx, 0.0f, true);
        std::printf("pages after the append (non-zero bytes of K+V per logical page, layers "
                    "%u):", layers);
        for (std::size_t page = 0; page < pages.logical_; ++page) {
            std::size_t total = 0;
            for (std::size_t layer = 0; layer < layers; ++layer) {
                total += pages.nonzero_bytes(layer, page);
            }
            std::printf("  p%zu=%zu", page, total);
        }
        std::printf("\n");
        std::fflush(stdout);
        const auto base = fingerprint(round);
        std::printf("ARM   %-26s the reference\n", "BASE append+walk+tail");

        // ---- ARM NO-APPEND: the pages stay empty, everything else identical -------------------
        run(g.ctx, 0.0f, false);
        print_fingers("NO-APPEND (context never fed)", base, fingerprint(round));

        // ---- ARMS PERTURB-PAGE-p: flip one page's bytes after the append ----------------------
        for (std::size_t page = 0; page < pages.logical_; ++page) {
            reset_round(g, round, pages, arena, g.ctx, 0.0f);
            const execution::DspineExecutionView view{parameters, arena, cuda, pages.cache_.get()};
            step("dspine_append_context (perturb arm)", [&] {
                execution::dspine_append_context(
                    view, round.taps.view({g.h * g.taps, g.ctx, g.b, 1}),
                    round.context_positions.slice(0, 0, g.ctx),
                    round.context_counts, round.table_rows,
                    ops::KVCacheAppendPrefixExecutionEnvelope{
                        static_cast<std::uint32_t>(g.ctx), static_cast<std::uint32_t>(g.ctx)});
                cuda_ok(cudaGetLastError(), "dspine_append_context launch");
            });
            std::size_t before = 0, touched = 0;
            for (std::size_t layer = 0; layer < layers; ++layer) {
                before += pages.nonzero_bytes(layer, page);
            }
            for (std::size_t layer = 0; layer < layers; ++layer) {
                pages.perturb(layer, page, 0x5a);
            }
            for (std::size_t layer = 0; layer < layers; ++layer) {
                touched += pages.nonzero_bytes(layer, page);
            }
            const execution::DspineWalkInputs walk_inputs{round.anchors, round.lengths,
                                                          round.valid, round.table_rows};
            execution::DspineWalkOutputs walk_outputs{
                round.ids, round.positions, round.stash, round.rms, round.message, round.normed};
            step("dspine_walk (perturb arm)", [&] {
                execution::dspine_walk(view, walk_inputs, walk_outputs,
                                       ops::ContextAttentionExecutionEnvelope{
                                           static_cast<std::uint32_t>(g.ctx),
                                           static_cast<std::uint32_t>(g.ctx)});
                cuda_ok(cudaGetLastError(), "dspine_walk launch");
            });
            const execution::DspineTailView tail_view{
                parameters.dspine->output_head.weight, parameters.dspine->final_norm,
                parameters.dspine->gate_bias, parameters.dspine->tg, parameters.dspine->tsv, g.r};
            const execution::DspineTailInputs tail_inputs{
                round.stash, round.rms, round.message, round.normed, round.anchors,
                round.draw_positions, sampling};
            execution::DspineTailOutputs tail_outputs{
                round.drafts, round.candidate_ids, round.proposal_q};
            step("dspine_propose_tail (perturb arm)", [&] {
                execution::dspine_propose_tail(tail_view, tail_inputs, tail_outputs, arena, cuda);
                cuda_ok(cudaGetLastError(), "dspine_propose_tail launch");
            });
            const auto label = "PERTURB page " + std::to_string(page);
            std::printf("       (page %zu bytes non-zero: %zu before the flip, %zu after; the flip "
                        "wrote 0x5a over page_bytes %llu x %u planes x %u layers)\n",
                        page, before, touched,
                        static_cast<unsigned long long>(pages.pages_), 2u,
                        static_cast<unsigned>(layers));
            print_fingers(label.c_str(), base, fingerprint(round));
        }

        // ---- ARM TAP-CONTROL: re-seed the tap field ONLY, everything else identical ------------
        run(g.ctx, 0.7f, true);
        print_fingers("TAP FIELD RE-SEEDED (control)", base, fingerprint(round));

        // ---- ARM SHORT CONTEXT: the same text, half the context -------------------------------
        run(g.ctx / 2, 0.0f, true);
        print_fingers("HALF THE CONTEXT", base, fingerprint(round));

        std::printf("reading: BASE is the reference. A PERTURB-page arm that moves NOTHING means "
                    "the walk does not read that page; a NO-APPEND arm that moves NOTHING means "
                    "the whole context feed is inert. The TAP-FIELD arm must move: it is the "
                    "positive control on the walk's input path.\n");
        std::fflush(stdout);
        return 0;
    } catch (const std::exception& error) {
        std::printf("STOP: %s\n", error.what());
        std::fflush(stdout);
        return 1;
    }
}
