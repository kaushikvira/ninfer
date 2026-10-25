#include "models/qwen3_5/execution/dflash_tfdump.h"

#include "core/device.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace ninfer::models::qwen3_5::execution {
namespace {

float bf16_to_float(std::uint16_t bits) {
    const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16U;
    float value;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

// .npy v1.0, little-endian, C order, one dimension.
void write_npy(const std::filesystem::path& path, std::string_view descr, std::size_t count,
               const void* data, std::size_t element_bytes) {
    std::ostringstream header;
    header << "{'descr': '" << descr << "', 'fortran_order': False, 'shape': (" << count << ",), }";
    std::string text           = header.str();
    const std::size_t unpadded = 10 + text.size() + 1;
    text.append((64 - unpadded % 64) % 64, ' ');
    text.push_back('\n');
    std::ofstream out(path, std::ios::binary);
    if (!out) { throw std::runtime_error("dflash_tfdump: cannot write " + path.string()); }
    const char magic[] = {'\x93', 'N', 'U', 'M', 'P', 'Y', '\x01', '\x00'};
    out.write(magic, sizeof(magic));
    const auto length          = static_cast<std::uint16_t>(text.size());
    const char length_bytes[2] = {static_cast<char>(length & 0xFF), static_cast<char>(length >> 8)};
    out.write(length_bytes, 2);
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(count * element_bytes));
    if (!out) { throw std::runtime_error("dflash_tfdump: short write " + path.string()); }
}

std::string position_name(std::int32_t position) {
    char name[32];
    std::snprintf(name, sizeof(name), "p%06d.npy", position);
    return name;
}

} // namespace

std::unique_ptr<DFlashTfDump> DFlashTfDump::from_environment(std::int32_t layers) {
    const char* directory = std::getenv("NINFER_DFLASH_TFDUMP");
    if (directory == nullptr || *directory == '\0') { return nullptr; }
    std::vector<bool> selected(static_cast<std::size_t>(layers), true);
    if (const char* list = std::getenv("NINFER_DFLASH_TFDUMP_LAYERS")) {
        const std::string_view text(list);
        if (text != "all") {
            selected.assign(selected.size(), false);
            if (text != "none") {
                std::stringstream stream{std::string(text)};
                std::string item;
                while (std::getline(stream, item, ',')) {
                    const int layer = std::stoi(item);
                    if (layer < 0 || layer >= layers) {
                        throw std::invalid_argument("NINFER_DFLASH_TFDUMP_LAYERS out of range");
                    }
                    selected[static_cast<std::size_t>(layer)] = true;
                }
            }
        }
    }
    const char* logits = std::getenv("NINFER_DFLASH_TFDUMP_LOGITS");
    bool compact       = true;
    if (const char* c = std::getenv("NINFER_DFLASH_TFDUMP_COMPACT")) {
        compact = std::string_view(c) != "0";
    }
    std::int32_t topk = 0;
    if (const char* t = std::getenv("NINFER_DFLASH_TFDUMP_TOPK")) {
        topk = std::max(0, std::atoi(t));
    }
    return std::make_unique<DFlashTfDump>(directory, std::move(selected),
                                          logits == nullptr || std::string_view(logits) != "0",
                                          compact, topk);
}

DFlashTfDump::DFlashTfDump(std::filesystem::path directory, std::vector<bool> layers, bool logits,
                           bool compact, std::int32_t topk)
    : directory_(std::move(directory)), layers_(std::move(layers)), logits_(logits),
      compact_(compact), topk_(topk) {
    std::filesystem::create_directories(directory_);
}

void DFlashTfDump::capture_layer(std::int32_t layer, const Tensor& value, std::int32_t first_position,
                                 cudaStream_t stream) {
    if (layer < 0 || static_cast<std::size_t>(layer) >= layers_.size() ||
        !layers_[static_cast<std::size_t>(layer)]) {
        return;
    }
    if (value.dtype != DType::BF16 || !value.is_contiguous()) {
        throw std::invalid_argument("dflash_tfdump: taps take contiguous BF16 [rows, T]");
    }
    const auto rows    = static_cast<std::size_t>(value.ne[0]);
    const auto columns = static_cast<std::size_t>(value.numel()) / rows;
    scratch_.resize(rows * columns);
    CUDA_CHECK(cudaMemcpyAsync(scratch_.data(), value.data, value.bytes(), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    char name[16];
    std::snprintf(name, sizeof(name), "L%02d", layer);
    if (compact_) {
        std::ofstream out(directory_ / ("hidden_" + std::string(name) + ".bin"),
                          std::ios::binary | std::ios::app);
        if (!out) { throw std::runtime_error("dflash_tfdump: cannot open compact hidden tap"); }
        const std::int32_t head[3] = {first_position, static_cast<std::int32_t>(columns),
                                      static_cast<std::int32_t>(rows)};
        out.write(reinterpret_cast<const char*>(head), sizeof(head));
        out.write(reinterpret_cast<const char*>(scratch_.data()),
                  static_cast<std::streamsize>(scratch_.size() * sizeof(std::uint16_t)));
        if (!out) { throw std::runtime_error("dflash_tfdump: short write of a compact hidden tap"); }
        return;
    }
    const auto directory = directory_ / "hidden" / name;
    std::filesystem::create_directories(directory);
    std::vector<float> column(rows);
    for (std::size_t c = 0; c < columns; ++c) {
        const std::int32_t position =
            c < positions_.size() ? positions_[static_cast<std::size_t>(c)]
                                  : first_position + static_cast<std::int32_t>(c);
        for (std::size_t r = 0; r < rows; ++r) {
            column[r] = bf16_to_float(scratch_[c * rows + r]);
        }
        write_npy(directory / position_name(position), "<f4", rows, column.data(), sizeof(float));
    }
}

void DFlashTfDump::capture_positions(const Tensor& positions, cudaStream_t stream) {
    if (positions.dtype != DType::I32) {
        throw std::invalid_argument("dflash_tfdump: positions are int32 [T]");
    }
    const auto count = static_cast<std::size_t>(positions.numel());
    positions_.resize(count);
    CUDA_CHECK(cudaMemcpyAsync(positions_.data(), positions.data,
                               static_cast<std::size_t>(count) * sizeof(std::int32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<std::int32_t> host(positions_.begin(), positions_.end());
    write_npy(directory_ / "positions.npy", "<i4", host.size(), host.data(), sizeof(std::int32_t));
}

void DFlashTfDump::write_tokens(std::span<const std::int32_t> tokens) {
    std::vector<std::int64_t> wide(tokens.begin(), tokens.end());
    write_npy(directory_ / "tokens.npy", "<i8", wide.size(), wide.data(), sizeof(std::int64_t));
}

void DFlashTfDump::write_logits(std::span<const std::uint16_t> logits, std::int32_t rows,
                                std::int32_t cols, std::span<const std::int32_t> positions) const {
    if (!logits_) { return; }
    const auto r = static_cast<std::size_t>(rows);
    const auto t = static_cast<std::size_t>(cols);
    if (logits.size() < r * t) {
        throw std::invalid_argument("dflash_tfdump: logits do not cover their positions");
    }
    if (topk_ > 0) {
        std::ofstream out(directory_ / "topk.bin", std::ios::binary | std::ios::app);
        if (!out) { throw std::runtime_error("dflash_tfdump: cannot open topk.bin"); }
        const auto k = static_cast<std::size_t>(std::min<std::int32_t>(topk_, rows));
        std::vector<float> column(r);
        std::vector<std::int32_t> order(r);
        std::vector<float> lp(k);
        for (std::size_t c = 0; c < positions.size(); ++c) {
            float top = -INFINITY;
            for (std::size_t i = 0; i < r; ++i) {
                column[i] = bf16_to_float(logits[i * t + c]);
                top       = std::max(top, column[i]);
            }
            double sum = 0;
            for (std::size_t i = 0; i < r; ++i) {
                sum += std::exp(static_cast<double>(column[i] - top));
            }
            const float lse = top + static_cast<float>(std::log(sum));
            for (std::size_t i = 0; i < r; ++i) { order[i] = static_cast<std::int32_t>(i); }
            std::partial_sort(
                order.begin(), order.begin() + static_cast<std::ptrdiff_t>(k), order.end(),
                [&](std::int32_t a, std::int32_t b) { return column[a] > column[b]; });
            for (std::size_t j = 0; j < k; ++j) { lp[j] = column[order[j]] - lse; }
            out.write(reinterpret_cast<const char*>(&positions[c]), sizeof(std::int32_t));
            out.write(reinterpret_cast<const char*>(order.data()),
                      static_cast<std::streamsize>(k * sizeof(std::int32_t)));
            out.write(reinterpret_cast<const char*>(lp.data()),
                      static_cast<std::streamsize>(k * sizeof(float)));
        }
        if (!out) { throw std::runtime_error("dflash_tfdump: short write of the top-k logits"); }
        return;
    }
    const auto directory = directory_ / "logits";
    std::filesystem::create_directories(directory);
    std::vector<float> column(r);
    for (std::size_t c = 0; c < positions.size(); ++c) {
        for (std::size_t i = 0; i < r; ++i) { column[i] = bf16_to_float(logits[i * t + c]); }
        write_npy(directory / position_name(positions[c]), "<f4", r, column.data(), sizeof(float));
    }
}

void DFlashTfDump::write_meta(const std::string& producer, const std::string& artifact) const {
    std::ofstream out(directory_ / "meta.json");
    out << "{\"format\":\"tfdump/1\",\"producer\":\"" << producer << "\",\"artifact\":\""
        << artifact << "\",\"logits\":\"bf16 output head widened to f32\"}\n";
}

} // namespace ninfer::models::qwen3_5::execution
