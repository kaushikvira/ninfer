// P2 NVFP4 parity fixture: calls the REAL compiled DSpine bridge ops
// (build/rtx5090-ops/lib/libninfer_ops.so of this tree) on one CUDA device and dumps raw
// output bytes so a Python reference can compare them.
//
// Superset of tests/dspine_parity/fixture/dspine_fixture.cpp (P0b), extended with
//   * --nvtbl=NAME:CODES:SCALES:DIVISOR  (a block-scaled NVFP4 gather table), and
//   * --op=serial_tail                   (the sampled tail op, with the request policy).
// The P0b fixture is left untouched: the BF16 tables are still what P0b verified, and the
// whole point here is to drive the SAME ops through their NVFP4 gather branch.
//
// Inputs are raw little-endian binary files produced by PyTorch (bf16 = uint16 view,
// fp32, i32). Nothing here computes math: rounding is 100 % the shipped kernels.
//
// usage: dspine_nvfp4_fixture --op=NAME [--dev=N] [--out=DIR] [--eps=F] [--stash=0|1]
//                             [--g K=V]... [--in=NAME:PATH]...
//                             [--tbl=NAME:PATH] [--nvtbl=NAME:CODES:SCALES:DIVISOR]...
//                             [--fp8=NAME:CODES:SCALES]
//                             [--temp=F] [--topk=N] [--topp=F] [--minp=F] [--seed=N]
// geometry keys: H (hidden) R (rank) C (code size) W (width) B (batch) V (vocab)
// D (drafts) defaults to W-1; columns = W*B.
//
// --nvtbl uploads a NVFP4 table exactly as the artifact loader does: qdata = the packed
// E2M1 code plane [n, k/2], scales = the swizzled (block_scale_k16_m128x4_v1) E4M3 plane,
// weight_scale = the divisor the dequant divides by, scale_dtype = FP8_E4M3FN.
//
// Outputs are written to <out>/<name>.bin (device bytes, little-endian).

#include "ninfer/ops/dspine_bridge.h"

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace ninfer;
using ninfer::ops::dspine_finish;
using ninfer::ops::dspine_message;
using ninfer::ops::dspine_normalize_codes;
using ninfer::ops::dspine_prepare_read;
using ninfer::ops::dspine_serial_chain;
using ninfer::ops::dspine_serial_scores;
using ninfer::ops::dspine_serial_tail;

namespace {

[[noreturn]] void die(const std::string& message) {
    std::fprintf(stderr, "fixture: %s\n", message.c_str());
    std::exit(3);
}

void cuda_ok(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        die(std::string(what) + ": " + cudaGetErrorString(error));
    }
}

std::int32_t i32(std::int64_t value) {
    return static_cast<std::int32_t>(value);
}

std::int64_t dtype_bytes(DType dtype) {
    switch (dtype) {
    case DType::BF16: return 2;
    case DType::FP32: return 4;
    case DType::I32: return 4;
    case DType::U8: return 1;
    default: die("unsupported dtype in fixture");
    }
}

struct Dev {
    std::vector<void*> pointers;
    std::vector<std::size_t> sizes;

    ~Dev() {
        for (void* pointer : pointers) { cudaFree(pointer); }
    }
};

// A device buffer filled from a binary file. `expect` is the byte count the op geometry
// requires; a size mismatch is a hard error (a truncated vector would silently pass).
// Reads a file and checks it has exactly `expect` bytes (a truncated vector would
// otherwise let a partial table silently pass).
std::vector<char> read_file(const std::string& path, std::size_t expect) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { die("cannot open input " + path); }
    std::vector<char> host(static_cast<std::size_t>(std::filesystem::file_size(path)));
    in.read(host.data(), static_cast<std::int64_t>(host.size()));
    if (static_cast<std::size_t>(in.gcount()) != host.size()) { die("short read on " + path); }
    if (host.size() != expect) {
        die("input " + path + " is " + std::to_string(host.size()) + " bytes, geometry wants " +
            std::to_string(expect));
    }
    return host;
}

// A device buffer filled from bytes read by read_file.
void* upload_bytes(Dev& dev, std::span<const char> host) {
    void* device = nullptr;
    const std::size_t bytes = host.size();
    cuda_ok(cudaMalloc(&device, bytes == 0 ? 1 : bytes), "cudaMalloc");
    cuda_ok(cudaMemcpy(device, host.data(), bytes, cudaMemcpyHostToDevice), "copyin");
    dev.pointers.push_back(device);
    return device;
}

void* upload(Dev& dev, const std::string& path, std::size_t expect) {
    return upload_bytes(dev, read_file(path, expect));
}

void download(const std::string& dir, const std::string& name, const void* device,
               std::size_t bytes) {
    std::vector<char> host(bytes);
    cuda_ok(cudaMemcpy(host.data(), device, bytes, cudaMemcpyDeviceToHost), "copyout");
    const std::string path = dir + "/" + name + ".bin";
    std::ofstream out(path, std::ios::binary);
    if (!out) { die("cannot write " + path); }
    out.write(host.data(), static_cast<std::int64_t>(bytes));
    if (!out) { die("short write on " + path); }
}

struct Args {
    std::string op;
    std::string out = ".";
    int device = 1;
    float eps = 1e-6f;
    bool stash = false;
    std::map<std::string, std::int64_t> geo;
    std::vector<std::pair<std::string, std::string>> inputs;
    std::vector<std::pair<std::string, std::string>> tables;
    std::vector<std::tuple<std::string, std::string, std::string, float>> nvtables;
    std::vector<std::tuple<std::string, std::string, std::string>> fp8s;

    // serial_tail request policy (K1 = T 1.0 / top_k 20 / top_p 1.0 / min_p 0).
    float temperature = 1.0f;
    std::int32_t top_k = 20;
    float top_p = 1.0f;
    float min_p = 0.0f;
    unsigned long long seed = 0;

    std::int64_t g(const std::string& key) const {
        const auto found = geo.find(key);
        if (found == geo.end()) { die("missing geometry --g " + key); }
        return found->second;
    }
    std::string path(const std::string& key) const {
        for (const auto& [name, file] : inputs) {
            if (name == key) { return file; }
        }
        die("missing --in " + key);
        return {};
    }
};

struct Ten {
    void* data = nullptr;
    DType dtype = DType::BF16;
    std::vector<std::int32_t> shape;
    std::size_t bytes = 0;
};

struct Builder {
    explicit Builder(const Args& arguments) : args(arguments), device(arguments.device) {}

    const Args& args;
    int device = 1;
    Dev dev;

    Ten make(const std::string& key, DType dtype, std::vector<std::int32_t> shape) {
        std::int64_t elements = 1;
        for (std::int32_t dim : shape) { elements *= dim; }
        Ten ten;
        ten.dtype = dtype;
        ten.shape = std::move(shape);
        ten.bytes = static_cast<std::size_t>(elements) * static_cast<std::size_t>(dtype_bytes(dtype));
        ten.data = upload(dev, args.path(key), ten.bytes);
        return ten;
    }

    Tensor tensor(Ten& ten) {
        std::int32_t shape[4] = {1, 1, 1, 1};
        for (std::size_t i = 0; i < ten.shape.size(); ++i) { shape[i] = ten.shape[i]; }
        return Tensor(ten.data, Device::cuda(device), ten.dtype,
                       {shape[0], shape[1], shape[2], shape[3]});
    }

    Tensor out_tensor(const std::string& name, DType dtype, std::vector<std::int32_t> shape) {
        // Output buffers are zero-filled device allocations of the exact geometry.
        std::int64_t elements = 1;
        for (std::int32_t dim : shape) { elements *= dim; }
        const std::size_t bytes =
            static_cast<std::size_t>(elements) * static_cast<std::size_t>(dtype_bytes(dtype));
        void* data = nullptr;
        cuda_ok(cudaMalloc(&data, bytes), "cudaMalloc out");
        cuda_ok(cudaMemset(data, 0, bytes), "cudaMemset out");
        dev.pointers.push_back(data);
        outputs.emplace_back(name, data, bytes);
        std::int32_t dims[4] = {1, 1, 1, 1};
        for (std::size_t i = 0; i < shape.size(); ++i) { dims[i] = shape[i]; }
        return Tensor(data, Device::cuda(device), dtype, {dims[0], dims[1], dims[2], dims[3]});
    }

    struct OutSpec {
        std::string name;
        void* data;
        std::size_t bytes;
    };
    std::vector<OutSpec> outputs;

    Weight bf16_table(const std::string& key, std::int32_t n, std::int32_t k) {
        for (const auto& [name, file] : args.tables) {
            if (name != key) { continue; }
            const std::size_t bytes = static_cast<std::size_t>(n) * static_cast<std::size_t>(k) * 2;
            Weight weight{};
            weight.device = Device::cuda(device);
            weight.qtype = QType::BF16;
            weight.qdata = upload(dev, file, bytes);
            weight.n = n;
            weight.k = k;
            weight.layout = QuantLayout::RowSplit;
            weight.scale_dtype = DType::BF16;
            return weight;
        }
        die("missing --tbl " + key);
        return {};
    }

    // The scale plane of a block_scale_k16_m128x4_v1 matrix (core/weight_view.cpp
    // weight_scale_offset): one 512-byte tile per (128 rows, 64 columns).
    static std::size_t scale_plane_bytes(std::int32_t n, std::int32_t k) {
        if (n % 128 != 0 || k % 64 != 0) {
            die("NVFP4 gather tables need n % 128 == 0 and k % 64 == 0");
        }
        return static_cast<std::size_t>(n / 128) * static_cast<std::size_t>(k / 64) * 512;
    }

    // A gather table the op dequantizes on access: the packed E2M1 code plane plus the
    // swizzled E4M3 scale plane and the parent divisor (Weight::weight_scale).
    // Built the way the artifact loader materializes one, because that is the shape the op
    // itself gates on (ops/common/host/operands.cpp:validate_nvfp4_weight): one payload with
    // the code plane at 0 and the swizzled scale plane at align_up(code_bytes, 256), planes
    // inside it and non-overlapping, qhigh absent, and both scales finite and positive.
    Weight nvfp4_table(const std::string& key, std::int32_t n, std::int32_t k) {
        for (const auto& [name, codes, scales, divisor] : args.nvtables) {
            if (name != key) { continue; }
            const std::size_t code_bytes = static_cast<std::size_t>(n) * (k / 2);
            const std::size_t scale_bytes = scale_plane_bytes(n, k);
            const std::size_t scale_offset = ((code_bytes + 255) / 256) * 256;
            const std::size_t payload_bytes = scale_offset + scale_bytes + sizeof(float);

            std::vector<char> host(payload_bytes, '\0');
            const std::vector<char> code_plane = read_file(codes, code_bytes);
            const std::vector<char> scale_plane = read_file(scales, scale_bytes);
            std::copy(code_plane.begin(), code_plane.end(), host.begin());
            std::copy(scale_plane.begin(), scale_plane.end(), host.begin() + scale_offset);
            char* base = static_cast<char*>(upload_bytes(dev, host));

            Weight weight{};
            weight.device = Device::cuda(device);
            weight.qtype = QType::NVFP4;
            weight.layout = QuantLayout::BlockScaleK16M128x4;
            weight.group_size = 16;
            weight.group = 16;
            weight.ndim = 2;
            weight.shape[0] = n;
            weight.shape[1] = k;
            weight.padded_shape[0] = n;
            weight.padded_shape[1] = k;
            weight.payload = base;
            weight.payload_bytes = payload_bytes;
            weight.qdata = base;
            weight.scales = base + scale_offset;
            weight.qhigh = nullptr;
            weight.high_plane_bytes = 0;
            weight.scale_dtype = DType::FP8_E4M3FN;
            if (!(divisor > 0.0F)) {
                die("--nvtbl " + key + ": divisor parsed to " + std::to_string(divisor)
                    + "; it must be a positive decimal (the matrix scalar)");
            }
            weight.weight_scale = divisor;
            weight.input_scale = 1.0F;  // the NVFP4 gate requires a positive input scale
            weight.n = n;
            weight.k = k;
            return weight;
        }
        die("missing --nvtbl " + key);
        return {};
    }

    // Whichever table flavour the driver supplied: --tbl (BF16) or --nvtbl (NVFP4). The op
    // picks its dequant branch off Weight::qtype, so this is the whole format switch.
    Weight gather_table(const std::string& key, std::int32_t n, std::int32_t k) {
        for (const auto& [name, file] : args.tables) {
            if (name == key) { return bf16_table(key, n, k); }
        }
        return nvfp4_table(key, n, k);
    }

    Weight fp8_row(const std::string& key, std::int32_t n, std::int32_t k) {
        for (const auto& [name, codes, scales] : args.fp8s) {
            if (name != key) { continue; }
            Weight weight{};
            weight.device = Device::cuda(device);
            weight.qtype = QType::FP8_E4M3FN_ROW_BF16;
            weight.layout = QuantLayout::RowScale;
            weight.qdata = upload(dev, codes, static_cast<std::size_t>(n) * k);
            weight.scales = upload(dev, scales, static_cast<std::size_t>(n) * 2);
            weight.scale_dtype = DType::BF16;
            weight.n = n;
            weight.k = k;
            return weight;
        }
        die("missing --fp8 " + key);
        return {};
    }

};

std::vector<std::string> split(const std::string& text, char separator) {
    std::vector<std::string> parts;
    std::size_t start = 0;
    while (true) {
        const std::size_t next = text.find(separator, start);
        parts.push_back(text.substr(start, next == std::string::npos ? std::string::npos
                                                                    : next - start));
        if (next == std::string::npos) { return parts; }
        start = next + 1;
    }
}

Args parse(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string argument(argv[i]);
        const auto value = [&](const std::string& prefix) {
            return argument.substr(prefix.size());
        };
        if (argument.rfind("--op=", 0) == 0) {
            args.op = value("--op=");
        } else if (argument.rfind("--out=", 0) == 0) {
            args.out = value("--out=");
        } else if (argument.rfind("--dev=", 0) == 0) {
            args.device = std::stoi(value("--dev="));
        } else if (argument.rfind("--eps=", 0) == 0) {
            args.eps = std::stof(value("--eps="));
        } else if (argument.rfind("--stash=", 0) == 0) {
            args.stash = value("--stash=") == "1";
        } else if (argument.rfind("--g=", 0) == 0) {
            const auto body = split(value("--g="), '=');
            args.geo[body.at(0)] = std::stoll(body.at(1));
        } else if (argument.rfind("--in=", 0) == 0) {
            const auto body = split(value("--in="), ':');
            args.inputs.emplace_back(body.at(0), body.at(1));
        } else if (argument.rfind("--tbl=", 0) == 0) {
            const auto body = split(value("--tbl="), ':');
            args.tables.emplace_back(body.at(0), body.at(1));
        } else if (argument.rfind("--fp8=", 0) == 0) {
            const auto body = split(value("--fp8="), ':');
            args.fp8s.emplace_back(body.at(0), body.at(1), body.at(2));
        } else if (argument.rfind("--nvtbl=", 0) == 0) {
            const auto body = split(value("--nvtbl="), ':');
            if (body.size() != 4) { die("--nvtbl wants NAME:CODES:SCALES:DIVISOR"); }
            args.nvtables.emplace_back(body.at(0), body.at(1), body.at(2),
                                       std::strtof(body.at(3).c_str(), nullptr));
        } else if (argument.rfind("--temp=", 0) == 0) {
            args.temperature = std::strtof(value("--temp=").c_str(), nullptr);
        } else if (argument.rfind("--topk=", 0) == 0) {
            args.top_k = std::stoi(value("--topk="));
        } else if (argument.rfind("--topp=", 0) == 0) {
            args.top_p = std::strtof(value("--topp=").c_str(), nullptr);
        } else if (argument.rfind("--minp=", 0) == 0) {
            args.min_p = std::strtof(value("--minp=").c_str(), nullptr);
        } else if (argument.rfind("--seed=", 0) == 0) {
            args.seed = std::stoull(value("--seed="));
        } else {
            die("unknown argument " + argument);
        }
    }
    if (args.op.empty()) { die("--op is required"); }
    return args;
}

} // namespace

int main(int argc, char** argv) {
    const Args args = parse(argc, argv);
    cuda_ok(cudaSetDevice(args.device), "cudaSetDevice");
    // Every kernel launch is on the default stream of the selected device.
    const cudaStream_t stream = nullptr;

    const std::int64_t H = args.geo.count("H") ? args.g("H") : 0;
    const std::int64_t R = args.geo.count("R") ? args.g("R") : 0;
    const std::int64_t C = args.geo.count("C") ? args.g("C") : 0;
    const std::int64_t W = args.geo.count("W") ? args.g("W") : 0;
    const std::int64_t B = args.geo.count("B") ? args.g("B") : 1;
    const std::int64_t V = args.geo.count("V") ? args.g("V") : 0;
    const std::int64_t D = args.geo.count("D") ? args.g("D") : W - 1;
    const std::int64_t cols = W * B;

    Builder builder(args);
    Dev& dev = builder.dev;

    auto run = [&]() {
        // 1) prepare_read: residual [H, cols] -> normalized [H, cols], rms [cols]
        if (args.op == "prepare_read") {
            Ten residual = builder.make("residual", DType::BF16, {i32(H), i32(cols)});
            auto normalized = builder.out_tensor("normalized", DType::BF16, {i32(H), i32(cols)});
            auto rms = builder.out_tensor("rms", DType::FP32, {i32(cols)});
            dspine_prepare_read(builder.tensor(residual), args.eps, normalized, rms, stream);
        }
        // 2) normalize_codes: projected [C, cols] -> codes [C, cols]
        else if (args.op == "normalize_codes") {
            Ten projected = builder.make("projected", DType::BF16, {i32(C), i32(cols)});
            Ten anchors = builder.make("anchors", DType::I32, {i32(B)});
            Weight codebook = builder.gather_table("codebook", i32(V), i32(C));
            auto codes = builder.out_tensor("codes", DType::BF16, {i32(C), i32(cols)});
            dspine_normalize_codes(builder.tensor(projected), builder.tensor(anchors), codebook,
                                   i32(W), args.eps, codes, stream);
        }
        // 3) message: packed [3R, W, B] -> message [R, W, B]
        else if (args.op == "message") {
            Ten packed = builder.make("packed", DType::BF16, {i32(3 * R), i32(W), i32(B)});
            Ten gate_bias = builder.make("gate_bias", DType::BF16, {i32(R)});
            auto message = builder.out_tensor("message", DType::BF16, {i32(R), i32(W), i32(B)});
            dspine_message(builder.tensor(packed), builder.tensor(gate_bias), message, stream);
        }
        // 4) finish: residual [H, W, B] in/out, + normed, + optional stash
        else if (args.op == "finish") {
            Ten residual = builder.make("residual", DType::BF16, {i32(H), i32(W), i32(B)});
            Ten rms = builder.make("rms", DType::FP32, {i32(cols)});
            Ten delta = builder.make("delta", DType::BF16, {i32(H), i32(W), i32(B)});
            Ten next_norm = builder.make("next_norm", DType::BF16, {i32(H)});
            auto residual_out =
                builder.out_tensor("residual_out", DType::BF16, {i32(H), i32(W), i32(B)});
            auto normed = builder.out_tensor("normed", DType::BF16, {i32(H), i32(W), i32(B)});
            Tensor stash;
            Tensor* stash_ptr = nullptr;
            if (args.stash) {
                void* data = nullptr;
                cuda_ok(cudaMalloc(&data, static_cast<std::size_t>(H * cols) * 2), "stash alloc");
                cuda_ok(cudaMemset(data, 0, static_cast<std::size_t>(H * cols) * 2), "stash set");
                dev.pointers.push_back(data);
                builder.outputs.emplace_back(Builder::OutSpec{"stash", data,
                                                              static_cast<std::size_t>(H * cols) * 2});
                stash = Tensor(data, Device::cuda(args.device), DType::BF16,
                               {i32(H), i32(W), i32(B), i32(1)});
                stash_ptr = &stash;
            }
            // The op refines the residual stream in place.
            Tensor residual_in = builder.tensor(residual);
            dspine_finish(residual_in, builder.tensor(rms), builder.tensor(delta),
                          builder.tensor(next_norm), args.eps, normed, stash_ptr, stream);
            // residual tensor is the input buffer; copy it out to residual_out.
            cuda_ok(cudaMemcpy(residual_out.data, residual.data, residual.bytes,
                               cudaMemcpyDeviceToDevice),
                    "copy refined residual out");
        }
        // 5) serial_scores: -> idx [16, D, B]
        else if (args.op == "serial_scores") {
            Ten cand = builder.make("cand", DType::I32, {16, i32(D), i32(B)});
            Ten raw = builder.make("raw", DType::BF16, {i32(H), i32(W), i32(B)});
            Ten rms = builder.make("rms", DType::FP32, {i32(W), i32(B)});
            Ten packed_q = builder.make("packed_q", DType::BF16, {i32(R), i32(W), i32(B)});
            Ten anchors = builder.make("anchors", DType::I32, {i32(B)});
            Ten norm_w = builder.make("norm_w", DType::BF16, {i32(H)});
            Ten gate_bias = builder.make("gate_bias", DType::BF16, {i32(R)});
            Weight head = builder.fp8_row("head", i32(V), i32(H));
            Weight tg = builder.gather_table("tg", i32(V), i32(R));
            Weight tsv = builder.gather_table("tsv", i32(V), i32(2 * R));
            auto idx = builder.out_tensor("idx", DType::I32, {16, i32(D), i32(B)});
            dspine_serial_scores(builder.tensor(cand), builder.tensor(raw), builder.tensor(rms),
                                 builder.tensor(packed_q), builder.tensor(anchors),
                                 builder.tensor(norm_w), builder.tensor(gate_bias), head, tg, tsv,
                                 i32(W), idx, stream);
        }
        // 5b) serial_tail: the sampled sibling. table FP32 [B, 16, 16, D] (candidate, sender,
        //     slot, row; first axis fastest), cand I32 [B, D, 16], positions I32 [B].
        //     The request policy comes from --temp/--topk/--topp/--minp/--seed.
        else if (args.op == "serial_tail") {
            Ten table = builder.make("table", DType::FP32, {i32(B), 16, 16, i32(D)});
            Ten cand = builder.make("cand", DType::I32, {i32(B), i32(D), 16});
            Ten positions = builder.make("positions", DType::I32, {i32(B)});
            auto drafts = builder.out_tensor("drafts", DType::I32, {i32(B), i32(D), 1, 1});
            auto proposal_q = builder.out_tensor("proposal_q", DType::FP32,
                                                {i32(B), i32(D), 16, 1});
            auto idx = builder.out_tensor("idx", DType::I32, {i32(B), i32(D), 16, 1});

            std::vector<ops::SamplingConfig> configs(static_cast<std::size_t>(B));
            for (ops::SamplingConfig& config : configs) {
                config.temperature = args.temperature;
                config.top_k = args.top_k;
                config.top_p = args.top_p;
                config.min_p = args.min_p;
                config.seed = args.seed;
            }
            void* config_memory = nullptr;
            const std::size_t config_bytes = configs.size() * sizeof(ops::SamplingConfig);
            cuda_ok(cudaMalloc(&config_memory, config_bytes), "configs alloc");
            cuda_ok(cudaMemcpy(config_memory, configs.data(), config_bytes,
                               cudaMemcpyHostToDevice),
                    "configs copyin");
            dev.pointers.push_back(config_memory);

            dspine_serial_tail(builder.tensor(table), builder.tensor(cand),
                               static_cast<const ops::SamplingConfig*>(config_memory),
                               builder.tensor(positions), drafts, proposal_q, idx, stream);
        }
        // 6) serial_chain: cand/idx [16, D, B] -> drafts [D, B]
        else if (args.op == "serial_chain") {
            Ten cand = builder.make("cand", DType::I32, {16, i32(D), i32(B)});
            Ten idx = builder.make("idx", DType::I32, {16, i32(D), i32(B)});
            auto drafts = builder.out_tensor("drafts", DType::I32, {i32(D), i32(B)});
            dspine_serial_chain(builder.tensor(cand), builder.tensor(idx), drafts, stream);
        } else {
            die("unknown --op " + args.op);
        }
    };

    try {
        run();
    } catch (const std::exception& error) {
        die(std::string("op threw: ") + error.what());
    }

    cuda_ok(cudaGetLastError(), "kernel launch");
    cuda_ok(cudaDeviceSynchronize(), "device sync");

    for (const auto& output : builder.outputs) {
        download(args.out, output.name, output.data, output.bytes);
    }
    std::printf("ok op=%s device=%d outputs=%zu\n", args.op.c_str(), args.device,
                builder.outputs.size());
    return 0;
}
