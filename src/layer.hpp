#pragma once
// Layer interface + registry.  Adding support for a new ncnn layer type means
// implementing this interface and registering a factory (see layers.cpp).
#include "params.hpp"
#include "tensor.hpp"
#include "thread_pool.hpp"

#include <cstdint>
#include <istream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// How convolution weights are kept in RAM.  Stream keeps nothing resident: each
// layer reads its fp32 weights from the model file right before it runs.
enum class WeightFormat { F32, F16, I8, Stream };

// Sequential reader for the ncnn .bin weight file (streamed, never fully loaded).
class ModelBin {
public:
    explicit ModelBin(std::istream& in) : in_(in) {}
    // ncnn "type 0" block: 4-byte tag followed by data (fp32 or fp16).
    // Calls sink(ptr, n) with consecutive chunks of decoded fp32 values to
    // keep the transient memory bounded.
    template <class Sink>
    void read_tagged(std::size_t count, Sink&& sink);
    // ncnn "type 1" block: raw fp32 values.
    void read_raw(float* dst, std::size_t count);
    // Skips a tagged fp32 block, returning the file offset of its data
    // (or -1 if the block is not plain fp32 and must be loaded instead).
    std::streamoff skip_tagged_f32(std::size_t count);
    std::istream& stream() { return in_; }

private:
    std::istream& in_;
};

struct ExecContext {
    ThreadPool* pool = nullptr;
    std::istream* weight_file = nullptr;  // for WeightFormat::Stream
    void* workspace = nullptr;  // per-op scratch, carved out of the planned arena
    std::size_t workspace_bytes = 0;
};

struct EngineConfig {
    int threads = 4;
    WeightFormat weight_format = WeightFormat::F16;
    // Size-based mixed precision for convolution weights (counts are weights per layer).
    // Small layers (stem, early high-resolution convs, head outputs, DFL) hold little
    // memory but are the most precision-sensitive, so they stay fp32. With int8, only
    // the large layers - which hold almost all of the weights - are quantized; the
    // medium ones use fp16.
    std::size_t fp32_below = 16384;  // fp16/int8 modes: layers smaller than this stay fp32
    std::size_t int8_min = 32768;    // int8 mode: layers at least this large are int8
    bool winograd = true;            // Winograd F(2x2,3x3) for 3x3 stride-1 convolutions
};

// Resident format of one convolution layer's weights under the configured policy.
inline WeightFormat weight_format_for(const EngineConfig& cfg, std::size_t count) {
    const WeightFormat f = cfg.weight_format;
    if (f == WeightFormat::F32 || f == WeightFormat::Stream) return f;
    if (count < cfg.fp32_below) return WeightFormat::F32;
    if (f == WeightFormat::I8 && count < cfg.int8_min) return WeightFormat::F16;
    return f;
}

class Layer {
public:
    std::string type, name;
    int num_outputs = 1;  // number of top blobs (set by the loader)
    virtual ~Layer() = default;

    virtual void load_param(const ParamDict&) {}
    virtual void load_model(ModelBin&, const EngineConfig&) {}

    virtual std::vector<Shape> infer_shapes(const std::vector<Shape>& in) const = 0;
    virtual void forward(const std::vector<TensorView>& in, const std::vector<TensorView>& out,
                         ExecContext& ctx) const = 0;

    // Scratch bytes needed during forward (planned by the memory planner).
    virtual std::size_t workspace_bytes(const std::vector<Shape>& /*in*/, const std::vector<Shape>& /*out*/,
                                        int /*threads*/) const { return 0; }
    // Output `out_idx` can be a zero-copy view into input 0 at `elem_offset`.
    virtual bool view_output(const std::vector<Shape>& /*in*/, int /*out_idx*/, std::size_t& /*elem_offset*/) const {
        return false;
    }
    // Elementwise op: output 0 may overwrite input 0 in place.
    virtual bool inplace_capable() const { return false; }
    // For concatenations: input `in_idx` occupies a contiguous slot at `elem_offset`
    // of the output, so its producer can write there directly.
    virtual bool concat_slot(const std::vector<Shape>& /*in*/, int /*in_idx*/, std::size_t& /*elem_offset*/) const {
        return false;
    }
    // Constant producers (MemoryData) return their data here.
    virtual const float* constant_data() const { return nullptr; }
    // Bytes of resident parameters (weights, biases, constants).
    virtual std::size_t param_bytes() const { return 0; }
};

std::unique_ptr<Layer> create_layer(const std::string& type);

// ---- ModelBin template implementation ----
std::uint32_t modelbin_read_tag(std::istream& in);
float modelbin_half_to_float(std::uint16_t h);

template <class Sink>
void ModelBin::read_tagged(std::size_t count, Sink&& sink) {
    const std::uint32_t tag = modelbin_read_tag(in_);
    constexpr std::size_t kChunk = 4096;
    float buf[kChunk];
    if (tag == 0) {
        for (std::size_t done = 0; done < count;) {
            const std::size_t n = count - done < kChunk ? count - done : kChunk;
            read_raw(buf, n);
            sink(buf, n);
            done += n;
        }
    } else if (tag == 0x01306B47u) {  // fp16 storage
        std::uint16_t hb[kChunk];
        for (std::size_t done = 0; done < count;) {
            const std::size_t n = count - done < kChunk ? count - done : kChunk;
            in_.read(reinterpret_cast<char*>(hb), static_cast<std::streamsize>(n * 2));
            for (std::size_t i = 0; i < n; ++i) buf[i] = modelbin_half_to_float(hb[i]);
            sink(buf, n);
            done += n;
        }
        if (count % 2) in_.ignore(2);  // ncnn aligns fp16 blocks to 4 bytes
    } else {
        throw std::runtime_error("unsupported weight storage tag in .bin");
    }
    if (!in_) throw std::runtime_error("unexpected end of .bin weight file");
}
