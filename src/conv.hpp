#pragma once
#include "layer.hpp"

#include <cstdint>
#include <vector>

// Planar (CHW) row addressing.  ring > 0 turns the plane into a ring buffer of
// `ring` rows per channel (row y lives at slot y % ring) - used by streamed
// convolution chains so that large intermediates are never materialized.
struct Plane {
    void* base = nullptr;
    DType dtype = DType::F32;
    float scale = 1.f;  // uint8 dequantization scale
    int C = 0, H = 0, W = 0;
    std::size_t cstride = 0;  // elements between channels
    int ring = 0;
    std::size_t row_offset(int y) const { return static_cast<std::size_t>(ring ? y % ring : y) * static_cast<std::size_t>(W); }
    static Plane dense(const TensorView& t) {
        Plane pl;
        pl.base = t.data; pl.dtype = t.dtype; pl.scale = t.scale;
        pl.C = t.shape.c(); pl.H = t.shape.h(); pl.W = t.shape.w();
        pl.cstride = static_cast<std::size_t>(pl.H) * static_cast<std::size_t>(pl.W);
        return pl;
    }
};

enum class Activation { None = 0, ReLU, LeakyReLU, Sigmoid, Swish };

// Convolution weights packed as [M/MR][K][MR] (GEMM A-panels), stored in the
// configured resident format and decoded to fp32 on the fly.
class PackedWeights {
public:
    static constexpr int MR = 8;
    static int int8_group;  // K-group size for int8 scales (per output channel)
    WeightFormat format = WeightFormat::F32;
    int M = 0, K = 0;
    int blocks() const { return (M + MR - 1) / MR; }
    std::size_t padded_count() const { return static_cast<std::size_t>(blocks()) * MR * static_cast<std::size_t>(K); }
    void init(int m, int k, WeightFormat f);
    // Stream format: weights stay in the model file at `offset` ([oc][k] fp32).
    void init_stream(int m, int k, std::streamoff offset);
    // Reads + packs streamed weights into dst (staging needs MR*K floats).
    void read_stream(std::istream& file, float* dst, float* staging) const;
    // write one output channel (K values, in [ic][ky][kx] order)
    void set_row(int oc, const float* w);
    void finalize();
    // decode block range to fp32 packed layout (dst sized padded_count()).
    void decode_blocks(float* dst, int b0, int b1) const;
    // block b, reduction indices [k0, k1), all MR lanes, as fp32 in [k][MR] order
    void decode_range(float* dst, int b, int k0, int k1) const;
    const float* f32() const { return f32_.data(); }
    std::size_t bytes() const;

private:
    std::vector<float> f32_;
    std::vector<std::uint16_t> f16_;
    std::vector<std::int8_t> i8_;
    std::vector<float> scale_;
    int groups_ = 1;
    std::streamoff offset_ = -1;
};

class ConvolutionLayer : public Layer {
public:
    static constexpr int MR = PackedWeights::MR;
    static constexpr int NR = 8;

    int num_output = 0, in_channels = 0;
    int kw = 1, kh = 1, dw = 1, dh = 1, sw = 1, sh = 1, pl = 0, pr = 0, pt = 0, pb = 0;
    int bias_term = 0, weight_data_size = 0;
    Activation act = Activation::None;
    float act_alpha = 0.f;
    bool has_residual = false;  // fused "+ input[1]" after activation
    bool winograd = false;      // 3x3 stride-1 pad-1: Winograd F(2x2,3x3) instead of im2col
    PackedWeights weights;
    std::vector<float> bias;

    void load_param(const ParamDict& pd) override;
    void load_model(ModelBin& mb, const EngineConfig& cfg) override;
    std::vector<Shape> infer_shapes(const std::vector<Shape>& in) const override;
    std::size_t workspace_bytes(const std::vector<Shape>& in, const std::vector<Shape>& out, int threads) const override;
    void forward(const std::vector<TensorView>& in, const std::vector<TensorView>& out, ExecContext& ctx) const override;
    std::size_t param_bytes() const override { return weights.bytes() + bias.size() * sizeof(float); }

    // ---- building blocks shared with the streamed chain layer ----
    int K() const { return in_channels * kh * kw; }
    int out_h(int in_h) const { return (in_h + pt + pb - dh * (kh - 1) - 1) / sh + 1; }
    int out_w(int in_w) const { return (in_w + pl + pr - dw * (kw - 1) - 1) / sw + 1; }
    std::size_t decoded_weight_bytes() const;
    std::size_t winograd_weight_count() const;
    std::size_t thread_workspace_bytes() const;
    // returns fp32 packed weights (decoding/reading into `scratch` if not resident fp32)
    const float* prepare_weights(float* scratch, ExecContext& ctx) const;
    // computes output rows [oy0, oy1)
    void run_rows(const float* w32, const Plane& in, const Plane& out, const Plane* res, int oy0, int oy1,
                  unsigned char* thread_ws, std::size_t thread_ws_stride, ThreadPool& pool) const;
    // first / one-past-last input row needed for output rows [oy0, oy1)
    int input_row_begin(int oy0) const { return oy0 * sh - pt; }
    int input_row_end(int oy1) const { return (oy1 - 1) * sh - pt + (kh - 1) * dh + 1; }

    static std::size_t panel_budget_bytes;  // per-thread im2col panel budget (tunable)

private:
    int tile_pixels() const;
    void run_rows_winograd(const float* u, const Plane& in, const Plane& out, const Plane* res, int oy0, int oy1,
                           unsigned char* thread_ws, std::size_t stride, ThreadPool& pool) const;
    void winograd_chunk(const float* u, const Plane& in, const Plane& out, const Plane* res, int oy, int rows,
                        int tx0, unsigned char* ws) const;
    void run_tile(const float* w32, const Plane& in, const Plane& out, const Plane* res, int out_w, int pa, int pbnd,
                  unsigned char* ws) const;
    template <class Tag>
    void pack_panel(const Plane& in, bool pointwise, int pa, int npix, int ntmax, const int* yoff, const int* xoff,
                    const int* cix0, const unsigned char* fast, float* panel) const;
};
