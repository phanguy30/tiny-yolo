#include "conv.hpp"

#include "dtype.hpp"
#include "fastmath.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <type_traits>

std::size_t ConvolutionLayer::panel_budget_bytes = 32 * 1024;

namespace {
constexpr std::size_t kAlign = 64;
std::size_t align_up(std::size_t v) { return (v + kAlign - 1) / kAlign * kAlign; }
}  // namespace

// ---------------------------------------------------------------- PackedWeights
int PackedWeights::int8_group = 64;

void PackedWeights::init(int m, int k, WeightFormat f) {
    M = m; K = k; format = f;
    const std::size_t n = padded_count();
    f32_.clear(); f16_.clear(); i8_.clear(); scale_.clear();
    if (f == WeightFormat::F32) f32_.assign(n, 0.f);
    else if (f == WeightFormat::F16) f16_.assign(n, 0);
    else {
        groups_ = (K + int8_group - 1) / int8_group;
        i8_.assign(n, 0);
        scale_.assign(static_cast<std::size_t>(blocks()) * MR * static_cast<std::size_t>(groups_), 0.f);
    }
}

void PackedWeights::init_stream(int m, int k, std::streamoff offset) {
    M = m; K = k; format = WeightFormat::Stream; offset_ = offset;
    f32_.clear(); f16_.clear(); i8_.clear(); scale_.clear();
}

void PackedWeights::read_stream(std::istream& file, float* dst, float* staging) const {
    file.clear();
    file.seekg(offset_);
    for (int b = 0; b < blocks(); ++b) {
        const int rows = std::min(MR, M - b * MR);
        file.read(reinterpret_cast<char*>(staging), static_cast<std::streamsize>(static_cast<std::size_t>(rows) * K * sizeof(float)));
        if (!file) throw std::runtime_error("failed to stream weights from model file");
        float* d = dst + static_cast<std::size_t>(b) * MR * static_cast<std::size_t>(K);
        for (int k = 0; k < K; ++k)
            for (int i = 0; i < MR; ++i) d[k * MR + i] = i < rows ? staging[static_cast<std::size_t>(i) * K + k] : 0.f;
    }
}

void PackedWeights::set_row(int oc, const float* w) {
    const std::size_t blk = static_cast<std::size_t>(oc / MR), lane = static_cast<std::size_t>(oc % MR);
    const std::size_t base = blk * MR * static_cast<std::size_t>(K) + lane;
    if (format == WeightFormat::F32) {
        for (int k = 0; k < K; ++k) f32_[base + static_cast<std::size_t>(k) * MR] = w[k];
    } else if (format == WeightFormat::F16) {
        for (int k = 0; k < K; ++k) f16_[base + static_cast<std::size_t>(k) * MR] = fm::float_to_half(w[k]);
    } else {  // symmetric int8, one scale per (output channel, group of int8_group weights along K)
        for (int g = 0; g < groups_; ++g) {
            const int k0 = g * int8_group, k1 = std::min(K, k0 + int8_group);
            float amax = 0.f;
            for (int k = k0; k < k1; ++k) amax = std::max(amax, std::fabs(w[k]));
            const float scale = amax > 0.f ? amax / 127.f : 1.f;
            scale_[static_cast<std::size_t>(oc) * groups_ + static_cast<std::size_t>(g)] = scale;
            for (int k = k0; k < k1; ++k) {
                const float q = std::nearbyint(w[k] / scale);
                i8_[base + static_cast<std::size_t>(k) * MR] = static_cast<std::int8_t>(std::max(-127.f, std::min(127.f, q)));
            }
        }
    }
}

void PackedWeights::finalize() {
    f32_.shrink_to_fit(); f16_.shrink_to_fit(); i8_.shrink_to_fit(); scale_.shrink_to_fit();
}

void PackedWeights::decode_blocks(float* dst, int b0, int b1) const {
    const std::size_t per = static_cast<std::size_t>(MR) * static_cast<std::size_t>(K);
    for (int b = b0; b < b1; ++b) {
        float* d = dst + static_cast<std::size_t>(b) * per;
        const std::size_t off = static_cast<std::size_t>(b) * per;
        if (format == WeightFormat::F32) {
            std::memcpy(d, f32_.data() + off, per * sizeof(float));
        } else if (format == WeightFormat::F16) {
            const std::uint16_t* s = f16_.data() + off;
            for (std::size_t i = 0; i < per; ++i) d[i] = fm::half_to_float(s[i]);
        } else {
            const std::int8_t* s = i8_.data() + off;
            for (int g = 0; g < groups_; ++g) {
                float sc[MR];
                for (int i = 0; i < MR; ++i) sc[i] = scale_[(static_cast<std::size_t>(b) * MR + i) * groups_ + g];
                const int k0 = g * int8_group, k1 = std::min(K, k0 + int8_group);
                for (int k = k0; k < k1; ++k)
                    for (int i = 0; i < MR; ++i) d[k * MR + i] = static_cast<float>(s[k * MR + i]) * sc[i];
            }
        }
    }
}

std::size_t PackedWeights::bytes() const {
    return f32_.size() * 4 + f16_.size() * 2 + i8_.size() + scale_.size() * 4;
}

// ---------------------------------------------------------------- layer setup
void ConvolutionLayer::load_param(const ParamDict& pd) {
    num_output = pd.get_int(0, 0);
    kw = pd.get_int(1, 0); kh = pd.get_int(11, kw);
    dw = pd.get_int(2, 1); dh = pd.get_int(12, dw);
    sw = pd.get_int(3, 1); sh = pd.get_int(13, sw);
    pl = pd.get_int(4, 0); pt = pd.get_int(14, pl);
    pr = pd.get_int(15, pl); pb = pd.get_int(16, pt);
    bias_term = pd.get_int(5, 0);
    weight_data_size = pd.get_int(6, 0);
    if (pl < 0 || pt < 0 || pr < 0 || pb < 0) throw std::runtime_error(name + ": SAME padding modes not supported");
    if (pd.get_int(8, 0) != 0) throw std::runtime_error(name + ": int8 ncnn models not supported");
    switch (pd.get_int(9, 0)) {
        case 0: act = Activation::None; break;
        case 1: act = Activation::ReLU; break;
        case 2: {
            act = Activation::LeakyReLU;
            auto a = pd.get_array(10);
            act_alpha = a.empty() ? 0.f : a[0];
            break;
        }
        case 4: act = Activation::Sigmoid; break;
        default: throw std::runtime_error(name + ": unsupported fused activation type");
    }
    if (num_output <= 0 || kw <= 0 || weight_data_size % (num_output * kh * kw) != 0)
        throw std::runtime_error(name + ": inconsistent convolution parameters");
    in_channels = weight_data_size / (num_output * kh * kw);
}

void ConvolutionLayer::load_model(ModelBin& mb, const EngineConfig& cfg) {
    const int k = K();
    // Size-based mixed precision: see EngineConfig::fp32_below / int8_min.
    WeightFormat fmt = weight_format_for(cfg, static_cast<std::size_t>(weight_data_size));
    if (fmt == WeightFormat::Stream) {
        const std::streamoff off = mb.skip_tagged_f32(static_cast<std::size_t>(weight_data_size));
        if (off >= 0) {
            weights.init_stream(num_output, k, off);
            bias.assign(static_cast<std::size_t>(num_output), 0.f);
            if (bias_term) mb.read_raw(bias.data(), bias.size());
            return;
        }
        fmt = WeightFormat::F16;  // non-fp32 block: keep resident
    }
    weights.init(num_output, k, fmt);
    // Stream the weights one output channel at a time: bounded transient memory.
    std::vector<float> row(static_cast<std::size_t>(k));
    std::size_t filled = 0;
    int oc = 0;
    mb.read_tagged(static_cast<std::size_t>(weight_data_size), [&](const float* p, std::size_t n) {
        while (n > 0) {
            const std::size_t take = std::min(n, static_cast<std::size_t>(k) - filled);
            std::memcpy(row.data() + filled, p, take * sizeof(float));
            filled += take; p += take; n -= take;
            if (filled == static_cast<std::size_t>(k)) { weights.set_row(oc++, row.data()); filled = 0; }
        }
    });
    weights.finalize();
    bias.assign(static_cast<std::size_t>(num_output), 0.f);
    if (bias_term) mb.read_raw(bias.data(), bias.size());
}

std::vector<Shape> ConvolutionLayer::infer_shapes(const std::vector<Shape>& in) const {
    const Shape& s = in.at(0);
    if (s.c() != in_channels) throw std::runtime_error(name + ": input channel mismatch " + s.str());
    Shape o{num_output, out_h(s.h()), out_w(s.w())};
    if (has_residual && in.at(1) != o) throw std::runtime_error(name + ": residual shape mismatch");
    return {o};
}

int ConvolutionLayer::tile_pixels() const {
    int nt = static_cast<int>(panel_budget_bytes / (static_cast<std::size_t>(K()) * sizeof(float)));
    nt = nt / NR * NR;
    return std::max(NR, std::min(nt, 256));
}

std::size_t ConvolutionLayer::decoded_weight_bytes() const {
    if (weights.format == WeightFormat::F32) return 0;
    std::size_t b = align_up(weights.padded_count() * sizeof(float));
    if (weights.format == WeightFormat::Stream)  // staging for one block of MR rows
        b += align_up(static_cast<std::size_t>(MR) * static_cast<std::size_t>(K()) * sizeof(float));
    return b;
}

std::size_t ConvolutionLayer::thread_workspace_bytes() const {
    const std::size_t nt = static_cast<std::size_t>(tile_pixels());
    std::size_t b = align_up(static_cast<std::size_t>(K()) * nt * sizeof(float));             // im2col panel
    b += align_up((static_cast<std::size_t>(kh + kw) + 3) * nt * sizeof(int) + nt / NR + 8);  // index tables
    return b;
}

std::size_t ConvolutionLayer::workspace_bytes(const std::vector<Shape>&, const std::vector<Shape>&, int threads) const {
    return decoded_weight_bytes() + static_cast<std::size_t>(threads) * thread_workspace_bytes();
}

const float* ConvolutionLayer::prepare_weights(float* scratch, ExecContext& ctx) const {
    if (weights.format == WeightFormat::F32) return weights.f32();
    if (weights.format == WeightFormat::Stream) {
        if (!ctx.weight_file) throw std::runtime_error(name + ": streamed weights need an open model file");
        weights.read_stream(*ctx.weight_file, scratch, scratch + align_up(weights.padded_count() * sizeof(float)) / sizeof(float));
        return scratch;
    }
    ThreadPool& pool = *ctx.pool;
    const int nb = weights.blocks();
    const int chunks = std::min(nb, pool.size() * 4);
    pool.parallel_for(chunks, [&](int t, int) {
        weights.decode_blocks(scratch, nb * t / chunks, nb * (t + 1) / chunks);
    });
    return scratch;
}

void ConvolutionLayer::forward(const std::vector<TensorView>& in, const std::vector<TensorView>& out, ExecContext& ctx) const {
    auto* ws = static_cast<unsigned char*>(ctx.workspace);
    const float* w32 = prepare_weights(reinterpret_cast<float*>(ws), ctx);
    ws += decoded_weight_bytes();
    const Plane ip = Plane::dense(in[0]);
    const Plane op = Plane::dense(out[0]);
    Plane rp;
    if (has_residual) rp = Plane::dense(in[1]);
    run_rows(w32, ip, op, has_residual ? &rp : nullptr, 0, op.H, ws, thread_workspace_bytes(), *ctx.pool);
}

// ---------------------------------------------------------------- compute
namespace {

constexpr int MR = ConvolutionLayer::MR;
constexpr int NR = ConvolutionLayer::NR;

// C[MR][NR] = A[K][MR]^T * B[K][NR].  Plain C++ written so that GCC/Clang keep
// the 8x8 tile in vector registers (NEON: 16 q-regs, AVX: 8 ymm) and emit
// broadcast-FMA sequences.  This form is ~5x faster than a naive 2D-array
// accumulator, which compilers vectorize with expensive shuffles.
inline void micro_kernel(int K, const float* __restrict A, const float* __restrict B, float* __restrict C) {
    for (int i = 0; i < MR * NR; ++i) C[i] = 0.f;
    for (int k = 0; k < K; ++k) {
        const float* a = A + k * MR;
        const float* b = B + k * NR;
#pragma GCC unroll 8
        for (int i = 0; i < MR; ++i) {
            const float ai = a[i];
#pragma GCC ivdep
            for (int j = 0; j < NR; ++j) C[i * NR + j] += ai * b[j];
        }
    }
}

inline void apply_activation(float* v, Activation act, float alpha) {
    switch (act) {
        case Activation::None: break;
        case Activation::ReLU: for (int j = 0; j < NR; ++j) v[j] = v[j] > 0.f ? v[j] : 0.f; break;
        case Activation::LeakyReLU: for (int j = 0; j < NR; ++j) v[j] = v[j] > 0.f ? v[j] : v[j] * alpha; break;
        case Activation::Sigmoid: for (int j = 0; j < NR; ++j) v[j] = fm::sigmoid(v[j]); break;
        case Activation::Swish: for (int j = 0; j < NR; ++j) v[j] = fm::swish(v[j]); break;
    }
}

}  // namespace

void ConvolutionLayer::run_rows(const float* w32, const Plane& in, const Plane& out, const Plane* res, int oy0, int oy1,
                                unsigned char* thread_ws, std::size_t stride, ThreadPool& pool) const {
    if (oy1 <= oy0) return;
    const int ow = out.W;
    const int p0 = oy0 * ow, p1 = oy1 * ow, n = p1 - p0;
    int nt = tile_pixels();
    const int want = 2 * pool.size();
    if ((n + nt - 1) / nt < want) nt = std::max(NR, ((n + want - 1) / want + NR - 1) / NR * NR);
    const int tiles = (n + nt - 1) / nt;
    pool.parallel_for(tiles, [&](int t, int th) {
        const int pa = p0 + t * nt;
        run_tile(w32, in, out, res, ow, pa, std::min(pa + nt, p1), thread_ws + static_cast<std::size_t>(th) * stride);
    });
}

void ConvolutionLayer::run_tile(const float* w32, const Plane& in, const Plane& out, const Plane* res, int ow, int pa,
                                int pbnd, unsigned char* ws) const {
    const int k_total = K();
    const int npix = pbnd - pa;
    const int nchunks = (npix + NR - 1) / NR;
    const int ntp = nchunks * NR;  // padded pixel count
    const int ntmax = tile_pixels();

    float* panel = reinterpret_cast<float*>(ws);
    int* yoff = reinterpret_cast<int*>(ws + align_up(static_cast<std::size_t>(k_total) * static_cast<std::size_t>(ntmax) * sizeof(float)));
    int* xoff = yoff + static_cast<std::size_t>(kh) * static_cast<std::size_t>(ntmax);
    int* ooff = xoff + static_cast<std::size_t>(kw) * static_cast<std::size_t>(ntmax);
    int* roff = ooff + ntmax;
    int* cix0 = roff + ntmax;  // per chunk: input x of the first pixel (may be negative)
    unsigned char* fast = reinterpret_cast<unsigned char*>(cix0 + ntmax);

    // ---- per-pixel index tables (one div/mod per tile, not per element)
    {
        int oy = pa / ow, ox = pa - (pa / ow) * ow;
        for (int j = 0; j < ntp; ++j) {
            const bool valid = j < npix;
            const int iy0 = oy * sh - pt, ix0 = ox * sw - pl;
            for (int ky = 0; ky < kh; ++ky) {
                const int iy = iy0 + ky * dh;
                yoff[ky * ntmax + j] = (valid && iy >= 0 && iy < in.H) ? static_cast<int>(in.row_offset(iy)) : -1;
            }
            for (int kx = 0; kx < kw; ++kx) {
                const int ix = ix0 + kx * dw;
                xoff[kx * ntmax + j] = (valid && ix >= 0 && ix < in.W) ? ix : -1;
            }
            ooff[j] = valid ? static_cast<int>(out.row_offset(oy)) + ox : 0;
            roff[j] = (valid && res) ? static_cast<int>(res->row_offset(oy)) + ox : 0;
            if (j % NR == 0) {  // chunk lies in one output row and is fully valid
                fast[j / NR] = static_cast<unsigned char>(j + NR <= npix && ox + NR <= ow);
                cix0[j / NR] = ix0;
            }
            if (++ox == ow) { ox = 0; ++oy; }
        }
    }

    // ---- im2col into packed panel [chunk][K][NR]
    const bool pointwise = kh == 1 && kw == 1 && sh == 1 && sw == 1 && pt == 0 && pl == 0 && in.ring == 0 &&
                           in.W == out.W && in.H == out.H;
    dispatch_dtype(in.dtype, [&](auto tag) {
        pack_panel<decltype(tag)>(in, pointwise, pa, npix, ntmax, yoff, xoff, cix0, fast, panel);
    });

    // ---- GEMM + fused epilogue (bias, activation, residual)
    const int nblocks = weights.blocks();
    alignas(64) float acc[MR * NR];
    alignas(64) float v[NR];
    for (int mb = 0; mb < nblocks; ++mb) {
        const float* a = w32 + static_cast<std::size_t>(mb) * MR * static_cast<std::size_t>(k_total);
        for (int q = 0; q < nchunks; ++q) {
            micro_kernel(k_total, a, panel + static_cast<std::size_t>(q) * static_cast<std::size_t>(k_total) * NR, acc);
            const int j0 = q * NR;
            const int nv = std::min(NR, npix - j0);
            for (int i = 0; i < MR; ++i) {
                const int oc = mb * MR + i;
                if (oc >= num_output) break;
                const float b = bias[static_cast<std::size_t>(oc)];
                for (int j = 0; j < NR; ++j) v[j] = acc[i * NR + j] + b;
                apply_activation(v, act, act_alpha);
                const std::size_t obase = static_cast<std::size_t>(oc) * out.cstride;
                if (res) {
                    const std::size_t rbase = static_cast<std::size_t>(oc) * res->cstride;
                    if (res->dtype == DType::F16) {
                        const auto* r = static_cast<const std::uint16_t*>(res->base) + rbase;
                        for (int j = 0; j < nv; ++j) v[j] += fm::half_to_float(r[roff[j0 + j]]);
                    } else {
                        const auto* r = static_cast<const float*>(res->base) + rbase;
                        for (int j = 0; j < nv; ++j) v[j] += r[roff[j0 + j]];
                    }
                }
                if (out.dtype == DType::F16) {
                    auto* o = static_cast<std::uint16_t*>(out.base) + obase;
                    for (int j = 0; j < nv; ++j) o[ooff[j0 + j]] = fm::float_to_half_fast(v[j]);
                } else {
                    auto* o = static_cast<float*>(out.base) + obase;
                    for (int j = 0; j < nv; ++j) o[ooff[j0 + j]] = v[j];
                }
            }
        }
    }
}

template <class Tag>
void ConvolutionLayer::pack_panel(const Plane& in, bool pointwise, int pa, int npix, int ntmax, const int* yoff,
                                  const int* xoff, const int* cix0, const unsigned char* fast, float* panel) const {
    constexpr int kMaxSpan = 64;
    using T = typename Tag::T;
    const T* in_base = static_cast<const T*>(in.base);
    const float scale = in.scale;
    auto cvt = [scale](const T* p, std::size_t i) {
        if constexpr (std::is_same<Tag, U8Tag>::value) return Tag::ld(p, i) * scale;
        else return Tag::ld(p, i);
    };
    const int k_total = K();
    const int nchunks = (npix + NR - 1) / NR;
    for (int q = 0; q < nchunks; ++q) {
        float* dst = panel + static_cast<std::size_t>(q) * static_cast<std::size_t>(k_total) * NR;
        const int j0 = q * NR;
        const int nv = std::min(NR, npix - j0);
        if (pointwise) {
            for (int ic = 0; ic < in_channels; ++ic) {
                const T* src = in_base + static_cast<std::size_t>(ic) * in.cstride + static_cast<std::size_t>(pa + j0);
                float* d = dst + static_cast<std::size_t>(ic) * NR;
                if (nv == NR) { for (int j = 0; j < NR; ++j) d[j] = cvt(src, static_cast<std::size_t>(j)); }
                else { for (int j = 0; j < NR; ++j) d[j] = j < nv ? cvt(src, static_cast<std::size_t>(j)) : 0.f; }
            }
            continue;
        }
        const int span = (NR - 1) * sw + (kw - 1) * dw + 1;
        if (fast[q] && span <= kMaxSpan) {
            // All NR pixels share one output row: per (ic, ky) convert the source
            // window once, then emit the kw shifted copies (3x fewer loads/conversions
            // for 3x3 kernels, no per-element bounds checks).
            const int x0 = cix0[q];
            const bool inside = x0 >= 0 && x0 + span <= in.W;
            float win[kMaxSpan];
            for (int ic = 0; ic < in_channels; ++ic) {
                const T* cb = in_base + static_cast<std::size_t>(ic) * in.cstride;
                for (int ky = 0; ky < kh; ++ky) {
                    float* d = dst + static_cast<std::size_t>(ic * kh + ky) * kw * NR;
                    const int yo = yoff[ky * ntmax + j0];
                    if (yo < 0) { for (int t = 0; t < kw * NR; ++t) d[t] = 0.f; continue; }
                    const T* row = cb + yo;
                    if (inside) {
                        for (int t = 0; t < span; ++t) win[t] = cvt(row, static_cast<std::size_t>(x0 + t));
                    } else {
                        for (int t = 0; t < span; ++t) {
                            const int x = x0 + t;
                            win[t] = (x >= 0 && x < in.W) ? cvt(row, static_cast<std::size_t>(x)) : 0.f;
                        }
                    }
                    if (sw == 1) {
                        for (int kx = 0; kx < kw; ++kx)
                            for (int j = 0; j < NR; ++j) d[kx * NR + j] = win[kx * dw + j];
                    } else {
                        for (int kx = 0; kx < kw; ++kx)
                            for (int j = 0; j < NR; ++j) d[kx * NR + j] = win[kx * dw + j * sw];
                    }
                }
            }
            continue;
        }
        for (int ic = 0; ic < in_channels; ++ic) {
            const T* cb = in_base + static_cast<std::size_t>(ic) * in.cstride;
            for (int ky = 0; ky < kh; ++ky) {
                const int* yo = yoff + ky * ntmax + j0;
                for (int kx = 0; kx < kw; ++kx) {
                    float* d = dst + (static_cast<std::size_t>(ic * kh + ky) * kw + kx) * NR;
                    const int* xo = xoff + kx * ntmax + j0;
                    for (int j = 0; j < NR; ++j)
                        d[j] = (yo[j] >= 0 && xo[j] >= 0) ? cvt(cb, static_cast<std::size_t>(yo[j] + xo[j])) : 0.f;
                }
            }
        }
    }
}
