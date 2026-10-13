// Non-convolution layers + the layer registry.
#include "conv.hpp"
#include "dtype.hpp"
#include "fastmath.hpp"
#include "layer.hpp"
#include "layers.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <unordered_map>

// ---------------------------------------------------------------- ModelBin
std::uint32_t modelbin_read_tag(std::istream& in) {
    std::uint32_t tag = 0;
    in.read(reinterpret_cast<char*>(&tag), 4);
    return tag;
}
float modelbin_half_to_float(std::uint16_t h) { return fm::half_to_float(h); }

std::streamoff ModelBin::skip_tagged_f32(std::size_t count) {
    const std::streamoff start = in_.tellg();
    const std::uint32_t tag = modelbin_read_tag(in_);
    if (tag != 0) { in_.seekg(start); return -1; }
    const std::streamoff data = in_.tellg();
    in_.seekg(static_cast<std::streamoff>(count * sizeof(float)), std::ios::cur);
    if (!in_) throw std::runtime_error("unexpected end of .bin weight file");
    return data;
}

void ModelBin::read_raw(float* dst, std::size_t count) {
    in_.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(count * sizeof(float)));
    if (!in_) throw std::runtime_error("unexpected end of .bin weight file");
}

namespace {

// Splits [0, n) into chunks and runs fn(begin, end) on the pool.
template <class F>
void parallel_range(ThreadPool& pool, std::size_t n, std::size_t min_chunk, F&& fn) {
    const std::size_t max_tasks = static_cast<std::size_t>(pool.size()) * 4;
    std::size_t tasks = std::min(max_tasks, (n + min_chunk - 1) / min_chunk);
    if (tasks <= 1) { fn(std::size_t{0}, n); return; }
    pool.parallel_for(static_cast<int>(tasks), [&](int t, int) {
        fn(n * static_cast<std::size_t>(t) / tasks, n * static_cast<std::size_t>(t + 1) / tasks);
    });
}

void outer_inner(const Shape& s, int axis, std::size_t& outer, std::size_t& inner) {
    outer = 1; inner = 1;
    for (int d = 0; d < axis; ++d) outer *= static_cast<std::size_t>(s[d]);
    for (int d = axis + 1; d < s.ndim; ++d) inner *= static_cast<std::size_t>(s[d]);
}

// ---------------------------------------------------------------- Input
class InputLayer : public Layer {
public:
    std::vector<Shape> infer_shapes(const std::vector<Shape>&) const override {
        throw std::runtime_error("Input shape is provided by the engine");
    }
    void forward(const std::vector<TensorView>&, const std::vector<TensorView>&, ExecContext&) const override {}
};

// ---------------------------------------------------------------- elementwise activations
class UnaryLayer : public Layer {
public:
    explicit UnaryLayer(Activation a) : act_(a) {}
    Activation activation() const { return act_; }
    std::vector<Shape> infer_shapes(const std::vector<Shape>& in) const override { return {in.at(0)}; }
    bool inplace_capable() const override { return true; }
    void forward(const std::vector<TensorView>& in, const std::vector<TensorView>& out, ExecContext& ctx) const override {
        dispatch_float2(in[0].dtype, out[0].dtype, [&](auto ti, auto to) {
            using TI = decltype(ti);
            using TO = decltype(to);
            const auto* x = static_cast<const typename TI::T*>(in[0].data);
            auto* y = static_cast<typename TO::T*>(out[0].data);
            parallel_range(*ctx.pool, in[0].size(), 16384, [&](std::size_t b, std::size_t e) {
                if (act_ == Activation::Swish) for (std::size_t i = b; i < e; ++i) TO::st(y, i, fm::swish(TI::ld(x, i)));
                else if (act_ == Activation::Sigmoid) for (std::size_t i = b; i < e; ++i) TO::st(y, i, fm::sigmoid(TI::ld(x, i)));
                else for (std::size_t i = b; i < e; ++i) { const float v = TI::ld(x, i); TO::st(y, i, v > 0.f ? v : 0.f); }
            });
        });
    }

private:
    Activation act_;
};

// ---------------------------------------------------------------- Pooling
class PoolingLayer : public Layer {
public:
    int type = 0, kw = 1, kh = 1, sw = 1, sh = 1, pl = 0, pr = 0, pt = 0, pb = 0, global = 0, pad_mode = 0, count_pad = 1;
    void load_param(const ParamDict& pd) override {
        type = pd.get_int(0, 0);
        kw = pd.get_int(1, 0); kh = pd.get_int(11, kw);
        sw = pd.get_int(2, 1); sh = pd.get_int(12, sw);
        pl = pd.get_int(3, 0); pr = pd.get_int(14, pl);
        pt = pd.get_int(13, pl); pb = pd.get_int(15, pt);
        global = pd.get_int(4, 0);
        pad_mode = pd.get_int(5, 0);
        count_pad = pd.get_int(6, 1);
        if (pd.get_int(7, 0)) throw std::runtime_error(name + ": adaptive pooling not supported");
        if (pad_mode > 1) throw std::runtime_error(name + ": SAME pad modes not supported");
    }
    void pads(const Shape& s, int& r, int& b) const {
        r = pr; b = pb;
        if (pad_mode == 0) {  // "full" padding: extend so the last window fits (ceil mode)
            const int wt = (s.w() + pl + pr - kw) % sw, ht = (s.h() + pt + pb - kh) % sh;
            if (wt) r += sw - wt;
            if (ht) b += sh - ht;
        }
    }
    std::vector<Shape> infer_shapes(const std::vector<Shape>& in) const override {
        const Shape& s = in.at(0);
        if (global) return {Shape{s.c()}};
        int r, b; pads(s, r, b);
        return {Shape{s.c(), (s.h() + pt + b - kh) / sh + 1, (s.w() + pl + r - kw) / sw + 1}};
    }
    void forward(const std::vector<TensorView>& in, const std::vector<TensorView>& out, ExecContext& ctx) const override {
        const Shape& s = in[0].shape;
        const int c = s.c(), h = s.h(), w = s.w();
        dispatch_float2(in[0].dtype, out[0].dtype, [&](auto ti, auto to) {
            using TI = decltype(ti);
            using TO = decltype(to);
            const auto* xin = static_cast<const typename TI::T*>(in[0].data);
            auto* yout = static_cast<typename TO::T*>(out[0].data);
            if (global) {
                ctx.pool->parallel_for(c, [&](int ch, int) {
                    const auto* x = xin + static_cast<std::size_t>(ch) * h * w;
                    float acc = type == 0 ? -std::numeric_limits<float>::infinity() : 0.f;
                    for (int i = 0; i < h * w; ++i) {
                        const float v = TI::ld(x, static_cast<std::size_t>(i));
                        acc = type == 0 ? std::max(acc, v) : acc + v;
                    }
                    TO::st(yout, static_cast<std::size_t>(ch), type == 0 ? acc : acc / static_cast<float>(h * w));
                });
                return;
            }
            const int oh = out[0].shape.h(), ow = out[0].shape.w();
            int padr, padb;
            pads(s, padr, padb);
            ctx.pool->parallel_for(c, [&](int ch, int) {
                const auto* x = xin + static_cast<std::size_t>(ch) * h * w;
                auto* y = yout + static_cast<std::size_t>(ch) * oh * ow;
                for (int oy = 0; oy < oh; ++oy) {
                    const int y0 = oy * sh - pt;
                    const int ya = std::max(y0, 0), yb = std::min(y0 + kh, h);
                    for (int ox = 0; ox < ow; ++ox) {
                        const int x0 = ox * sw - pl;
                        const int xa = std::max(x0, 0), xb = std::min(x0 + kw, w);
                        float r;
                        if (type == 0) {
                            r = -std::numeric_limits<float>::infinity();
                            for (int iy = ya; iy < yb; ++iy)
                                for (int ix = xa; ix < xb; ++ix) r = std::max(r, TI::ld(x, static_cast<std::size_t>(iy * w + ix)));
                        } else {
                            float sum = 0.f;
                            for (int iy = ya; iy < yb; ++iy)
                                for (int ix = xa; ix < xb; ++ix) sum += TI::ld(x, static_cast<std::size_t>(iy * w + ix));
                            const int area = count_pad ? (std::min(y0 + kh, h + padb) - y0) * (std::min(x0 + kw, w + padr) - x0)
                                                       : (yb - ya) * (xb - xa);
                            r = sum / static_cast<float>(std::max(area, 1));
                        }
                        TO::st(y, static_cast<std::size_t>(oy * ow + ox), r);
                    }
                }
            });
        });
    }
};

// ---------------------------------------------------------------- Interp (nearest)
class InterpLayer : public Layer {
public:
    int resize_type = 1, out_h = 0, out_w = 0;
    float hs = 1.f, ws = 1.f;
    void load_param(const ParamDict& pd) override {
        resize_type = pd.get_int(0, 1);
        hs = pd.get_float(1, 1.f); ws = pd.get_float(2, 1.f);
        out_h = pd.get_int(3, 0); out_w = pd.get_int(4, 0);
        if (resize_type != 1) throw std::runtime_error(name + ": only nearest Interp is supported");
    }
    std::vector<Shape> infer_shapes(const std::vector<Shape>& in) const override {
        const Shape& s = in.at(0);
        const int oh = out_h ? out_h : static_cast<int>(static_cast<float>(s.h()) * hs);
        const int ow = out_w ? out_w : static_cast<int>(static_cast<float>(s.w()) * ws);
        return {Shape{s.c(), oh, ow}};
    }
    void forward(const std::vector<TensorView>& in, const std::vector<TensorView>& out, ExecContext& ctx) const override {
        const int c = in[0].shape.c(), h = in[0].shape.h(), w = in[0].shape.w();
        const int oh = out[0].shape.h(), ow = out[0].shape.w();
        const float fy = out_h ? static_cast<float>(h) / static_cast<float>(oh) : 1.f / hs;
        const float fx = out_w ? static_cast<float>(w) / static_cast<float>(ow) : 1.f / ws;
        dispatch_float2(in[0].dtype, out[0].dtype, [&](auto ti, auto to) {
            using TI = decltype(ti);
            using TO = decltype(to);
            ctx.pool->parallel_for(c, [&](int ch, int) {
                const auto* x = static_cast<const typename TI::T*>(in[0].data) + static_cast<std::size_t>(ch) * h * w;
                auto* y = static_cast<typename TO::T*>(out[0].data) + static_cast<std::size_t>(ch) * oh * ow;
                for (int oy = 0; oy < oh; ++oy) {
                    const int sy = std::min(static_cast<int>(static_cast<float>(oy) * fy), h - 1);
                    for (int ox = 0; ox < ow; ++ox) {
                        const int sx = std::min(static_cast<int>(static_cast<float>(ox) * fx), w - 1);
                        TO::st(y, static_cast<std::size_t>(oy * ow + ox), TI::ld(x, static_cast<std::size_t>(sy * w + sx)));
                    }
                }
            });
        });
    }
};

// ---------------------------------------------------------------- Concat
class ConcatLayer : public Layer {
public:
    int axis = 0;
    void load_param(const ParamDict& pd) override { axis = pd.get_int(0, 0); }
    int real_axis(const Shape& s) const { return axis < 0 ? axis + s.ndim : axis; }
    std::vector<Shape> infer_shapes(const std::vector<Shape>& in) const override {
        Shape s = in.at(0);
        const int a = real_axis(s);
        for (std::size_t k = 1; k < in.size(); ++k) {
            for (int d = 0; d < s.ndim; ++d)
                if (d != a && in[k][d] != s[d]) throw std::runtime_error(name + ": concat shape mismatch");
            s[a] += in[k][a];
        }
        return {s};
    }
    bool concat_slot(const std::vector<Shape>& in, int idx, std::size_t& off) const override {
        const int a = real_axis(in[0]);
        std::size_t outer, inner;
        outer_inner(in[0], a, outer, inner);
        if (outer != 1) return false;
        std::size_t o = 0;
        for (int k = 0; k < idx; ++k) o += static_cast<std::size_t>(in[static_cast<std::size_t>(k)][a]) * inner;
        off = o;
        return true;
    }
    void forward(const std::vector<TensorView>& in, const std::vector<TensorView>& out, ExecContext& ctx) const override {
        const int a = real_axis(out[0].shape);
        std::size_t outer, inner;
        outer_inner(out[0].shape, a, outer, inner);
        const std::size_t out_axis = static_cast<std::size_t>(out[0].shape[a]);
        const std::size_t oes = dtype_size(out[0].dtype);
        std::size_t axis_off = 0;
        for (const auto& x : in) {
            const std::size_t n = static_cast<std::size_t>(x.shape[a]) * inner;
            unsigned char* dst0 = static_cast<unsigned char*>(out[0].data) + axis_off * inner * oes;
            axis_off += static_cast<std::size_t>(x.shape[a]);
            if (outer == 1 && x.data == dst0 && x.dtype == out[0].dtype) continue;  // producer wrote in place
            const std::size_t ies = dtype_size(x.dtype);
            parallel_range(*ctx.pool, outer * n, 65536, [&](std::size_t b, std::size_t e) {
                for (std::size_t i = b; i < e;) {
                    const std::size_t o = i / n, r = i % n, len = std::min(n - r, e - i);
                    copy_convert(static_cast<const unsigned char*>(x.data) + (o * n + r) * ies, x.dtype,
                                 dst0 + (o * out_axis * inner + r) * oes, out[0].dtype, len);
                    i += len;
                }
            });
        }
    }
};

// ---------------------------------------------------------------- Slice
class SliceLayer : public Layer {
public:
    std::vector<int> slices;
    int axis = 0;
    void load_param(const ParamDict& pd) override {
        for (float v : pd.get_array(0)) slices.push_back(static_cast<int>(v));
        axis = pd.get_int(1, 0);
    }
    std::vector<int> sizes(const Shape& s) const {
        const int a = axis < 0 ? axis + s.ndim : axis;
        std::vector<int> out = slices;
        int fixed = 0, automatic = 0;
        for (int v : out) (v == -233 ? ++automatic : fixed += v);
        const int each = automatic ? (s[a] - fixed) / automatic : 0;
        for (int& v : out) if (v == -233) v = each;
        return out;
    }
    std::vector<Shape> infer_shapes(const std::vector<Shape>& in) const override {
        const int a = axis < 0 ? axis + in[0].ndim : axis;
        std::vector<Shape> out;
        for (int v : sizes(in[0])) { Shape s = in[0]; s[a] = v; out.push_back(s); }
        return out;
    }
    bool offset_of(const Shape& s, int idx, std::size_t& off) const {
        const int a = axis < 0 ? axis + s.ndim : axis;
        std::size_t outer, inner;
        outer_inner(s, a, outer, inner);
        auto sz = sizes(s);
        std::size_t o = 0;
        for (int k = 0; k < idx; ++k) o += static_cast<std::size_t>(sz[static_cast<std::size_t>(k)]) * inner;
        off = o;
        return outer == 1;
    }
    bool view_output(const std::vector<Shape>& in, int idx, std::size_t& off) const override { return offset_of(in[0], idx, off); }
    void forward(const std::vector<TensorView>& in, const std::vector<TensorView>& out, ExecContext&) const override {
        const Shape& s = in[0].shape;
        const int a = axis < 0 ? axis + s.ndim : axis;
        std::size_t outer, inner;
        outer_inner(s, a, outer, inner);
        std::size_t axis_off = 0;
        for (std::size_t k = 0; k < out.size(); ++k) {
            const std::size_t part = static_cast<std::size_t>(out[k].shape[a]);
            std::size_t view_off = 0;
            const bool view = offset_of(s, static_cast<int>(k), view_off);
            const std::size_t ies = dtype_size(in[0].dtype), oes = dtype_size(out[k].dtype);
            const auto* src = static_cast<const unsigned char*>(in[0].data);
            if (!(view && out[k].data == src + view_off * ies && out[k].dtype == in[0].dtype)) {
                for (std::size_t o = 0; o < outer; ++o)
                    copy_convert(src + (o * static_cast<std::size_t>(s[a]) + axis_off) * inner * ies, in[0].dtype,
                                 static_cast<unsigned char*>(out[k].data) + o * part * inner * oes, out[k].dtype, part * inner);
            }
            axis_off += part;
        }
    }
};

// ---------------------------------------------------------------- Split (fan-out, always zero-copy)
class SplitLayer : public Layer {
public:
    std::vector<Shape> infer_shapes(const std::vector<Shape>& in) const override {
        return std::vector<Shape>(static_cast<std::size_t>(num_outputs), in.at(0));
    }
    bool view_output(const std::vector<Shape>&, int, std::size_t& off) const override { off = 0; return true; }
    void forward(const std::vector<TensorView>& in, const std::vector<TensorView>& out, ExecContext&) const override {
        for (const auto& o : out)
            if (o.data != in[0].data || o.dtype != in[0].dtype) copy_convert(in[0].data, in[0].dtype, o.data, o.dtype, in[0].size());
    }
};

// ---------------------------------------------------------------- Reshape (zero-copy view)
class ReshapeLayer : public Layer {
public:
    int w = -233, h = -233, c = -233;
    void load_param(const ParamDict& pd) override { w = pd.get_int(0, -233); h = pd.get_int(1, -233); c = pd.get_int(2, -233); }
    std::vector<Shape> infer_shapes(const std::vector<Shape>& in) const override {
        const Shape& s = in.at(0);
        std::vector<int> dims;
        if (c != -233) dims = {c, h, w}; else if (h != -233) dims = {h, w}; else dims = {w};
        // 0 = keep input dim (aligned from the innermost), -1 = infer
        int infer = -1;
        std::size_t known = 1;
        for (std::size_t i = 0; i < dims.size(); ++i) {
            const int from_inner = static_cast<int>(dims.size() - 1 - i);
            if (dims[i] == 0) dims[i] = from_inner < s.ndim ? s[s.ndim - 1 - from_inner] : 1;
            if (dims[i] == -1) { infer = static_cast<int>(i); continue; }
            known *= static_cast<std::size_t>(dims[i]);
        }
        if (infer >= 0) dims[static_cast<std::size_t>(infer)] = static_cast<int>(s.count() / known);
        Shape o = Shape::from(dims);
        if (o.count() != s.count()) throw std::runtime_error(name + ": reshape element count mismatch");
        return {o};
    }
    bool view_output(const std::vector<Shape>&, int, std::size_t& off) const override { off = 0; return true; }
    void forward(const std::vector<TensorView>& in, const std::vector<TensorView>& out, ExecContext&) const override {
        if (out[0].data != in[0].data || out[0].dtype != in[0].dtype)
            copy_convert(in[0].data, in[0].dtype, out[0].data, out[0].dtype, in[0].size());
    }
};

// ---------------------------------------------------------------- Permute
class PermuteLayer : public Layer {
public:
    int order = 0;
    void load_param(const ParamDict& pd) override { order = pd.get_int(0, 0); }
    // output axis -> input axis (axes in (c,h,w) order)
    std::array<int, 3> perm(int ndim) const {
        static const int p3[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
        if (ndim == 2) return order == 1 ? std::array<int, 3>{0, 2, 1} : std::array<int, 3>{0, 1, 2};
        if (order < 0 || order > 5) throw std::runtime_error(name + ": unsupported permute order");
        return {p3[order][0], p3[order][1], p3[order][2]};
    }
    std::vector<Shape> infer_shapes(const std::vector<Shape>& in) const override {
        const Shape& s = in.at(0);
        if (s.ndim < 2 || s.ndim > 3) throw std::runtime_error(name + ": permute supports 2D/3D");
        const int dims[3] = {s.c(), s.h(), s.w()};
        auto p = perm(s.ndim);
        if (s.ndim == 2) return {Shape{dims[p[1]], dims[p[2]]}};
        return {Shape{dims[p[0]], dims[p[1]], dims[p[2]]}};
    }
    void forward(const std::vector<TensorView>& in, const std::vector<TensorView>& out, ExecContext& ctx) const override {
        const Shape& s = in[0].shape;
        const std::size_t istr[3] = {static_cast<std::size_t>(s.h()) * s.w(), static_cast<std::size_t>(s.w()), 1};
        const int dims[3] = {s.c(), s.h(), s.w()};
        auto p = perm(s.ndim);
        const int oc = dims[p[0]], oh = dims[p[1]], ow = dims[p[2]];
        const std::size_t s0 = istr[p[0]], s1 = istr[p[1]], s2 = istr[p[2]];
        dispatch_float2(in[0].dtype, out[0].dtype, [&](auto ti, auto to) {
            using TI = decltype(ti);
            using TO = decltype(to);
            const auto* x = static_cast<const typename TI::T*>(in[0].data);
            ctx.pool->parallel_for(oc, [&](int a, int) {
                auto* y = static_cast<typename TO::T*>(out[0].data) + static_cast<std::size_t>(a) * oh * ow;
                for (int b = 0; b < oh; ++b)
                    for (int k = 0; k < ow; ++k)
                        TO::st(y, static_cast<std::size_t>(b * ow + k), TI::ld(x, a * s0 + b * s1 + k * s2));
            });
        });
    }
};

// ---------------------------------------------------------------- Softmax
class SoftmaxLayer : public Layer {
public:
    int axis = 0;
    void load_param(const ParamDict& pd) override { axis = pd.get_int(0, 0); }
    std::vector<Shape> infer_shapes(const std::vector<Shape>& in) const override { return {in.at(0)}; }
    bool inplace_capable() const override { return true; }
    void forward(const std::vector<TensorView>& in, const std::vector<TensorView>& out, ExecContext& ctx) const override {
        const Shape& s = in[0].shape;
        const int a = axis < 0 ? axis + s.ndim : axis;
        std::size_t outer, inner;
        outer_inner(s, a, outer, inner);
        const std::size_t n = static_cast<std::size_t>(s[a]);
        dispatch_float2(in[0].dtype, out[0].dtype, [&](auto ti, auto to) {
            using TI = decltype(ti);
            using TO = decltype(to);
            parallel_range(*ctx.pool, outer * inner, 2048, [&](std::size_t b, std::size_t e) {
                for (std::size_t idx = b; idx < e; ++idx) {
                    const std::size_t o = idx / inner, k = idx % inner;
                    const auto* x = static_cast<const typename TI::T*>(in[0].data) + o * n * inner + k;
                    auto* y = static_cast<typename TO::T*>(out[0].data) + o * n * inner + k;
                    float m = -std::numeric_limits<float>::infinity();
                    for (std::size_t i = 0; i < n; ++i) m = std::max(m, TI::ld(x, i * inner));
                    float sum = 0.f;
                    for (std::size_t i = 0; i < n; ++i) sum += fm::exp(TI::ld(x, i * inner) - m);
                    const float inv = 1.f / sum;
                    for (std::size_t i = 0; i < n; ++i) TO::st(y, i * inner, fm::exp(TI::ld(x, i * inner) - m) * inv);
                }
            });
        });
    }
};

// ---------------------------------------------------------------- BinaryOp
class BinaryOpLayer : public Layer {
public:
    int op = 0, with_scalar = 0;
    float scalar = 0.f;
    void load_param(const ParamDict& pd) override {
        op = pd.get_int(0, 0); with_scalar = pd.get_int(1, 0); scalar = pd.get_float(2, 0.f);
        if (op < 0 || op > 8 || op == 6) throw std::runtime_error(name + ": unsupported BinaryOp type");
    }
    std::vector<Shape> infer_shapes(const std::vector<Shape>& in) const override {
        if (in.size() > 1 && (in[1].count() == 0 || in[0].count() % in[1].count() != 0))
            throw std::runtime_error(name + ": unsupported broadcast");
        return {in.at(0)};
    }
    bool inplace_capable() const override { return true; }
    static float apply(int op, float x, float z) {
        switch (op) {
            case 0: return x + z;
            case 1: return x - z;
            case 2: return x * z;
            case 3: return x / z;
            case 4: return std::max(x, z);
            case 5: return std::min(x, z);
            case 7: return z - x;
            default: return z / x;
        }
    }
    void forward(const std::vector<TensorView>& in, const std::vector<TensorView>& out, ExecContext& ctx) const override {
        const std::size_t n = in[0].size();
        dispatch_float2(in[0].dtype, out[0].dtype, [&](auto ta, auto to) {
            using TA = decltype(ta);
            using TO = decltype(to);
            const auto* a = static_cast<const typename TA::T*>(in[0].data);
            auto* y = static_cast<typename TO::T*>(out[0].data);
            if (in.size() == 1) {
                parallel_range(*ctx.pool, n, 16384, [&](std::size_t b, std::size_t e) {
                    for (std::size_t i = b; i < e; ++i) TO::st(y, i, apply(op, TA::ld(a, i), scalar));
                });
                return;
            }
            dispatch_float(in[1].dtype, [&](auto tb) {
                using TB = decltype(tb);
                const auto* bp = static_cast<const typename TB::T*>(in[1].data);
                const std::size_t bn = in[1].size();
                parallel_range(*ctx.pool, n, 16384, [&](std::size_t b, std::size_t e) {
                    for (std::size_t i = b; i < e; ++i) TO::st(y, i, apply(op, TA::ld(a, i), TB::ld(bp, i % bn)));
                });
            });
        });
    }
};

// ---------------------------------------------------------------- MemoryData (constant)
class MemoryDataLayer : public Layer {
public:
    Shape shape;
    std::vector<float> data;
    void load_param(const ParamDict& pd) override {
        const int w = pd.get_int(0, 0), h = pd.get_int(1, 0), c = pd.get_int(2, 0);
        shape = c > 0 ? Shape{c, h, w} : (h > 0 ? Shape{h, w} : Shape{w});
    }
    void load_model(ModelBin& mb, const EngineConfig&) override {
        data.resize(shape.count());
        mb.read_raw(data.data(), data.size());
    }
    std::vector<Shape> infer_shapes(const std::vector<Shape>&) const override { return {shape}; }
    void forward(const std::vector<TensorView>&, const std::vector<TensorView>&, ExecContext&) const override {}
    const float* constant_data() const override { return data.data(); }
    std::size_t param_bytes() const override { return data.size() * sizeof(float); }
};

using Factory = std::function<std::unique_ptr<Layer>()>;

const std::unordered_map<std::string, Factory>& registry() {
    static const std::unordered_map<std::string, Factory> r = {
        {"Input", [] { return std::unique_ptr<Layer>(new InputLayer); }},
        {"Convolution", [] { return std::unique_ptr<Layer>(new ConvolutionLayer); }},
        {"Swish", [] { return std::unique_ptr<Layer>(new UnaryLayer(Activation::Swish)); }},
        {"Sigmoid", [] { return std::unique_ptr<Layer>(new UnaryLayer(Activation::Sigmoid)); }},
        {"ReLU", [] { return std::unique_ptr<Layer>(new UnaryLayer(Activation::ReLU)); }},
        {"Pooling", [] { return std::unique_ptr<Layer>(new PoolingLayer); }},
        {"Interp", [] { return std::unique_ptr<Layer>(new InterpLayer); }},
        {"Concat", [] { return std::unique_ptr<Layer>(new ConcatLayer); }},
        {"Slice", [] { return std::unique_ptr<Layer>(new SliceLayer); }},
        {"Split", [] { return std::unique_ptr<Layer>(new SplitLayer); }},
        {"Reshape", [] { return std::unique_ptr<Layer>(new ReshapeLayer); }},
        {"Permute", [] { return std::unique_ptr<Layer>(new PermuteLayer); }},
        {"Softmax", [] { return std::unique_ptr<Layer>(new SoftmaxLayer); }},
        {"BinaryOp", [] { return std::unique_ptr<Layer>(new BinaryOpLayer); }},
        {"MemoryData", [] { return std::unique_ptr<Layer>(new MemoryDataLayer); }},
    };
    return r;
}

}  // namespace

Activation unary_activation(const Layer& l) {
    if (auto* u = dynamic_cast<const UnaryLayer*>(&l)) return u->activation();
    return Activation::None;
}

bool is_binary_add(const Layer& l) {
    auto* b = dynamic_cast<const BinaryOpLayer*>(&l);
    return b && b->op == 0 && !b->with_scalar;
}

std::unique_ptr<Layer> create_layer(const std::string& type) {
    auto it = registry().find(type);
    if (it == registry().end()) throw std::runtime_error("unsupported layer type: " + type);
    auto l = it->second();
    l->type = type;
    return l;
}
