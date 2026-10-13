#include "conv_chain.hpp"

#include <algorithm>
#include <functional>
#include <stdexcept>

int ConvChainLayer::band_target_pixels = 2048;
std::size_t ConvChainLayer::ring_budget_bytes = 128 * 1024;

namespace {
constexpr std::size_t kAlign = 64;
std::size_t align_up(std::size_t v) { return (v + kAlign - 1) / kAlign * kAlign; }
}  // namespace

void ConvChainLayer::configure(const Shape& input_shape, int threads) {
    (void)threads;
    shapes_.assign(1, input_shape);
    for (auto& st : stages) {
        const Shape& s = shapes_.back();
        shapes_.push_back(Shape{st->num_output, st->out_h(s.h()), st->out_w(s.w())});
    }
    const std::size_t n = stages.size();
    band_.assign(n, 1);
    for (std::size_t s = 0; s < n; ++s) {
        const int w = shapes_[s + 1].w(), h = shapes_[s + 1].h();
        band_[s] = std::max(1, std::min(h, (band_target_pixels + w - 1) / w));
    }
    // cap bands so that each ring stays within the budget
    for (std::size_t s = 0; s + 1 < n; ++s) {
        const ConvolutionLayer& next = *stages[s + 1];
        const std::size_t row_bytes = static_cast<std::size_t>(shapes_[s + 1].c()) * shapes_[s + 1].w() * sizeof(float);
        const int max_rows = static_cast<int>(ring_budget / std::max<std::size_t>(row_bytes, 1));
        const int max_band = (max_rows - (next.kh - 1) * next.dh - 1) / next.sh + 1;
        band_[s + 1] = std::max(1, std::min(band_[s + 1], max_band));
    }
    ring_.assign(n ? n - 1 : 0, 0);
    for (std::size_t s = 0; s + 1 < n; ++s) {
        const ConvolutionLayer& next = *stages[s + 1];
        ring_[s] = std::min(shapes_[s + 1].h(), (band_[s + 1] - 1) * next.sh + (next.kh - 1) * next.dh + 1);
    }
}

std::vector<Shape> ConvChainLayer::infer_shapes(const std::vector<Shape>& in) const {
    Shape s = in.at(0);
    std::size_t ri = 1;
    for (auto& st : stages) {
        std::vector<Shape> sin{s};
        if (st->has_residual) sin.push_back(in.at(ri++));
        s = st->infer_shapes(sin)[0];
    }
    return {s};
}

std::size_t ConvChainLayer::workspace_bytes(const std::vector<Shape>&, const std::vector<Shape>&, int threads) const {
    std::size_t b = 0, tws = 0;
    for (auto& st : stages) { b += st->decoded_weight_bytes(); tws = std::max(tws, st->thread_workspace_bytes()); }
    for (std::size_t s = 0; s < ring_.size(); ++s)
        b += align_up(static_cast<std::size_t>(shapes_[s + 1].c()) * static_cast<std::size_t>(ring_[s]) *
                      static_cast<std::size_t>(shapes_[s + 1].w()) * sizeof(float));
    return b + static_cast<std::size_t>(threads) * tws;
}

std::size_t ConvChainLayer::param_bytes() const {
    std::size_t b = 0;
    for (auto& st : stages) b += st->param_bytes();
    return b;
}

void ConvChainLayer::forward(const std::vector<TensorView>& in, const std::vector<TensorView>& out, ExecContext& ctx) const {
    const std::size_t n = stages.size();
    auto* ws = static_cast<unsigned char*>(ctx.workspace);
    std::vector<const float*> w(n);
    for (std::size_t s = 0; s < n; ++s) {
        w[s] = stages[s]->prepare_weights(reinterpret_cast<float*>(ws), ctx);
        ws += stages[s]->decoded_weight_bytes();
    }
    // planes: 0 = chain input, s+1 = output of stage s (rings for intermediates)
    std::vector<Plane> planes(n + 1);
    planes[0] = Plane::dense(in[0]);
    for (std::size_t s = 0; s + 1 < n; ++s) {
        Plane& p = planes[s + 1];
        const Shape& sh = shapes_[s + 1];
        p.base = ws;
        p.dtype = DType::F32;  // ring buffers stay fp32 (small, avoids extra rounding)
        p.C = sh.c(); p.H = sh.h(); p.W = sh.w();
        p.ring = ring_[s];
        p.cstride = static_cast<std::size_t>(p.ring) * static_cast<std::size_t>(p.W);
        ws += align_up(static_cast<std::size_t>(p.C) * p.cstride * sizeof(float));
    }
    planes[n] = Plane::dense(out[0]);
    std::vector<Plane> res(n);
    std::vector<bool> has_res(n, false);
    for (std::size_t s = 0, ri = 1; s < n; ++s)
        if (stages[s]->has_residual) { res[s] = Plane::dense(in[ri]); has_res[s] = true; ++ri; }

    std::size_t tws = 0;
    for (auto& st : stages) tws = std::max(tws, st->thread_workspace_bytes());

    std::vector<int> done(n, 0);
    std::function<void(std::size_t, int)> produce = [&](std::size_t s, int rows) {
        while (done[s] < rows) {
            const int end = std::min(done[s] + band_[s], rows);
            if (s > 0) produce(s - 1, std::min(planes[s].H, stages[s]->input_row_end(end)));
            stages[s]->run_rows(w[s], planes[s], planes[s + 1], has_res[s] ? &res[s] : nullptr, done[s], end, ws, tws,
                                *ctx.pool);
            done[s] = end;
        }
    };
    produce(n - 1, planes[n].H);
}
