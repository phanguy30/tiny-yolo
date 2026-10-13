#pragma once
// Typed load/store helpers so kernels can be written once and instantiated for
// every storage type (fp32 / fp16 / uint8).
#include "fastmath.hpp"
#include "tensor.hpp"

#include <cstring>
#include <stdexcept>

struct F32Tag {
    using T = float;
    static float ld(const T* p, std::size_t i) { return p[i]; }
    static void st(T* p, std::size_t i, float v) { p[i] = v; }
};
struct F16Tag {
    using T = std::uint16_t;
    static float ld(const T* p, std::size_t i) { return fm::half_to_float(p[i]); }
    static void st(T* p, std::size_t i, float v) { p[i] = fm::float_to_half_fast(v); }
};
struct U8Tag {  // raw bytes; callers apply TensorView::scale
    using T = unsigned char;
    static float ld(const T* p, std::size_t i) { return static_cast<float>(p[i]); }
    static void st(T* p, std::size_t i, float v) {
        v = v < 0.f ? 0.f : (v > 255.f ? 255.f : v);
        p[i] = static_cast<unsigned char>(v + 0.5f);
    }
};

template <class F>
void dispatch_dtype(DType d, F&& f) {
    switch (d) {
        case DType::F32: f(F32Tag{}); break;
        case DType::F16: f(F16Tag{}); break;
        case DType::U8: f(U8Tag{}); break;
    }
}

// Float-only dispatch (F32/F16) for layers that do not accept uint8.
template <class F>
void dispatch_float(DType d, F&& f) {
    if (d == DType::F32) f(F32Tag{});
    else if (d == DType::F16) f(F16Tag{});
    else throw std::runtime_error("layer does not support uint8 tensors");
}

template <class F>
void dispatch_float2(DType a, DType b, F&& f) {
    dispatch_float(a, [&](auto ta) { dispatch_float(b, [&](auto tb) { f(ta, tb); }); });
}

// Copies n elements, converting between storage types if necessary.
inline void copy_convert(const void* src, DType sd, void* dst, DType dd, std::size_t n) {
    if (sd == dd) { std::memmove(dst, src, n * dtype_size(sd)); return; }
    dispatch_float2(sd, dd, [&](auto ts, auto td) {
        using TS = decltype(ts);
        using TD = decltype(td);
        const auto* s = static_cast<const typename TS::T*>(src);
        auto* d = static_cast<typename TD::T*>(dst);
        for (std::size_t i = 0; i < n; ++i) TD::st(d, i, TS::ld(s, i));
    });
}
