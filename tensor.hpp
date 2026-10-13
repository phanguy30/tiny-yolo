#pragma once
// Lightweight shape + non-owning tensor view.  All activation memory is owned by
// the engine's single planned arena; tensors are just (pointer, shape) pairs.
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

struct Shape {
    static constexpr int kMaxDims = 4;
    int ndim = 0;
    std::array<int, kMaxDims> d{{1, 1, 1, 1}};

    Shape() = default;
    Shape(std::initializer_list<int> l) {
        for (int v : l) d[static_cast<std::size_t>(ndim++)] = v;
    }
    static Shape from(const std::vector<int>& v) {
        Shape s;
        for (int x : v) s.d[static_cast<std::size_t>(s.ndim++)] = x;
        return s;
    }
    std::size_t count() const {
        std::size_t n = 1;
        for (int i = 0; i < ndim; ++i) n *= static_cast<std::size_t>(d[static_cast<std::size_t>(i)]);
        return ndim ? n : 0;
    }
    int operator[](int i) const { return d[static_cast<std::size_t>(i)]; }
    int& operator[](int i) { return d[static_cast<std::size_t>(i)]; }
    // CHW accessors (missing leading dims are treated as 1)
    int c() const { return ndim >= 3 ? d[static_cast<std::size_t>(ndim - 3)] : 1; }
    int h() const { return ndim >= 2 ? d[static_cast<std::size_t>(ndim - 2)] : 1; }
    int w() const { return ndim >= 1 ? d[static_cast<std::size_t>(ndim - 1)] : 1; }
    bool operator==(const Shape& o) const {
        if (ndim != o.ndim) return false;
        for (int i = 0; i < ndim; ++i) if (d[static_cast<std::size_t>(i)] != o.d[static_cast<std::size_t>(i)]) return false;
        return true;
    }
    bool operator!=(const Shape& o) const { return !(*this == o); }
    std::string str() const {
        std::string s = "[";
        for (int i = 0; i < ndim; ++i) s += (i ? "," : "") + std::to_string(d[static_cast<std::size_t>(i)]);
        return s + "]";
    }
};

// Storage type of a tensor.  Compute is always fp32; storage may be fp16
// (activations) or uint8 (network input, value = byte * scale).
enum class DType { F32, F16, U8 };

inline std::size_t dtype_size(DType d) { return d == DType::F32 ? 4 : (d == DType::F16 ? 2 : 1); }
inline const char* dtype_name(DType d) { return d == DType::F32 ? "f32" : (d == DType::F16 ? "f16" : "u8"); }

struct TensorView {
    void* data = nullptr;
    Shape shape;
    DType dtype = DType::F32;
    float scale = 1.f;  // dequantization scale for U8
    std::size_t size() const { return shape.count(); }
    std::size_t bytes() const { return size() * dtype_size(dtype); }
    float* f32() const { return static_cast<float*>(data); }
    std::uint16_t* f16() const { return static_cast<std::uint16_t*>(data); }
    unsigned char* u8() const { return static_cast<unsigned char*>(data); }
};
