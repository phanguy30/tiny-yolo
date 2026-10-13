#pragma once
// Branch-free, auto-vectorizable math helpers (no intrinsics -> portable to ARM/x86).
#include <cstdint>
#include <cstring>

namespace fm {

inline float bits_to_float(std::uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
inline std::uint32_t float_to_bits(float f) { std::uint32_t u; std::memcpy(&u, &f, 4); return u; }

// exp(x), Cephes-style range reduction + degree-6 polynomial. ~1-2 ulp for |x| < 87.
inline float exp(float x) {
    x = x > 88.3762626647949f ? 88.3762626647949f : x;
    x = x < -87.3365447504f ? -87.3365447504f : x;
    float fx = x * 1.44269504088896341f;
    // round to nearest (works for |fx| < 2^22)
    float n = (fx + 12582912.f) - 12582912.f;
    float r = x - n * 0.693359375f;
    r = r - n * -2.12194440e-4f;
    float r2 = r * r;
    float p = 1.9875691500E-4f;
    p = p * r + 1.3981999507E-3f;
    p = p * r + 8.3334519073E-3f;
    p = p * r + 4.1665795894E-2f;
    p = p * r + 1.6666665459E-1f;
    p = p * r + 5.0000001201E-1f;
    p = p * r2 + r + 1.f;
    const std::int32_t e = static_cast<std::int32_t>(n) + 127;
    return p * bits_to_float(static_cast<std::uint32_t>(e) << 23);
}

inline float sigmoid(float x) { return 1.f / (1.f + fm::exp(-x)); }
inline float swish(float x) { return x / (1.f + fm::exp(-x)); }

// IEEE half <-> float (round-to-nearest-even), no hardware support required.
inline std::uint16_t float_to_half(float f) {
    std::uint32_t x = float_to_bits(f);
    const std::uint32_t sign = (x >> 16) & 0x8000u;
    x &= 0x7fffffffu;
    if (x >= 0x7f800000u) return static_cast<std::uint16_t>(sign | (x > 0x7f800000u ? 0x7e00u : 0x7c00u));
    if (x >= 0x477ff000u) return static_cast<std::uint16_t>(sign | 0x7c00u);  // overflow -> inf
    if (x < 0x38800000u) {                                                    // subnormal / zero
        if (x < 0x33000000u) return static_cast<std::uint16_t>(sign);
        const std::uint32_t e = x >> 23;
        std::uint32_t m = (x & 0x7fffffu) | 0x800000u;
        const std::uint32_t shift = 126u - e;  // 14..24
        std::uint32_t h = m >> shift;
        const std::uint32_t rem = m & ((1u << shift) - 1u), half = 1u << (shift - 1u);
        if (rem > half || (rem == half && (h & 1u))) ++h;
        return static_cast<std::uint16_t>(sign | h);
    }
    std::uint32_t h = ((x - 0x38000000u) >> 13);
    const std::uint32_t rem = x & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
    return static_cast<std::uint16_t>(sign | h);
}

// Branch-free (vectorizable) float -> half, round-to-nearest-even
// (after F. Giesen's float_to_half_fast3_rtne).
inline std::uint16_t float_to_half_fast(float f) {
    std::uint32_t x = float_to_bits(f);
    const std::uint32_t sign = (x >> 16) & 0x8000u;
    x &= 0x7fffffffu;
    // subnormal / zero results: let the FPU round by adding a magic number
    const float denorm_magic = bits_to_float(((127u - 15u) + (23u - 10u) + 1u) << 23);
    const std::uint32_t sub = float_to_bits(bits_to_float(x) + denorm_magic) - (((127u - 15u) + (23u - 10u) + 1u) << 23);
    // normal results: rebias exponent and round
    const std::uint32_t mant_odd = (x >> 13) & 1u;
    const std::uint32_t nrm = (x + 0xC8000FFFu + mant_odd) >> 13;  // 0xC8000000 = (15-127)<<23
    std::uint32_t r = x < (113u << 23) ? sub : nrm;
    r = x >= ((127u + 16u) << 23) ? (x > 0x7f800000u ? 0x7e00u : 0x7c00u) : r;
    return static_cast<std::uint16_t>(r | sign);
}

// Branch-free (vectorizable) half -> float.
inline float half_to_float(std::uint16_t h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    const std::uint32_t em = static_cast<std::uint32_t>(h & 0x7fffu);
    // normal numbers: rebias exponent
    const std::uint32_t normal = (em << 13) + 0x38000000u;
    // inf / nan
    const std::uint32_t infnan = (em << 13) | 0x7f800000u;
    // subnormals: value = em * 2^-24, computed exactly in float
    const float sub = static_cast<float>(em) * 5.9604644775390625e-8f;
    const std::uint32_t subbits = float_to_bits(sub);
    std::uint32_t r = em >= 0x7c00u ? infnan : normal;
    r = em < 0x0400u ? subbits : r;
    return bits_to_float(r | sign);
}

}  // namespace fm
