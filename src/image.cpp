#include "image.hpp"

#include "dtype.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace {
int read_header_int(std::istream& in) {
    int c = in.peek();
    while (in && (std::isspace(c) || c == '#')) {
        if (c == '#') { std::string skip; std::getline(in, skip); }
        else in.get();
        c = in.peek();
    }
    int v = -1;
    in >> v;
    return v;
}
}  // namespace

Letterbox letterbox_ppm_into(const std::string& path, const TensorView& input) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open image: " + path);
    std::string magic;
    file >> magic;
    const int iw = read_header_int(file), ih = read_header_int(file), maxv = read_header_int(file);
    if (!file || magic != "P6" || maxv != 255 || iw <= 0 || ih <= 0) throw std::runtime_error("bad PPM image: " + path);
    file.get();  // single whitespace after header

    const int dst_h = input.shape.h(), dst_w = input.shape.w();
    Letterbox box;
    box.scale = std::min(static_cast<float>(dst_w) / iw, static_cast<float>(dst_h) / ih);
    const int width = static_cast<int>(std::round(iw * box.scale));
    const int height = static_cast<int>(std::round(ih * box.scale));
    box.left = (dst_w - width) / 2;
    box.top = (dst_h - height) / 2;
    box.width = iw;
    box.height = ih;

    const std::size_t plane = static_cast<std::size_t>(dst_h) * dst_w;
    // The engine may store its input as fp32, fp16 or uint8 (value = byte * scale).
    const bool u8 = input.dtype == DType::U8;
    if (u8) std::fill(input.u8(), input.u8() + 3 * plane, static_cast<unsigned char>(114));
    else if (input.dtype == DType::F16) std::fill(input.f16(), input.f16() + 3 * plane, fm::float_to_half(114.f / 255.f));
    else std::fill(input.f32(), input.f32() + 3 * plane, 114.f / 255.f);

    // per-column sampling coordinates
    std::vector<int> x0s(static_cast<std::size_t>(width)), x1s(static_cast<std::size_t>(width));
    std::vector<float> dxs(static_cast<std::size_t>(width));
    for (int x = 0; x < width; ++x) {
        float sx = (x + .5f) / box.scale - .5f;
        sx = std::clamp(sx, 0.f, float(iw - 1));
        const int x0 = int(sx);
        x0s[static_cast<std::size_t>(x)] = x0;
        x1s[static_cast<std::size_t>(x)] = std::min(x0 + 1, iw - 1);
        dxs[static_cast<std::size_t>(x)] = sx - x0;
    }

    std::vector<float> vals(static_cast<std::size_t>(width));  // one interpolated output row
    const std::size_t row_bytes = static_cast<std::size_t>(iw) * 3;
    std::vector<unsigned char> rows[2] = {std::vector<unsigned char>(row_bytes), std::vector<unsigned char>(row_bytes)};
    int row_id[2] = {-1, -1};
    int next_row = 0;  // next row to be read from the file
    auto fetch = [&](int r) -> const unsigned char* {  // rows are requested in non-decreasing order
        for (int k = 0; k < 2; ++k) if (row_id[k] == r) return rows[k].data();
        while (next_row <= r) {
            const int slot = row_id[0] < row_id[1] ? 0 : 1;  // evict the older row
            if (next_row < r - 1) {
                file.ignore(static_cast<std::streamsize>(row_bytes));
            } else {
                file.read(reinterpret_cast<char*>(rows[slot].data()), static_cast<std::streamsize>(row_bytes));
                row_id[slot] = next_row;
            }
            if (!file) throw std::runtime_error("truncated PPM image: " + path);
            ++next_row;
        }
        for (int k = 0; k < 2; ++k) if (row_id[k] == r) return rows[k].data();
        throw std::runtime_error("PPM row cache error");
    };

    for (int y = 0; y < height; ++y) {
        float sy = (y + .5f) / box.scale - .5f;
        sy = std::clamp(sy, 0.f, float(ih - 1));
        const int y0 = int(sy), y1 = std::min(y0 + 1, ih - 1);
        const float dy = sy - y0;
        const unsigned char* r0 = fetch(y0);
        const unsigned char* r1 = fetch(y1);
        for (int c = 0; c < 3; ++c) {
            const std::size_t off = static_cast<std::size_t>(c) * plane + static_cast<std::size_t>(y + box.top) * dst_w + box.left;
            for (int x = 0; x < width; ++x) {
                const int x0 = x0s[static_cast<std::size_t>(x)] * 3 + c, x1 = x1s[static_cast<std::size_t>(x)] * 3 + c;
                const float dx = dxs[static_cast<std::size_t>(x)];
                const float a = float(r0[x0]) * (1 - dx) + float(r0[x1]) * dx;
                const float b = float(r1[x0]) * (1 - dx) + float(r1[x1]) * dx;
                vals[static_cast<std::size_t>(x)] = a * (1 - dy) + b * dy;
            }
            if (u8) {
                for (int x = 0; x < width; ++x)  // like cv2.resize on uint8
                    input.u8()[off + x] = static_cast<unsigned char>(std::min(255.f, std::nearbyint(vals[static_cast<std::size_t>(x)])));
            } else {
                for (int x = 0; x < width; ++x) vals[static_cast<std::size_t>(x)] /= 255.f;
                if (input.dtype == DType::F16) fm::float_to_half_n(vals.data(), input.f16() + off, width);
                else std::copy(vals.begin(), vals.begin() + width, input.f32() + off);
            }
        }
    }
    return box;
}
