#pragma once
#include "tensor.hpp"

#include <string>

struct Letterbox {
    float scale = 1.f;
    int left = 0, top = 0, width = 0, height = 0;  // width/height = source image size
};

// Streams a binary PPM (P6) row by row and writes the letterboxed, bilinearly
// resized, [0,1]-normalized CHW image straight into `input` (shape [3,H,W]).
// Only two source rows are held in memory at any time.
Letterbox letterbox_ppm_into(const std::string& path, const TensorView& input);
