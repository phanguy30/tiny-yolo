#pragma once
// Streamed convolution chain: executes conv_0 -> conv_1 -> ... -> conv_n-1 band
// by band (rows), keeping each intermediate activation only as a small ring
// buffer of rows instead of a full tensor.  Created by the graph optimizer
// when it lowers the planned peak memory.
#include "conv.hpp"

#include <memory>
#include <vector>

class ConvChainLayer : public Layer {
public:
    std::vector<std::shared_ptr<ConvolutionLayer>> stages;
    static int band_target_pixels;          // pixels per band (parallelism vs ring size)
    static std::size_t ring_budget_bytes;   // cap for each intermediate ring buffer

    // Shapes of each stage's input (index s) and output (index s+1).
    void configure(const Shape& input_shape, int threads);

    std::vector<Shape> infer_shapes(const std::vector<Shape>& in) const override;
    std::size_t workspace_bytes(const std::vector<Shape>& in, const std::vector<Shape>& out, int threads) const override;
    void forward(const std::vector<TensorView>& in, const std::vector<TensorView>& out, ExecContext& ctx) const override;
    std::size_t param_bytes() const override;

private:
    std::vector<Shape> shapes_;  // stage inputs + final output
    std::vector<int> band_;      // rows per band for each stage
    std::vector<int> ring_;      // ring rows for each intermediate (stage s output, s < n-1)
};
