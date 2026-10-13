#pragma once
// Graph / engine: loads an ncnn model, optimizes the graph for memory, plans a
// single activation arena and executes.
#include "layer.hpp"

#include <fstream>
#include <memory>
#include <string>
#include <vector>

struct EngineOptions {
    EngineConfig config;           // threads + resident weight format
    bool fuse_activations = true;  // conv + Swish/Sigmoid/ReLU -> one op
    bool fuse_residual = true;     // conv + residual add -> one op
    bool zero_copy_views = true;   // Split / Reshape / channel Slice become views
    bool concat_in_place = true;   // producers write straight into concat outputs
    bool inplace_elementwise = true;
    bool stream_conv_chains = true;  // row-streamed conv->conv chains
    // Activation storage: tensors with >= f16_min_elements elements are stored as
    // fp16 (compute stays fp32).  Graph outputs and small tensors stay fp32.
    bool f16_activations = true;
    std::size_t f16_min_elements = 65536;
    // Storage of the graph input.  U8 (value = byte * input_u8_scale) is used only
    // if every consumer supports it (convolutions); otherwise F16/F32.
    DType input_dtype = DType::F16;
    float input_u8_scale = 1.f / 255.f;
    bool verbose = false;
};

struct MemoryReport {
    std::size_t arena_bytes = 0;        // planned activation + workspace arena
    std::size_t naive_bytes = 0;        // sum of all activation tensors (no reuse)
    std::size_t param_bytes = 0;        // resident weights/constants
    int exec_ops = 0;
};

class Engine {
public:
    explicit Engine(EngineOptions opt = {});
    ~Engine();
    void load(const std::string& param_path, const std::string& bin_path);
    // Optimizes and plans memory for the given input shape; allocates the arena.
    void prepare(const Shape& input_shape);
    TensorView input(int index = 0) const;
    void run();
    TensorView output(const std::string& name) const;
    TensorView output(int index = 0) const;
    const MemoryReport& report() const { return report_; }

private:
    struct Value {
        std::string name;
        Shape shape;
        int producer = -1;
        int alias_of = -1;            // value whose storage contains this one
        std::size_t alias_offset = 0; // element offset inside alias_of
        const float* constant = nullptr;
        bool is_input = false, is_output = false;
        DType dtype = DType::F32;
        float scale = 1.f;
        std::size_t bytes() const { return shape.count() * dtype_size(dtype); }
    };
    struct Op {
        std::shared_ptr<Layer> layer;
        std::vector<int> inputs, outputs;
        bool removed = false;
    };
    struct Root { int value; std::size_t offset; };
    struct Plan {
        std::size_t arena = 0;
        std::vector<int> exec;                 // op indices in execution order
        std::vector<std::size_t> value_offset; // byte offset of arena roots
        std::vector<std::size_t> ws_offset, ws_size;  // per exec op
        std::vector<std::size_t> live;                // live bytes per exec op
        std::size_t area = 0;                         // sum of live bytes over ops (memory x time)
    };

    Root resolve(int v) const;
    bool is_arena_root(int v) const;
    std::vector<Shape> in_shapes(const Op& op) const;
    std::vector<std::vector<int>> consumers() const;
    Plan make_plan() const;
    void pass_fuse();
    void pass_views();
    void pass_concat_in_place();
    void pass_inplace();
    void pass_conv_chains();
    void pass_winograd();
    void pass_chain_rings();
    void pass_dtypes();
    void bind();
    TensorView view_of(int v) const;

    EngineOptions opt_;
    std::unique_ptr<ThreadPool> pool_;
    std::vector<Value> values_;
    std::vector<Op> ops_;
    std::vector<int> inputs_, outputs_;
    Plan plan_;
    unsigned char* arena_ = nullptr;
    std::vector<std::vector<TensorView>> bound_in_, bound_out_;
    MemoryReport report_;
    std::unique_ptr<std::ifstream> weight_file_;  // kept open for streamed weights
    Shape input_shape_;
    bool prepared_ = false;
};
