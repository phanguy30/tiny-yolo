#include "graph.hpp"

#include "conv.hpp"
#include "conv_chain.hpp"
#include "layers.hpp"
#include "memory_planner.hpp"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <new>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace {
std::vector<std::shared_ptr<ConvolutionLayer>> conv_stages(const std::shared_ptr<Layer>& l) {
    if (auto c = std::dynamic_pointer_cast<ConvolutionLayer>(l)) return {c};
    if (auto ch = std::dynamic_pointer_cast<ConvChainLayer>(l)) return ch->stages;
    return {};
}
}  // namespace

Engine::Engine(EngineOptions opt) : opt_(opt), pool_(new ThreadPool(opt.config.threads)) {}

Engine::~Engine() {
    if (arena_) ::operator delete(arena_, std::align_val_t(64));
}

// ------------------------------------------------------------------ loading
void Engine::load(const std::string& param_path, const std::string& bin_path) {
    std::ifstream pf(param_path);
    if (!pf) throw std::runtime_error("cannot open param file: " + param_path);
    int magic = 0, layer_count = 0, blob_count = 0;
    pf >> magic >> layer_count >> blob_count;
    if (magic != 7767517) throw std::runtime_error("unsupported param file (bad magic)");
    std::unordered_map<std::string, int> blob_ids;
    std::string line;
    std::getline(pf, line);
    while (std::getline(pf, line)) {
        if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
        std::istringstream ss(line);
        std::string type, name;
        int nb = 0, nt = 0;
        ss >> type >> name >> nb >> nt;
        Op op;
        op.layer = std::shared_ptr<Layer>(create_layer(type));
        op.layer->name = name;
        op.layer->num_outputs = nt;
        std::string token;
        for (int i = 0; i < nb; ++i) {
            ss >> token;
            auto it = blob_ids.find(token);
            if (it == blob_ids.end()) throw std::runtime_error("layer " + name + " uses unknown blob " + token);
            op.inputs.push_back(it->second);
        }
        for (int i = 0; i < nt; ++i) {
            ss >> token;
            Value v;
            v.name = token;
            v.producer = static_cast<int>(ops_.size());
            v.is_input = type == "Input";
            blob_ids[token] = static_cast<int>(values_.size());
            if (v.is_input) inputs_.push_back(static_cast<int>(values_.size()));
            op.outputs.push_back(static_cast<int>(values_.size()));
            values_.push_back(v);
        }
        ParamDict pd;
        while (ss >> token) pd.parse_token(token);
        op.layer->load_param(pd);
        if (type == "Input") op.removed = true;  // input is written by the caller
        ops_.push_back(std::move(op));
    }
    // graph outputs = blobs nobody consumes
    std::vector<bool> used(values_.size(), false);
    for (auto& op : ops_)
        for (int v : op.inputs) used[static_cast<std::size_t>(v)] = true;
    for (std::size_t v = 0; v < values_.size(); ++v)
        if (!used[v]) { values_[v].is_output = true; outputs_.push_back(static_cast<int>(v)); }

    std::ifstream bf(bin_path, std::ios::binary);
    if (!bf) throw std::runtime_error("cannot open bin file: " + bin_path);
    ModelBin mb(bf);
    for (auto& op : ops_) {
        op.layer->load_model(mb, opt_.config);
        if (const float* c = op.layer->constant_data()) values_[static_cast<std::size_t>(op.outputs[0])].constant = c;
    }
    for (auto& op : ops_) report_.param_bytes += op.layer->param_bytes();
    if (opt_.config.weight_format == WeightFormat::Stream) {
        weight_file_.reset(new std::ifstream(bin_path, std::ios::binary));
        // small stream buffer: we only read whole layers sequentially
        if (!*weight_file_) throw std::runtime_error("cannot reopen bin file: " + bin_path);
    }
}

// ------------------------------------------------------------------ helpers
Engine::Root Engine::resolve(int v) const {
    std::size_t off = 0;
    while (values_[static_cast<std::size_t>(v)].alias_of >= 0) {
        off += values_[static_cast<std::size_t>(v)].alias_offset;
        v = values_[static_cast<std::size_t>(v)].alias_of;
    }
    return {v, off};
}

bool Engine::is_arena_root(int v) const {
    const Value& val = values_[static_cast<std::size_t>(v)];
    return val.alias_of < 0 && val.constant == nullptr;
}

std::vector<Shape> Engine::in_shapes(const Op& op) const {
    std::vector<Shape> s;
    for (int v : op.inputs) s.push_back(values_[static_cast<std::size_t>(v)].shape);
    return s;
}

std::vector<std::vector<int>> Engine::consumers() const {
    std::vector<std::vector<int>> c(values_.size());
    for (std::size_t i = 0; i < ops_.size(); ++i) {
        if (ops_[i].removed) continue;
        for (int v : ops_[i].inputs) c[static_cast<std::size_t>(v)].push_back(static_cast<int>(i));
    }
    return c;
}

// ------------------------------------------------------------------ planning
Engine::Plan Engine::make_plan() const {
    Plan plan;
    std::vector<int> pos(ops_.size(), -1);
    for (std::size_t i = 0; i < ops_.size(); ++i)
        if (!ops_[i].removed) { pos[i] = static_cast<int>(plan.exec.size()); plan.exec.push_back(static_cast<int>(i)); }
    const int end = static_cast<int>(plan.exec.size());
    const std::size_t nv = values_.size();
    std::vector<int> def(nv, INT_MAX), last(nv, -1);
    for (std::size_t v = 0; v < nv; ++v) {
        const Value& val = values_[v];
        if (val.is_input) def[v] = 0;
        else if (val.producer >= 0 && pos[static_cast<std::size_t>(val.producer)] >= 0) {
            const auto& outs = ops_[static_cast<std::size_t>(val.producer)].outputs;
            if (std::find(outs.begin(), outs.end(), static_cast<int>(v)) != outs.end()) def[v] = pos[static_cast<std::size_t>(val.producer)];
        }
        if (val.is_output) last[v] = end;
    }
    for (int i : plan.exec)
        for (int v : ops_[static_cast<std::size_t>(i)].inputs) last[static_cast<std::size_t>(v)] = std::max(last[static_cast<std::size_t>(v)], pos[static_cast<std::size_t>(i)]);
    std::vector<int> rdef(nv, INT_MAX), rlast(nv, -1);
    for (std::size_t v = 0; v < nv; ++v) {
        if (def[v] == INT_MAX && last[v] < 0) continue;  // dead value
        const int r = resolve(static_cast<int>(v)).value;
        if (!is_arena_root(r)) continue;
        rdef[static_cast<std::size_t>(r)] = std::min(rdef[static_cast<std::size_t>(r)], def[v]);
        rlast[static_cast<std::size_t>(r)] = std::max({rlast[static_cast<std::size_t>(r)], last[v], def[v] == INT_MAX ? -1 : def[v]});
    }
    std::vector<PlanBuffer> bufs;
    std::vector<int> buf_value;
    for (std::size_t v = 0; v < nv; ++v) {
        if (rdef[v] == INT_MAX) {
            if (rlast[v] >= 0) throw std::runtime_error("planner: value " + values_[v].name + " is used but never produced");
            continue;
        }
        PlanBuffer b;
        b.size = values_[v].bytes();
        b.first = rdef[v];
        b.last = rlast[v];
        bufs.push_back(b);
        buf_value.push_back(static_cast<int>(v));
    }
    const std::size_t nvalue_bufs = bufs.size();
    for (int p = 0; p < end; ++p) {
        const Op& op = ops_[static_cast<std::size_t>(plan.exec[static_cast<std::size_t>(p)])];
        std::vector<Shape> outs;
        for (int v : op.outputs) outs.push_back(values_[static_cast<std::size_t>(v)].shape);
        PlanBuffer b;
        b.size = op.layer->workspace_bytes(in_shapes(op), outs, pool_->size());
        b.first = b.last = p;
        bufs.push_back(b);
    }
    plan.arena = plan_buffers(bufs);
    plan.live.assign(static_cast<std::size_t>(end), 0);
    for (const auto& b : bufs)
        for (int p = std::max(b.first, 0); p <= std::min(b.last, end - 1); ++p) plan.live[static_cast<std::size_t>(p)] += b.size;
    plan.area = 0;
    for (std::size_t l : plan.live) plan.area += l;
    plan.value_offset.assign(nv, 0);
    for (std::size_t k = 0; k < nvalue_bufs; ++k) plan.value_offset[static_cast<std::size_t>(buf_value[k])] = bufs[k].offset;
    for (int p = 0; p < end; ++p) {
        plan.ws_offset.push_back(bufs[nvalue_bufs + static_cast<std::size_t>(p)].offset);
        plan.ws_size.push_back(bufs[nvalue_bufs + static_cast<std::size_t>(p)].size);
    }
    return plan;
}

// ------------------------------------------------------------------ passes
void Engine::pass_fuse() {
    auto cons = consumers();
    for (std::size_t i = 0; i < ops_.size(); ++i) {
        Op& op = ops_[i];
        if (op.removed) continue;
        const Activation act = unary_activation(*op.layer);
        if (opt_.fuse_activations && act != Activation::None && op.inputs.size() == 1) {
            const int v = op.inputs[0];
            const int p = values_[static_cast<std::size_t>(v)].producer;
            auto conv = p >= 0 ? std::dynamic_pointer_cast<ConvolutionLayer>(ops_[static_cast<std::size_t>(p)].layer) : nullptr;
            if (conv && !ops_[static_cast<std::size_t>(p)].removed && conv->act == Activation::None && !conv->has_residual &&
                cons[static_cast<std::size_t>(v)].size() == 1 && !values_[static_cast<std::size_t>(v)].is_output) {
                conv->act = act;
                ops_[static_cast<std::size_t>(p)].outputs[0] = op.outputs[0];
                values_[static_cast<std::size_t>(op.outputs[0])].producer = p;
                op.removed = true;
                cons[static_cast<std::size_t>(v)].clear();
                values_[static_cast<std::size_t>(v)].producer = -1;  // value no longer exists
                continue;
            }
        }
        if (opt_.fuse_residual && is_binary_add(*op.layer) && op.inputs.size() == 2 && op.inputs[0] != op.inputs[1] &&
            values_[static_cast<std::size_t>(op.inputs[0])].shape == values_[static_cast<std::size_t>(op.inputs[1])].shape) {
            for (int side = 0; side < 2; ++side) {
                const int u = op.inputs[static_cast<std::size_t>(side)], r = op.inputs[static_cast<std::size_t>(1 - side)];
                const int p = values_[static_cast<std::size_t>(u)].producer;
                auto conv = p >= 0 ? std::dynamic_pointer_cast<ConvolutionLayer>(ops_[static_cast<std::size_t>(p)].layer) : nullptr;
                if (!conv || ops_[static_cast<std::size_t>(p)].removed || conv->has_residual) continue;
                if (cons[static_cast<std::size_t>(u)].size() != 1 || values_[static_cast<std::size_t>(u)].is_output) continue;
                if (values_[static_cast<std::size_t>(r)].producer >= p) continue;  // residual must exist before the conv runs
                conv->has_residual = true;
                ops_[static_cast<std::size_t>(p)].inputs.push_back(r);
                ops_[static_cast<std::size_t>(p)].outputs[0] = op.outputs[0];
                values_[static_cast<std::size_t>(op.outputs[0])].producer = p;
                op.removed = true;
                cons[static_cast<std::size_t>(u)].clear();
                values_[static_cast<std::size_t>(u)].producer = -1;  // value no longer exists
                break;
            }
        }
    }
}

void Engine::pass_views() {
    for (auto& op : ops_) {
        if (op.removed || op.inputs.empty()) continue;
        const auto shapes = in_shapes(op);
        std::vector<std::size_t> offs(op.outputs.size());
        bool all = true;
        for (std::size_t k = 0; k < op.outputs.size() && all; ++k)
            all = op.layer->view_output(shapes, static_cast<int>(k), offs[k]);
        if (!all) continue;
        for (std::size_t k = 0; k < op.outputs.size(); ++k) {
            values_[static_cast<std::size_t>(op.outputs[k])].alias_of = op.inputs[0];
            values_[static_cast<std::size_t>(op.outputs[k])].alias_offset = offs[k];
        }
        op.removed = true;
    }
}

void Engine::pass_concat_in_place() {
    Plan p0 = make_plan();
    std::pair<std::size_t, std::size_t> best{p0.arena, p0.area};
    for (auto& op : ops_) {
        if (op.removed) continue;
        const auto shapes = in_shapes(op);
        for (std::size_t k = 0; k < op.inputs.size(); ++k) {
            std::size_t off = 0;
            if (!op.layer->concat_slot(shapes, static_cast<int>(k), off)) continue;
            const Root rx = resolve(op.inputs[k]);
            const Root rc = resolve(op.outputs[0]);
            const Value& R = values_[static_cast<std::size_t>(rx.value)];
            if (rx.value == rc.value || !is_arena_root(rx.value) || R.is_input || R.is_output) continue;
            if (R.dtype != values_[static_cast<std::size_t>(rc.value)].dtype) continue;  // storage types must match
            if (off < rx.offset) continue;
            const std::size_t start = off - rx.offset, rsize = R.shape.count();
            if (start + rsize > values_[static_cast<std::size_t>(op.outputs[0])].shape.count()) continue;
            bool ok = true;  // R may only cover slots whose members are consistent views of R
            for (std::size_t m = 0; m < op.inputs.size() && ok; ++m) {
                if (m == k) continue;
                std::size_t offm = 0;
                op.layer->concat_slot(shapes, static_cast<int>(m), offm);
                const std::size_t sm = shapes[m].count();
                if (offm < start + rsize && start < offm + sm) {
                    const Root ry = resolve(op.inputs[m]);
                    if (ry.value != rx.value || ry.offset + start != offm) ok = false;
                }
            }
            if (!ok) continue;
            values_[static_cast<std::size_t>(rx.value)].alias_of = op.outputs[0];
            values_[static_cast<std::size_t>(rx.value)].alias_offset = start;
            const Plan p = make_plan();
            const std::pair<std::size_t, std::size_t> cost{p.arena, p.area};
            if (cost <= best) {  // never grow the arena; on ties prefer less memory x time
                best = cost;
            } else {
                values_[static_cast<std::size_t>(rx.value)].alias_of = -1;
                values_[static_cast<std::size_t>(rx.value)].alias_offset = 0;
            }
        }
    }
}

void Engine::pass_inplace() {
    std::vector<int> pos(ops_.size(), -1);
    int n = 0;
    for (std::size_t i = 0; i < ops_.size(); ++i)
        if (!ops_[i].removed) pos[i] = n++;
    std::vector<int> last(values_.size(), -1);
    for (std::size_t i = 0; i < ops_.size(); ++i)
        if (!ops_[i].removed)
            for (int v : ops_[i].inputs) last[static_cast<std::size_t>(v)] = std::max(last[static_cast<std::size_t>(v)], pos[i]);
    for (std::size_t v = 0; v < values_.size(); ++v)
        if (values_[v].is_output) last[v] = INT_MAX;

    for (std::size_t i = 0; i < ops_.size(); ++i) {
        Op& op = ops_[i];
        if (op.removed || !op.layer->inplace_capable() || op.inputs.empty()) continue;
        const int y = op.outputs[0];
        if (values_[static_cast<std::size_t>(y)].alias_of >= 0) continue;
        // Any full-size input may be overwritten: in-place-capable layers are elementwise, so
        // output element i depends only on element i of each full-size input. This lets
        // e.g. `constant - x` write into x when the first operand is a constant.
        for (std::size_t t = 0; t < op.inputs.size(); ++t) {
            const int a = op.inputs[t];
            const Root ra = resolve(a);
            if (!is_arena_root(ra.value) || values_[static_cast<std::size_t>(ra.value)].is_input) continue;
            const std::size_t na = values_[static_cast<std::size_t>(a)].shape.count();
            if (values_[static_cast<std::size_t>(y)].shape.count() != na) continue;
            if (values_[static_cast<std::size_t>(y)].dtype != values_[static_cast<std::size_t>(ra.value)].dtype) continue;
            bool ok = true;
            for (std::size_t k = 0; k < op.inputs.size(); ++k)
                if (k != t && op.inputs[k] != a && resolve(op.inputs[k]).value == ra.value) ok = false;
            // nobody may read the overwritten region after this op
            for (std::size_t v = 0; v < values_.size() && ok; ++v) {
                const Root rv = resolve(static_cast<int>(v));
                if (rv.value != ra.value) continue;
                const std::size_t nvv = values_[v].shape.count();
                if (rv.offset < ra.offset + na && ra.offset < rv.offset + nvv && last[v] > pos[i]) ok = false;
            }
            if (!ok) continue;
            values_[static_cast<std::size_t>(y)].alias_of = a;
            values_[static_cast<std::size_t>(y)].alias_offset = 0;
            break;
        }
    }
}

void Engine::pass_dtypes() {
    for (auto& v : values_) {
        if (v.constant) continue;
        v.dtype = (opt_.f16_activations && !v.is_output && v.shape.count() >= opt_.f16_min_elements) ? DType::F16 : DType::F32;
    }
    // graph input
    const int in = inputs_[0];
    DType want = opt_.input_dtype;
    if (want == DType::U8) {
        bool ok = true;
        for (std::size_t v = 0; v < values_.size(); ++v)
            if (values_[v].alias_of == in) ok = false;  // views of the input must stay float
        for (const auto& op : ops_) {
            if (op.removed) continue;
            for (std::size_t k = 0; k < op.inputs.size(); ++k)
                if (op.inputs[k] == in && (k != 0 || conv_stages(op.layer).empty())) ok = false;  // only conv data inputs read uint8
        }
        if (!ok) want = DType::F16;
    }
    values_[static_cast<std::size_t>(in)].dtype = want;
    values_[static_cast<std::size_t>(in)].scale = want == DType::U8 ? opt_.input_u8_scale : 1.f;
}

// Winograd convolutions need more scratch (16 transformed weights per 3x3 filter, and
// transformed input tiles per thread), so they must not raise the planned peak: starting
// from the plan without Winograd, layers are switched to it one at a time (smallest
// scratch first, i.e. the high-resolution layers that do most of the work) and each is
// kept only if the arena does not grow.
void Engine::pass_winograd() {
    std::vector<std::shared_ptr<ConvolutionLayer>> convs;
    for (const Op& op : ops_)
        if (!op.removed)
            for (auto& c : conv_stages(op.layer))
                if (c->winograd) { c->winograd = false; convs.push_back(c); }
    if (convs.empty()) return;
    auto scratch = [](const ConvolutionLayer& c) {
        return c.winograd_weight_count() * sizeof(float) + 16 * static_cast<std::size_t>(c.in_channels) * ConvolutionLayer::NR * sizeof(float);
    };
    std::stable_sort(convs.begin(), convs.end(), [&](const auto& x, const auto& y) { return scratch(*x) < scratch(*y); });
    const std::size_t limit = make_plan().arena;
    for (auto& c : convs) {
        c->winograd = true;
        if (make_plan().arena > limit) c->winograd = false;
    }
}

// Small rings mean short bands, and short bands mean few work items per band for the
// thread pool. Each chain's ring budget is doubled while the planned arena stays the same.
void Engine::pass_chain_rings() {
    const std::size_t limit = make_plan().arena;
    for (const Op& op : ops_) {
        auto ch = op.removed ? nullptr : std::dynamic_pointer_cast<ConvChainLayer>(op.layer);
        if (!ch) continue;
        for (int step = 0; step < 4; ++step) {
            const std::size_t prev = ch->ring_budget;
            ch->ring_budget = prev * 2;
            ch->configure();
            if (make_plan().arena > limit) {
                ch->ring_budget = prev;
                ch->configure();
                break;
            }
        }
    }
}

void Engine::pass_conv_chains() {
    Plan p0 = make_plan();
    std::pair<std::size_t, std::size_t> best{p0.arena, p0.area};
    bool improved = true;
    while (improved) {
        improved = false;
        const auto cons = consumers();
        for (std::size_t i = 0; i < ops_.size() && !improved; ++i) {
            if (ops_[i].removed) continue;
            auto first = conv_stages(ops_[i].layer);
            if (first.empty()) continue;
            const int v = ops_[i].outputs[0];
            const Value& val = values_[static_cast<std::size_t>(v)];
            if (val.is_output || val.alias_of >= 0 || cons[static_cast<std::size_t>(v)].size() != 1) continue;
            bool has_alias = false;
            for (std::size_t w = 0; w < values_.size() && !has_alias; ++w)
                if (static_cast<int>(w) != v && resolve(static_cast<int>(w)).value == v) has_alias = true;
            if (has_alias) continue;
            const std::size_t j = static_cast<std::size_t>(cons[static_cast<std::size_t>(v)][0]);
            auto second = conv_stages(ops_[j].layer);
            if (second.empty() || ops_[j].inputs[0] != v) continue;
            if (std::count(ops_[j].inputs.begin(), ops_[j].inputs.end(), v) != 1) continue;

            auto chain = std::make_shared<ConvChainLayer>();
            chain->type = "ConvChain";
            chain->name = ops_[i].layer->name + "+" + ops_[j].layer->name;
            chain->stages = first;
            chain->stages.insert(chain->stages.end(), second.begin(), second.end());
            chain->configure(values_[static_cast<std::size_t>(ops_[i].inputs[0])].shape, pool_->size());
            Op fused;
            fused.layer = chain;
            fused.inputs = ops_[i].inputs;
            fused.inputs.insert(fused.inputs.end(), ops_[j].inputs.begin() + 1, ops_[j].inputs.end());
            fused.outputs = ops_[j].outputs;

            Op saved_i = ops_[i], saved_j = ops_[j];
            ops_[i].removed = true;
            ops_[j] = fused;
            const Plan p = make_plan();
            const std::pair<std::size_t, std::size_t> cost{p.arena, p.area};
            if (cost < best) {
                best = cost;
                improved = true;
            } else {
                ops_[i] = saved_i;
                ops_[j] = saved_j;
            }
        }
    }
}

// ------------------------------------------------------------------ prepare / run
void Engine::prepare(const Shape& input_shape) {
    if (prepared_) throw std::runtime_error("Engine::prepare may only be called once");
    if (inputs_.size() != 1) throw std::runtime_error("engine currently expects exactly one Input layer");
    input_shape_ = input_shape;
    // shape inference
    for (auto& op : ops_) {
        if (op.layer->type == "Input") {
            values_[static_cast<std::size_t>(op.outputs[0])].shape = input_shape;
            continue;
        }
        auto shapes = op.layer->infer_shapes(in_shapes(op));
        for (std::size_t k = 0; k < op.outputs.size(); ++k) values_[static_cast<std::size_t>(op.outputs[k])].shape = shapes.at(k);
    }
    for (const auto& v : values_)
        if (!v.constant) report_.naive_bytes += v.shape.count() * sizeof(float);

    pass_fuse();
    if (opt_.zero_copy_views) pass_views();
    pass_dtypes();
    if (opt_.concat_in_place) pass_concat_in_place();
    if (opt_.inplace_elementwise) pass_inplace();
    if (opt_.stream_conv_chains) pass_conv_chains();
    pass_winograd();
    if (opt_.stream_conv_chains) pass_chain_rings();

    plan_ = make_plan();
    report_.arena_bytes = plan_.arena;
    report_.exec_ops = static_cast<int>(plan_.exec.size());
    arena_ = static_cast<unsigned char*>(::operator new(std::max<std::size_t>(plan_.arena, 64), std::align_val_t(64)));
    bind();
    prepared_ = true;

    if (opt_.verbose) {
        std::fprintf(stderr, "%-28s %-12s %-16s %10s %10s\n", "op", "type", "output", "workspace", "live_KiB");
        for (std::size_t p = 0; p < plan_.exec.size(); ++p) {
            const Op& op = ops_[static_cast<std::size_t>(plan_.exec[p])];
            std::fprintf(stderr, "%-28s %-12s %-16s %10zu %10zu\n", op.layer->name.c_str(), op.layer->type.c_str(),
                         (values_[static_cast<std::size_t>(op.outputs[0])].shape.str() + dtype_name(view_of(op.outputs[0]).dtype)).c_str(),
                         plan_.ws_size[p], plan_.live[p] / 1024);
        }
    }
}

TensorView Engine::view_of(int v) const {
    const Root r = resolve(v);
    const Value& rv = values_[static_cast<std::size_t>(r.value)];
    TensorView t;
    t.shape = values_[static_cast<std::size_t>(v)].shape;
    if (rv.constant) {
        t.data = const_cast<float*>(rv.constant) + r.offset;
        return t;
    }
    t.dtype = rv.dtype;
    t.scale = rv.scale;
    t.data = arena_ + plan_.value_offset[static_cast<std::size_t>(r.value)] + r.offset * dtype_size(rv.dtype);
    return t;
}

void Engine::bind() {
    bound_in_.clear();
    bound_out_.clear();
    for (int i : plan_.exec) {
        const Op& op = ops_[static_cast<std::size_t>(i)];
        std::vector<TensorView> in, out;
        for (int v : op.inputs) in.push_back(view_of(v));
        for (int v : op.outputs) out.push_back(view_of(v));
        bound_in_.push_back(std::move(in));
        bound_out_.push_back(std::move(out));
    }
}

TensorView Engine::input(int index) const { return view_of(inputs_.at(static_cast<std::size_t>(index))); }

TensorView Engine::output(int index) const { return view_of(outputs_.at(static_cast<std::size_t>(index))); }

TensorView Engine::output(const std::string& name) const {
    for (std::size_t v = 0; v < values_.size(); ++v) {
        if (values_[v].name != name) continue;
        if (!values_[v].is_output) throw std::runtime_error("blob " + name + " is not a graph output");
        return view_of(static_cast<int>(v));
    }
    throw std::runtime_error("unknown output blob " + name);
}

void Engine::run() {
    if (!prepared_) throw std::runtime_error("Engine::run called before prepare");
    ExecContext ctx;
    ctx.pool = pool_.get();
    ctx.weight_file = weight_file_.get();
    static const bool profile = std::getenv("TINY_YOLO_PROFILE") != nullptr;
    for (std::size_t p = 0; p < plan_.exec.size(); ++p) {
        const Op& op = ops_[static_cast<std::size_t>(plan_.exec[p])];
        ctx.workspace = arena_ + plan_.ws_offset[p];
        ctx.workspace_bytes = plan_.ws_size[p];
        const auto t0 = std::chrono::steady_clock::now();
        op.layer->forward(bound_in_[p], bound_out_[p], ctx);
        if (profile)
            std::fprintf(stderr, "[prof] %-30s %8.3f ms\n", op.layer->name.c_str(),
                         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
}
