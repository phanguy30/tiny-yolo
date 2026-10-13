#include "graph.hpp"
#include "conv_chain.hpp"
#include "dtype.hpp"
#include "image.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

struct Detection {
    float x1, y1, x2, y2, score;
    int class_id;
};

static float iou(const Detection& a, const Detection& b) {
    float x1 = std::max(a.x1, b.x1), y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2), y2 = std::min(a.y2, b.y2);
    float overlap = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    return overlap / (area_a + area_b - overlap + 1e-7f);
}

// output: [4 + num_classes, num_anchors] (cx, cy, w, h, class scores)
static std::vector<Detection> decode(const TensorView& output, const Letterbox& box, float threshold, float nms_threshold) {
    const int rows = output.shape.h(), count = output.shape.w();
    const int num_classes = rows - 4;
    std::vector<float> buf;  // output is fp32 by default; convert if needed
    const float* o = output.f32();
    if (output.dtype != DType::F32) {
        buf.resize(output.size());
        copy_convert(output.data, output.dtype, buf.data(), DType::F32, output.size());
        o = buf.data();
    }
    std::vector<Detection> candidates;
    for (int i = 0; i < count; ++i) {
        int class_id = 0;
        float score = o[4 * count + i];
        for (int k = 1; k < num_classes; ++k)
            if (o[(4 + k) * count + i] > score) { score = o[(4 + k) * count + i]; class_id = k; }
        if (score < threshold) continue;
        float cx = o[i], cy = o[count + i], w = o[2 * count + i], h = o[3 * count + i];
        Detection d;
        d.x1 = std::clamp((cx - w / 2 - box.left) / box.scale, 0.f, float(box.width));
        d.y1 = std::clamp((cy - h / 2 - box.top) / box.scale, 0.f, float(box.height));
        d.x2 = std::clamp((cx + w / 2 - box.left) / box.scale, 0.f, float(box.width));
        d.y2 = std::clamp((cy + h / 2 - box.top) / box.scale, 0.f, float(box.height));
        d.score = score;
        d.class_id = class_id;
        candidates.push_back(d);
    }
    std::stable_sort(candidates.begin(), candidates.end(), [](const Detection& a, const Detection& b) { return a.score > b.score; });
    std::vector<Detection> result;
    for (const auto& c : candidates) {
        bool keep = true;
        for (const auto& d : result)
            if (c.class_id == d.class_id && iou(c, d) > nms_threshold) { keep = false; break; }
        if (keep) result.push_back(c);
    }
    return result;
}

static long proc_status_kb(const char* key) {  // Linux only; returns -1 elsewhere
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line))
        if (line.compare(0, std::strlen(key), key) == 0) return std::atol(line.c_str() + std::strlen(key) + 1);
    return -1;
}

static void usage() {
    std::cout << "tiny_yolo model.param model.bin image1.ppm [image2.ppm ...] [--conf C] [--nms N] [--threads T] [options]\n"
                 "  --conf C             confidence threshold (default 0.25)\n"
                 "  --nms N              NMS IoU threshold (default 0.45)\n"
                 "  --threads N          worker threads (default: min(4, hw threads))\n"
                 "  --weights fp16|fp32|int8|stream   weight storage (default fp16; stream = read\n"
                 "                       from the model file per inference, ~0 resident)\n"
                 "  --fp32-below N       fp16/int8: layers with fewer than N weights stay fp32 (default 16384)\n"
                 "  --int8-min N         int8: only layers with at least N weights are int8,\n"
                 "                       smaller ones fp16 (default 32768)\n"
                 "  --input-size N       network input resolution (default 640)\n"
                 "  --bench N            run inference N times and report the mean\n"
                 "  --stats              print memory statistics to stderr\n"
                 "  --verbose            print the optimized execution plan\n"
                 "  --activations fp16|fp32    activation storage (default fp16; compute is fp32)\n"
                 "  --input fp16|u8|fp32 network input storage (default fp16)\n"
                 "  --no-fuse --no-views --no-concat-inplace --no-inplace --no-stream --no-winograd\n"
                 "                       disable optimizations\n";
}

int main(int argc, char** argv) {
    try {
        std::vector<std::string> pos;
        EngineOptions opt;
        unsigned hw = std::thread::hardware_concurrency();
        opt.config.threads = static_cast<int>(std::min(4u, hw ? hw : 1u));
        int bench = 1, input_size = 640;
        float confidence = .25f, nms = .45f;
        bool stats = false;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
                return argv[++i];
            };
            if (a == "--conf") confidence = std::stof(next());
            else if (a == "--nms") nms = std::stof(next());
            else if (a == "--threads") opt.config.threads = std::max(1, std::stoi(next()));
            else if (a == "--weights") {
                std::string f = next();
                if (f == "fp32") opt.config.weight_format = WeightFormat::F32;
                else if (f == "fp16") opt.config.weight_format = WeightFormat::F16;
                else if (f == "int8") opt.config.weight_format = WeightFormat::I8;
                else if (f == "stream") opt.config.weight_format = WeightFormat::Stream;
                else throw std::runtime_error("unknown weight format " + f);
            }
            else if (a == "--fp32-below") opt.config.fp32_below = static_cast<std::size_t>(std::max(0, std::stoi(next())));
            else if (a == "--int8-min") opt.config.int8_min = static_cast<std::size_t>(std::max(0, std::stoi(next())));
            else if (a == "--input-size") input_size = std::stoi(next());
            else if (a == "--band-pixels") ConvChainLayer::band_target_pixels = std::max(1, std::stoi(next()));
            else if (a == "--int8-group") PackedWeights::int8_group = std::max(1, std::stoi(next()));
            else if (a == "--ring-kb") ConvChainLayer::ring_budget_bytes = static_cast<std::size_t>(std::max(1, std::stoi(next()))) * 1024;
            else if (a == "--panel-kb") ConvolutionLayer::panel_budget_bytes = static_cast<std::size_t>(std::max(1, std::stoi(next()))) * 1024;
            else if (a == "--bench") bench = std::max(1, std::stoi(next()));
            else if (a == "--stats") stats = true;
            else if (a == "--verbose") opt.verbose = true;
            else if (a == "--no-fuse") opt.fuse_activations = opt.fuse_residual = false;
            else if (a == "--no-views") opt.zero_copy_views = false;
            else if (a == "--no-concat-inplace") opt.concat_in_place = false;
            else if (a == "--no-inplace") opt.inplace_elementwise = false;
            else if (a == "--no-stream") opt.stream_conv_chains = false;
            else if (a == "--no-winograd") opt.config.winograd = false;
            else if (a == "--input") {
                std::string f = next();
                if (f == "fp32") opt.input_dtype = DType::F32;
                else if (f == "fp16") opt.input_dtype = DType::F16;
                else if (f == "u8") opt.input_dtype = DType::U8;
                else throw std::runtime_error("unknown input format " + f);
            }
            else if (a == "--activations") {
                std::string f = next();
                if (f == "fp32") opt.f16_activations = false;
                else if (f == "fp16") opt.f16_activations = true;
                else throw std::runtime_error("unknown activation format " + f);
            }
            else if (a == "-h" || a == "--help") { usage(); return 0; }
            else if (a.rfind("--", 0) == 0) throw std::runtime_error("unknown option " + a);
            else pos.push_back(a);
        }
        if (pos.size() < 3) { usage(); return 1; }

        const bool trace = std::getenv("TINY_YOLO_TRACE_RSS") != nullptr;
        auto rss = [&](const char* stage) { if (trace) std::cerr << "[rss] " << stage << " " << proc_status_kb("VmRSS:") << " kB\n"; };
        rss("start");
        opt.input_u8_scale = 1.f / 255.f;  // the network expects pixels / 255
        Engine engine(opt);
        rss("engine");
        engine.load(pos[0], pos[1]);
        rss("loaded");
        engine.prepare(Shape{3, input_size, input_size});
        rss("prepared");

        const char* names[] = {"immature", "mature"};
        std::cout << std::fixed << std::setprecision(4);
        double best = 1e30;
        // The model is loaded and planned once; every image reuses the same arena.
        for (std::size_t img = 2; img < pos.size(); ++img) {
            const std::string& path = pos[img];
            std::cout << "image " << path << "\n";
            Letterbox box{};
            double total = 0.0;
            for (int r = 0; r < bench; ++r) {
                // the PPM is streamed straight into the network input on every run
                box = letterbox_ppm_into(path, engine.input());
                auto start = std::chrono::steady_clock::now();
                engine.run();
                const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                total += s;
                best = std::min(best, s);
            }
            rss("ran");
            const auto detections = decode(engine.output("out0"), box, confidence, nms);

            if (const char* dp = std::getenv("TINY_YOLO_DUMP_OUTPUT")) {  // debugging aid: raw output tensor
                const TensorView o = engine.output("out0");
                std::ofstream f(dp, std::ios::binary | std::ios::app);
                std::vector<float> tmp(o.size());
                copy_convert(o.data, o.dtype, tmp.data(), DType::F32, o.size());
                f.write(reinterpret_cast<const char*>(tmp.data()), static_cast<std::streamsize>(tmp.size() * sizeof(float)));
            }

            std::cout << "time " << total / bench << "\n";
            for (const auto& d : detections)
                std::cout << (d.class_id < 2 ? names[d.class_id] : ("class_" + std::to_string(d.class_id)).c_str()) << ' '
                          << d.score << ' ' << d.x1 << ' ' << d.y1 << ' ' << d.x2 << ' ' << d.y2 << '\n';
        }
        if (stats) {
            const MemoryReport& m = engine.report();
            std::cerr << std::fixed << std::setprecision(2)
                      << "[stats] threads=" << opt.config.threads << " exec_ops=" << m.exec_ops
                      << " arena=" << m.arena_bytes / 1048576.0 << "MiB"
                      << " params=" << m.param_bytes / 1048576.0 << "MiB"
                      << " naive_activations=" << m.naive_bytes / 1048576.0 << "MiB"
                      << " best_time=" << std::setprecision(4) << best;
            const long hwm = proc_status_kb("VmHWM:"), rss_now = proc_status_kb("VmRSS:");
            if (hwm > 0) std::cerr << " peak_rss=" << hwm / 1024.0 << "MiB";
            if (rss_now > 0) std::cerr << " rss_now=" << rss_now / 1024.0 << "MiB";
            std::cerr << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
