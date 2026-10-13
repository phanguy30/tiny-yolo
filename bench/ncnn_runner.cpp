// Reference runner on the stock ncnn library, with the same CLI and output format as tiny_yolo:
//   ncnn_runner model.param model.bin img1.ppm [...] [--threads T] [--bench N] [--conf C] [--nms N] [--fp32] [--lowmem]
// Preprocessing (bilinear letterbox to 640, pad 114, RGB / 255) and decoding (class-wise NMS)
// match tiny_yolo. "time" is inference only (input -> extract), like tiny_yolo's.
#include <ncnn/net.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

struct Det { float x1, y1, x2, y2, score; int cls; };

static float iou(const Det& a, const Det& b) {
    float x1 = std::max(a.x1, b.x1), y1 = std::max(a.y1, b.y1), x2 = std::min(a.x2, b.x2), y2 = std::min(a.y2, b.y2);
    float o = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
    return o / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - o + 1e-7f);
}

static std::vector<unsigned char> read_ppm(const std::string& path, int& w, int& h) {
    std::ifstream f(path, std::ios::binary);
    std::string magic;
    int maxv;
    f >> magic >> w >> h >> maxv;
    f.get();
    std::vector<unsigned char> px(static_cast<size_t>(w) * h * 3);
    f.read(reinterpret_cast<char*>(px.data()), static_cast<std::streamsize>(px.size()));
    if (!f || magic != "P6") throw std::runtime_error("bad PPM " + path);
    return px;
}

int main(int argc, char** argv) {
    std::vector<std::string> pos;
    int threads = 4, bench = 1;
    float conf = .25f, nms = .45f;
    bool fp32 = false, lowmem = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--threads") threads = std::stoi(argv[++i]);
        else if (a == "--bench") bench = std::stoi(argv[++i]);
        else if (a == "--conf") conf = std::stof(argv[++i]);
        else if (a == "--nms") nms = std::stof(argv[++i]);
        else if (a == "--fp32") fp32 = true;
        else if (a == "--lowmem") lowmem = true;
        else pos.push_back(a);
    }
    if (pos.size() < 3) { std::cerr << "usage: ncnn_runner model.param model.bin img.ppm...\n"; return 1; }

    ncnn::Net net;
    net.opt.num_threads = threads;
    net.opt.use_vulkan_compute = false;
    if (fp32) net.opt.use_fp16_storage = net.opt.use_fp16_packed = net.opt.use_fp16_arithmetic = false;
    // ncnn's memory-saving knobs: no winograd / sgemm weight transforms (they keep extra
    // transformed copies of the weights), light mode frees intermediates as soon as possible.
    if (lowmem) {
        net.opt.use_winograd_convolution = net.opt.use_sgemm_convolution = false;
        net.opt.lightmode = true;
    }
    if (net.load_param(pos[0].c_str()) || net.load_model(pos[1].c_str())) { std::cerr << "load failed\n"; return 1; }

    const int S = 640;
    const char* names[] = {"immature", "mature"};
    std::cout << std::fixed << std::setprecision(4);
    for (size_t k = 2; k < pos.size(); ++k) {
        int iw, ih;
        auto px = read_ppm(pos[k], iw, ih);
        float scale = std::min(float(S) / iw, float(S) / ih);
        int w = int(std::round(iw * scale)), h = int(std::round(ih * scale));
        int left = (S - w) / 2, top = (S - h) / 2;
        ncnn::Mat resized = ncnn::Mat::from_pixels_resize(px.data(), ncnn::Mat::PIXEL_RGB, iw, ih, w, h);
        ncnn::Mat in;
        ncnn::copy_make_border(resized, in, top, S - h - top, left, S - w - left, ncnn::BORDER_CONSTANT, 114.f);
        const float norm[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
        in.substract_mean_normalize(nullptr, norm);

        ncnn::Mat out;
        double total = 0;
        for (int r = 0; r < bench; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            ncnn::Extractor ex = net.create_extractor();
            ex.input("in0", in);
            ex.extract("out0", out);
            total += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }

        // out: [6, 8400] = cx, cy, w, h, class scores
        const int n = out.w, nc = out.h - 4;
        std::vector<Det> cand;
        for (int i = 0; i < n; ++i) {
            int c = 0;
            float s = out.row(4)[i];
            for (int j = 1; j < nc; ++j) if (out.row(4 + j)[i] > s) { s = out.row(4 + j)[i]; c = j; }
            if (s < conf) continue;
            float cx = out.row(0)[i], cy = out.row(1)[i], bw = out.row(2)[i], bh = out.row(3)[i];
            cand.push_back({std::clamp((cx - bw / 2 - left) / scale, 0.f, float(iw)), std::clamp((cy - bh / 2 - top) / scale, 0.f, float(ih)),
                            std::clamp((cx + bw / 2 - left) / scale, 0.f, float(iw)), std::clamp((cy + bh / 2 - top) / scale, 0.f, float(ih)), s, c});
        }
        std::stable_sort(cand.begin(), cand.end(), [](const Det& a, const Det& b) { return a.score > b.score; });
        std::vector<Det> keep;
        for (auto& c : cand) {
            bool ok = true;
            for (auto& d : keep) if (c.cls == d.cls && iou(c, d) > nms) { ok = false; break; }
            if (ok) keep.push_back(c);
        }
        std::cout << "image " << pos[k] << "\ntime " << total / bench << "\n";
        for (auto& d : keep)
            std::cout << names[d.cls] << ' ' << d.score << ' ' << d.x1 << ' ' << d.y1 << ' ' << d.x2 << ' ' << d.y2 << '\n';
    }
}
