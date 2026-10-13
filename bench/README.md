# tiny_yolo vs. ncnn, Ultralytics, OpenCV DNN and ONNX Runtime

The same cherry YOLOv8n weights (`best.pt`, and its ncnn export in `samples/model/`) run in
every engine, on the 8 sample images × 4 passes. tiny_yolo uses int8 weights and ncnn its
low-memory options, so both C++ engines are tuned for memory. Full tables, including
1-thread runs: [`RESULTS_mac.md`](RESULTS_mac.md) and [`RESULTS_linux.md`](RESULTS_linux.md).

## Headline (4 threads)

| Engine | Peak RSS (macOS) | Peak RSS (Linux) | Inference / img (macOS) | Inference / img (Linux) | Cold start (macOS) |
|---|---|---|---|---|---|
| **tiny_yolo** (int8 weights) | **11.2 MiB** | **10.3 MiB** | 24.3 ms | 28.4 ms | 0.04 s |
| ncnn (low-memory options) | 46.1 MiB | 46.6 MiB | 20.5 ms | 21.5 ms | 0.03 s |
| OpenCV DNN (Python) | 165.1 MiB | 135.6 MiB | 23.6 ms | 35.9 ms | 0.29 s |
| ONNX Runtime (Python) | 176.0 MiB | 168.1 MiB | 29.7 ms | 29.6 ms | 0.34 s |
| Ultralytics / PyTorch (Python) | 457.4 MiB | 464.8 MiB | 23.5 ms | 89.0 ms | 3.95 s |

Every engine finds all 33 reference detections (reference: Ultralytics fp32).

- **Memory is where tiny_yolo wins**: ~4× less than ncnn, 13–15× less than OpenCV DNN,
  ~16× less than ONNX Runtime and ~40–45× less than Ultralytics. The Python numbers include the
  interpreter and libraries (53–79 MiB for OpenCV and ONNX Runtime, ~265 MiB for PyTorch +
  Ultralytics with no model loaded), but even without them they need 80–200 MiB.
- **Speed is now competitive**: on 4 threads tiny_yolo is faster than ONNX Runtime on both
  platforms and than OpenCV DNN on Linux, level with OpenCV on macOS, and ~1.2–1.3× behind
  ncnn. On 1 thread it beats OpenCV DNN and ONNX Runtime and is ~1.1–1.2× behind ncnn
  (see the full tables).
- Starting a process and running the first image takes 0.04 s, against ~4 s for Ultralytics.
- Before the speed work (same memory), tiny_yolo took 37.3 ms (macOS) and 48.9 ms (Linux)
  on 4 threads.

## Notes

- ncnn is built from source for CPU only (no Vulkan). Its low-memory options turn off Winograd
  and sgemm convolution, which keep transformed copies of the weights.
- Detection differences of up to ~0.05 in score (tiny_yolo and ncnn) come from preprocessing:
  each C++ engine has its own bilinear resize, while the Python runners use `cv2.resize`.
  The weights are the same (the ncnn model matches `best.pt` to 3e-7).

## Running it

```sh
# macOS
uv venv bench/.venv --python 3.12
uv pip install --python bench/.venv/bin/python ultralytics torch opencv-python onnx onnxruntime onnxslim
cp /path/to/best.pt bench/models/model.pt
bench/.venv/bin/python -c "from ultralytics import YOLO; YOLO('bench/models/model.pt').export(format='onnx', opset=12, simplify=True)"
make && python3 tools/jpg_to_ppm.py samples/images samples/ppm
# CPU-only ncnn: build ncnn with -DNCNN_VULKAN=OFF -DNCNN_SHARED_LIB=OFF and OpenMP, then
c++ -O3 -std=c++17 -I$NCNN/include bench/ncnn_runner.cpp $NCNN/lib/libncnn.a -lomp -o bench/ncnn_runner_lean
bench/.venv/bin/python bench/benchmark.py --tag mac

# Linux (Docker; builds everything inside)
docker build -f bench/Dockerfile -t tiny-yolo-bench .
docker run --rm --cpus 4 --cpuset-cpus 0-3 -v "$PWD/bench:/out" tiny-yolo-bench
```

| File | Contents |
|---|---|
| `benchmark.py` | driver: fresh process per configuration, peak RSS, times, detection comparison |
| `ncnn_runner.cpp` | ncnn C++ runner with tiny_yolo's CLI, preprocessing and decoding (`--lowmem`, `--fp32`) |
| `py_runner.py` | Ultralytics (`YOLO().predict`), OpenCV DNN and ONNX Runtime runners |
| `Dockerfile` | Linux environment |

Inference time is the forward pass only, as reported by each engine (median, first pass dropped).
End-to-end time in the full tables also includes reading the image, preprocessing and NMS.
