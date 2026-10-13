# Linux benchmark environment with the same engines.
#   docker build -f bench/Dockerfile -t tiny-yolo-bench .
#   docker run --rm --cpus 4 -v "$PWD/bench:/out" tiny-yolo-bench
FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir ultralytics opencv-python-headless onnxruntime numpy pillow
# CPU-only ncnn (same release as the macOS runs), OpenMP via libgomp
RUN git clone -q --depth 1 --branch 20260526 https://github.com/Tencent/ncnn.git /ncnn-src \
    && cmake -S /ncnn-src -B /ncnn-build -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=OFF -DNCNN_BUILD_TOOLS=OFF \
       -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF -DNCNN_BUILD_TESTS=OFF -DNCNN_SHARED_LIB=OFF \
       -DCMAKE_INSTALL_PREFIX=/ncnn \
    && cmake --build /ncnn-build -j"$(nproc)" && cmake --install /ncnn-build && rm -rf /ncnn-src /ncnn-build
WORKDIR /bench
COPY Makefile ./
COPY src src
COPY tools tools
COPY samples samples
COPY bench/benchmark.py bench/py_runner.py bench/ncnn_runner.cpp bench/
COPY bench/models bench/models
RUN make && g++ -O3 -std=c++17 -I/ncnn/include bench/ncnn_runner.cpp /ncnn/lib/libncnn.a -fopenmp -o bench/ncnn_runner_lean \
    && python tools/jpg_to_ppm.py samples/images samples/ppm
ENV BENCH_PYTHON=/usr/local/bin/python
CMD python bench/benchmark.py --tag linux && cp bench/results_linux.json bench/RESULTS_linux.md /out/
