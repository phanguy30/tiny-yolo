# Benchmark results

Apple M3 Max, 8 sample images (1328×1328 → 640×640) × 4 passes, same cherry YOLOv8n weights in every engine.
Reference for detections: Ultralytics fp32, 4 threads.

| Engine | Mode | Threads | Inference / img | End-to-end / img | Peak RSS | Runtime-only RSS | Cold start | Detections vs. Ultralytics |
|---|---|---|---|---|---|---|---|---|
| tiny_yolo (C++) | default (fp16 storage) | 4 | 39.1 ms | 42.8 ms | **13.5 MiB** | – | 0.06 s | 33/33, IoU ≥ 0.9927, Δscore ≤ 0.0361 |
| ncnn (C++) | default (fp16 storage+arith) | 4 | 10.5 ms | 12.5 ms | **86.0 MiB** | – | 0.03 s | 33/33, IoU ≥ 0.9496, Δscore ≤ 0.0502 |
| Ultralytics (PyTorch) (Python) | fp32 | 4 | 24.2 ms | 29.7 ms | **460.2 MiB** | 263 MiB | 4.07 s | 33/33, IoU ≥ 1.0000, Δscore ≤ 0.0000 |
| OpenCV DNN (Python) | fp32 | 4 | 24.3 ms | 27.1 ms | **171.6 MiB** | 65 MiB | 0.30 s | 33/33, IoU ≥ 1.0000, Δscore ≤ 0.0001 |
| ONNX Runtime (Python) | fp32 | 4 | 29.9 ms | 33.4 ms | **184.9 MiB** | 80 MiB | 0.32 s | 33/33, IoU ≥ 1.0000, Δscore ≤ 0.0001 |
| tiny_yolo (C++) | default (fp16 storage) | 1 | 130.8 ms | 135.4 ms | **13.3 MiB** | – | 0.15 s | 33/33, IoU ≥ 0.9927, Δscore ≤ 0.0361 |
| ncnn (C++) | default (fp16 storage+arith) | 1 | 32.7 ms | 34.5 ms | **80.8 MiB** | – | 0.06 s | 33/33, IoU ≥ 0.9496, Δscore ≤ 0.0502 |
| Ultralytics (PyTorch) (Python) | fp32 | 1 | 24.1 ms | 33.0 ms | **463.4 MiB** | 263 MiB | 4.04 s | 33/33, IoU ≥ 1.0000, Δscore ≤ 0.0000 |
| OpenCV DNN (Python) | fp32 | 1 | 98.7 ms | 100.4 ms | **163.0 MiB** | 65 MiB | 0.39 s | 33/33, IoU ≥ 1.0000, Δscore ≤ 0.0001 |
| ONNX Runtime (Python) | fp32 | 1 | 99.9 ms | 103.5 ms | **175.7 MiB** | 79 MiB | 0.38 s | 33/33, IoU ≥ 1.0000, Δscore ≤ 0.0001 |
| tiny_yolo (C++) | --weights int8 | 4 | 37.8 ms | 41.3 ms | **11.0 MiB** | – | 0.06 s | 33/33, IoU ≥ 0.9579, Δscore ≤ 0.0485 |
| tiny_yolo (C++) | exact fp32 | 4 | 34.9 ms | 37.3 ms | **22.8 MiB** | – | 0.05 s | 33/33, IoU ≥ 0.9927, Δscore ≤ 0.0363 |
| ncnn (C++) | fp32 (fp16 off) | 4 | 16.9 ms | 19.3 ms | **159.1 MiB** | – | 0.04 s | 33/33, IoU ≥ 0.9587, Δscore ≤ 0.0344 |

Runtime-only RSS = the same Python process with its libraries imported but no model loaded.
