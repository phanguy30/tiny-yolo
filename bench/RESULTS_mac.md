# Benchmark results

Apple M3 Max, 8 sample images (1328×1328 → 640×640) × 4 passes, same cherry YOLOv8n weights in every engine.
Reference for detections: Ultralytics fp32, 4 threads.

| Engine | Mode | Threads | Inference / img | End-to-end / img | Peak RSS | Runtime-only RSS | Cold start | Detections vs. Ultralytics |
|---|---|---|---|---|---|---|---|---|
| tiny_yolo (C++) | int8 weights | 4 | 24.3 ms | 26.4 ms | **11.2 MiB** | – | 0.04 s | 33/33, IoU ≥ 0.9578, Δscore ≤ 0.0490 |
| ncnn (C++) | low-memory options | 4 | 20.5 ms | 22.6 ms | **46.1 MiB** | – | 0.03 s | 33/33, IoU ≥ 0.9571, Δscore ≤ 0.0492, +1 extra |
| Ultralytics (PyTorch) (Python) | fp32 | 4 | 23.5 ms | 52.1 ms | **457.4 MiB** | 263 MiB | 3.95 s | 33/33, IoU ≥ 1.0000, Δscore ≤ 0.0000 |
| OpenCV DNN (Python) | fp32 | 4 | 23.6 ms | 27.8 ms | **165.1 MiB** | 65 MiB | 0.29 s | 33/33, IoU ≥ 1.0000, Δscore ≤ 0.0001 |
| ONNX Runtime (Python) | fp32 | 4 | 29.7 ms | 32.5 ms | **176.0 MiB** | 79 MiB | 0.34 s | 33/33, IoU ≥ 1.0000, Δscore ≤ 0.0001 |
| tiny_yolo (C++) | int8 weights | 1 | 79.0 ms | 81.5 ms | **10.9 MiB** | – | 0.09 s | 33/33, IoU ≥ 0.9578, Δscore ≤ 0.0490 |
| ncnn (C++) | low-memory options | 1 | 72.8 ms | 74.5 ms | **45.9 MiB** | – | 0.08 s | 33/33, IoU ≥ 0.9571, Δscore ≤ 0.0492, +1 extra |
| Ultralytics (PyTorch) (Python) | fp32 | 1 | 23.4 ms | 25.7 ms | **452.7 MiB** | 264 MiB | 4.11 s | 33/33, IoU ≥ 1.0000, Δscore ≤ 0.0000 |
| OpenCV DNN (Python) | fp32 | 1 | 98.6 ms | 102.1 ms | **163.2 MiB** | 64 MiB | 0.37 s | 33/33, IoU ≥ 1.0000, Δscore ≤ 0.0001 |
| ONNX Runtime (Python) | fp32 | 1 | 100.2 ms | 104.4 ms | **183.0 MiB** | 79 MiB | 0.39 s | 33/33, IoU ≥ 1.0000, Δscore ≤ 0.0001 |

Runtime-only RSS = the same Python process with its libraries imported but no model loaded.
