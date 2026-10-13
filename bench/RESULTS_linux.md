# Benchmark results

Linux (16 CPUs), 8 sample images (1328×1328 → 640×640) × 4 passes, same cherry YOLOv8n weights in every engine.
Reference for detections: Ultralytics fp32, 4 threads.

| Engine | Mode | Threads | Inference / img | End-to-end / img | Peak RSS | Runtime-only RSS | Cold start | Detections vs. Ultralytics |
|---|---|---|---|---|---|---|---|---|
| tiny_yolo (C++) | int8 weights | 4 | 28.4 ms | 31.1 ms | **10.3 MiB** | – | 0.05 s | 33/33, IoU ≥ 0.9578, Δscore ≤ 0.0490 |
| ncnn (C++) | low-memory options | 4 | 21.5 ms | 23.6 ms | **46.6 MiB** | – | 0.05 s | 33/33, IoU ≥ 0.9571, Δscore ≤ 0.0492, +1 extra |
| Ultralytics (PyTorch) (Python) | fp32 | 4 | 89.0 ms | 97.2 ms | **464.8 MiB** | 268 MiB | 1.37 s | 33/33, IoU ≥ 1.0000, Δscore ≤ 0.0000 |
| OpenCV DNN (Python) | fp32 | 4 | 35.9 ms | 39.2 ms | **135.6 MiB** | 53 MiB | 0.18 s | 33/33, IoU ≥ 1.0000, Δscore ≤ 0.0001 |
| ONNX Runtime (Python) | fp32 | 4 | 29.6 ms | 32.7 ms | **168.1 MiB** | 72 MiB | 0.19 s | 33/33, IoU ≥ 1.0000, Δscore ≤ 0.0001 |
| tiny_yolo (C++) | int8 weights | 1 | 86.5 ms | 89.1 ms | **10.3 MiB** | – | 0.10 s | 33/33, IoU ≥ 0.9578, Δscore ≤ 0.0490 |
| ncnn (C++) | low-memory options | 1 | 71.8 ms | 74.1 ms | **46.6 MiB** | – | 0.09 s | 33/33, IoU ≥ 0.9571, Δscore ≤ 0.0492, +1 extra |
| Ultralytics (PyTorch) (Python) | fp32 | 1 | 86.0 ms | 90.7 ms | **464.5 MiB** | 268 MiB | 1.35 s | 33/33, IoU ≥ 1.0000, Δscore ≤ 0.0000 |
| OpenCV DNN (Python) | fp32 | 1 | 132.8 ms | 136.7 ms | **135.5 MiB** | 53 MiB | 0.25 s | 33/33, IoU ≥ 1.0000, Δscore ≤ 0.0001 |
| ONNX Runtime (Python) | fp32 | 1 | 100.2 ms | 102.8 ms | **168.7 MiB** | 72 MiB | 0.24 s | 33/33, IoU ≥ 1.0000, Δscore ≤ 0.0001 |

Runtime-only RSS = the same Python process with its libraries imported but no model loaded.
