# Sample model and images

## `model/`: cherry ripeness detector

A YOLOv8 detector with two classes, trained with Ultralytics and exported to ncnn format
(`model.param` + `model.bin`, fp32 weights, ~3.0 M parameters).

| | |
|---|---|
| Classes | `immature`, `mature` |
| Input | 640×640 RGB, letterboxed, pixels scaled to [0, 1] |
| Output | `out0`: `[6, 8400]`, 4 box values (cx, cy, w, h in input pixels) + 2 class scores per anchor |
| Training | 30 epochs, image size 640, batch 64, from pretrained weights |
| Validation (final epoch) | precision 0.848, recall 0.826, mAP@50 0.908, mAP@50-95 0.720 |

## `images/`: sample orchard photos

Eight 1328×1328 photos of cherry branches with ripe and unripe fruit. The notebook
(`notebooks/cherry_demo.ipynb`) and `tools/jpg_to_ppm.py` convert them to PPM, the engine's
input format, in `samples/ppm/` (git-ignored).
