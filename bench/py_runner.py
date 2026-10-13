#!/usr/bin/env python3
"""Python runners with the same CLI and output format as tiny_yolo.

  py_runner.py ENGINE MODEL img1.ppm [...] [--threads T] [--conf C] [--nms N] [--baseline]

ENGINE: ultralytics (MODEL = .pt, the stock YOLO().predict path),
        opencv (MODEL = .onnx, cv2.dnn), onnxruntime (MODEL = .onnx).
"time" is inference only (forward pass), like tiny_yolo's. --baseline imports the libraries
and exits, to measure the interpreter + library memory without a model.
"""
import argparse
import time

import numpy as np

NAMES = ["immature", "mature"]


def letterbox(path, cv2, size=640):
    """Same preprocessing as tiny_yolo: RGB, bilinear resize, pad 114, / 255, CHW."""
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    ih, iw = img.shape[:2]
    scale = min(size / iw, size / ih)
    w, h = round(iw * scale), round(ih * scale)
    left, top = (size - w) // 2, (size - h) // 2
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[top:top + h, left:left + w] = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    blob = (canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
    return np.ascontiguousarray(blob), (scale, left, top, iw, ih)


def decode(out, box, conf, nms):
    """out: [6, 8400]. Class-wise greedy NMS, same as tiny_yolo."""
    scale, left, top, iw, ih = box
    scores = out[4:]
    cls = scores.argmax(0)
    s = scores[cls, np.arange(out.shape[1])]
    keep = s >= conf
    cx, cy, w, h = out[:4, keep]
    s, cls = s[keep], cls[keep]
    xyxy = np.stack([(cx - w / 2 - left) / scale, (cy - h / 2 - top) / scale,
                     (cx + w / 2 - left) / scale, (cy + h / 2 - top) / scale], 1)
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, iw)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, ih)
    order = np.argsort(-s, kind="stable")
    result = []
    for i in order:
        b = xyxy[i]
        ok = True
        for j in result:
            if cls[j] != cls[i]:
                continue
            r = xyxy[j]
            iw_ = max(0.0, min(b[2], r[2]) - max(b[0], r[0]))
            ih_ = max(0.0, min(b[3], r[3]) - max(b[1], r[1]))
            inter = iw_ * ih_
            union = (b[2] - b[0]) * (b[3] - b[1]) + (r[2] - r[0]) * (r[3] - r[1]) - inter
            if inter / (union + 1e-7) > nms:
                ok = False
                break
        if ok:
            result.append(i)
    return [(NAMES[cls[i]], float(s[i]), *map(float, xyxy[i])) for i in result]


def emit(path, seconds, dets):
    print(f"image {path}\ntime {seconds:.4f}")
    for d in dets:
        print(f"{d[0]} {d[1]:.4f} {d[2]:.4f} {d[3]:.4f} {d[4]:.4f} {d[5]:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", choices=["ultralytics", "opencv", "onnxruntime"])
    ap.add_argument("model")
    ap.add_argument("images", nargs="*")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--nms", type=float, default=0.45)
    ap.add_argument("--baseline", action="store_true")
    a = ap.parse_args()

    if a.engine == "ultralytics":
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(a.threads)
        if a.baseline:
            return
        import cv2
        model = YOLO(a.model, task="detect")
        for p in a.images:  # Ultralytics doesn't accept .ppm paths, so pass the decoded BGR array
            r = model.predict(cv2.imread(p), imgsz=640, conf=a.conf, iou=a.nms, device="cpu", verbose=False)[0]
            dets = [(NAMES[int(c)], float(s), *map(float, b))
                    for b, s, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
            emit(p, r.speed["inference"] / 1000.0, dets)
        return

    import cv2
    if a.engine == "opencv":
        cv2.setNumThreads(a.threads)
        if a.baseline:
            return
        net = cv2.dnn.readNetFromONNX(a.model)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        def infer(blob):
            net.setInput(blob)
            return net.forward()
    else:
        import onnxruntime as ort
        if a.baseline:
            return
        so = ort.SessionOptions()
        so.intra_op_num_threads = a.threads
        so.inter_op_num_threads = 1
        sess = ort.InferenceSession(a.model, so, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name

        def infer(blob):
            return sess.run(None, {name: blob})[0]

    for p in a.images:
        blob, box = letterbox(p, cv2)
        t = time.perf_counter()
        out = infer(blob)
        dt = time.perf_counter() - t
        emit(p, dt, decode(np.asarray(out).reshape(-1, 8400), box, a.conf, a.nms))


if __name__ == "__main__":
    main()
