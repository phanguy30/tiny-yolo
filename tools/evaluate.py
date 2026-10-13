#!/usr/bin/env python3
"""Measure tiny_yolo and optionally compare its detections with a reference engine.

Usage:
  python3 tools/evaluate.py --engine ./tiny_yolo --model MODEL_DIR --images IMAGE_DIR
                            [--reference PATH_TO_REFERENCE_ENGINE] [--threads 4]
                            [--flags "--weights int8"]

MODEL_DIR must contain model.param and model.bin; IMAGE_DIR the .ppm images.
Both engines must use the multi-image CLI:
  ENGINE model.param model.bin img1.ppm [img2.ppm ...] --conf C --nms N --threads T

Reports, for the engine:
  * peak RSS of one process that processes every image (model loaded once),
  * per-image time = (time for all images - time for 1 image) / (n - 1), median of 3,
    so one-time model loading and planning are excluded.
With --reference, also matches detections image by image (same class, best IoU)
and reports matched count, worst IoU and largest score difference.
Standard library only.
"""
import argparse
import json
import shlex
import statistics
import subprocess
import sys
from pathlib import Path

HELPER = r"""
import json, resource, subprocess, sys, time
cmd = json.loads(sys.argv[1])
t = time.perf_counter()
out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
dt = time.perf_counter() - t
rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
print(json.dumps({"seconds": dt, "rss": rss, "stdout": out}))
"""


def run(cmd):
    """Run cmd in a fresh helper process so ru_maxrss belongs to this run only."""
    res = subprocess.run([sys.executable, "-c", HELPER, json.dumps(cmd)],
                         check=True, capture_output=True, text=True)
    r = json.loads(res.stdout)
    r["rss_mib"] = r["rss"] / (1048576 if sys.platform == "darwin" else 1024)
    return r


def parse(stdout):
    sections, current = {}, None
    for line in stdout.splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == "image":
            current = Path(line[6:]).name
            sections[current] = []
        elif p[0] != "time" and current is not None and len(p) == 6:
            sections[current].append((p[0], float(p[1]), *map(float, p[2:])))
    return sections


def iou(a, b):
    x1, y1 = max(a[2], b[2]), max(a[3], b[3])
    x2, y2 = min(a[4], b[4]), min(a[5], b[5])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = lambda d: (d[4] - d[2]) * (d[5] - d[3])
    return inter / (area(a) + area(b) - inter + 1e-9)


def compare(ref, new):
    total = matched = 0
    worst_iou, worst_score = 1.0, 0.0
    extra = 0
    for name, ref_dets in ref.items():
        cand = list(new.get(name, []))
        total += len(ref_dets)
        for r in sorted(ref_dets, key=lambda d: -d[1]):
            best = max(((iou(r, c), j) for j, c in enumerate(cand) if c[0] == r[0]), default=None)
            if best and best[0] > 0.5:
                matched += 1
                worst_iou = min(worst_iou, best[0])
                worst_score = max(worst_score, abs(r[1] - cand[best[1]][1]))
                cand.pop(best[1])
        extra += len(cand)
    return total, matched, extra, worst_iou, worst_score


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--reference")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--flags", default="", help="extra engine flags, e.g. \"--weights int8\"")
    ap.add_argument("--conf", default="0.25")
    ap.add_argument("--nms", default="0.45")
    a = ap.parse_args()

    images = sorted(str(p) for p in a.images.glob("*.ppm"))
    if len(images) < 2:
        sys.exit("need at least two .ppm images")
    base = [str(a.model / "model.param"), str(a.model / "model.bin")]
    opts = ["--conf", a.conf, "--nms", a.nms, "--threads", str(a.threads)] + shlex.split(a.flags)
    cmd = lambda exe, imgs, extra=opts: [exe, *base, *imgs, *extra]

    run(cmd(a.engine, images[:1]))  # warm the file cache
    per_image, peaks, full = [], [], None
    for _ in range(3):
        one = run(cmd(a.engine, images[:1]))
        full = run(cmd(a.engine, images))
        per_image.append((full["seconds"] - one["seconds"]) / (len(images) - 1))
        peaks.append(full["rss_mib"])
    t = statistics.median(per_image)
    print(f"images:            {len(images)}")
    print(f"peak RSS:          {statistics.median(peaks):.1f} MiB (median of 3 full runs)")
    print(f"per-image time:    {t * 1000:.1f} ms  ({1 / t:.1f} images/s, threads={a.threads})")

    if a.reference:
        ref = run(cmd(a.reference, images, ["--conf", a.conf, "--nms", a.nms]))
        ref_t = ref["seconds"] / len(images)
        total, matched, extra, worst_iou, worst_score = compare(parse(ref["stdout"]), parse(full["stdout"]))
        print(f"reference:         {ref['rss_mib']:.1f} MiB peak, ~{ref_t:.2f} s/image  (speedup ~{ref_t / t:.0f}x)")
        print(f"detections:        {matched}/{total} matched, {extra} extra, "
              f"worst IoU {worst_iou:.4f}, max score diff {worst_score:.4f}")


if __name__ == "__main__":
    main()
