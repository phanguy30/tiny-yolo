#!/usr/bin/env python3
"""Compare tiny_yolo with ncnn, Ultralytics (PyTorch), OpenCV DNN and ONNX Runtime.

  bench/.venv/bin/python bench/benchmark.py [--passes 4] [--repeats 2]

Every configuration runs in a fresh process that loads the model once and processes the
sample images `passes` times. Reported per configuration:
  * peak RSS of that process (ru_maxrss),
  * inference time per image: median of the engine's own forward-pass timings, first pass
    dropped as warm-up (best of `repeats` runs),
  * end-to-end time per image: (wall time for all images - wall time for 1 image) / (n - 1),
    i.e. image decode + preprocessing + inference + decoding/NMS, without process start-up
    and model loading,
  * cold start: wall time of a process that loads the model and runs 1 image,
  * detections vs. Ultralytics (fp32 PyTorch, the training framework) as reference.
Writes bench/results_<tag>.json and bench/RESULTS_<tag>.md.
"""
import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from evaluate import compare, parse  # noqa: E402  (same parser / matcher as tools/evaluate.py)

PY = os.environ.get("BENCH_PYTHON", str(ROOT / "bench/.venv/bin/python"))
PARAM, BIN = str(ROOT / "samples/model/model.param"), str(ROOT / "samples/model/model.bin")
PT, ONNX = str(ROOT / "bench/models/model.pt"), str(ROOT / "bench/models/model.onnx")

HELPER = r"""
import json, resource, subprocess, sys, time
cmd = json.loads(sys.argv[1])
t = time.perf_counter()
out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
dt = time.perf_counter() - t
print(json.dumps({"seconds": dt, "rss": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, "stdout": out}))
"""


def run(cmd):
    res = subprocess.run([sys.executable, "-c", HELPER, json.dumps(cmd)], check=True, capture_output=True, text=True)
    r = json.loads(res.stdout)
    r["rss_mib"] = r["rss"] / (1048576 if sys.platform == "darwin" else 1024)
    return r


def times(stdout):
    return [float(l.split()[1]) for l in stdout.splitlines() if l.startswith("time ")]


def configs(ncnn_bin):
    tiny = lambda *f: [str(ROOT / "tiny_yolo"), PARAM, BIN, "{images}", *f]
    ncnn = lambda *f: [ncnn_bin, PARAM, BIN, "{images}", *f]
    py = lambda e, m, *f: [PY, str(ROOT / "bench/py_runner.py"), e, m, "{images}", *f]
    c = []
    for t in (4, 1):
        c += [
            ("tiny_yolo", "int8 weights", t, "C++", tiny("--threads", str(t), "--weights", "int8"), None),
            ("ncnn", "low-memory options", t, "C++", ncnn("--threads", str(t), "--lowmem"), None),
            ("Ultralytics (PyTorch)", "fp32", t, "Python", py("ultralytics", PT, "--threads", str(t)), "ultralytics"),
            ("OpenCV DNN", "fp32", t, "Python", py("opencv", ONNX, "--threads", str(t)), "opencv"),
            ("ONNX Runtime", "fp32", t, "Python", py("onnxruntime", ONNX, "--threads", str(t)), "onnxruntime"),
        ]
    return c


def expand(cmd, images):
    i = cmd.index("{images}")
    return cmd[:i] + images + cmd[i + 1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--ncnn", default=str(ROOT / "bench/ncnn_runner_lean"), help="ncnn runner binary")
    ap.add_argument("--tag", default="mac", help="suffix for results_<tag>.json / RESULTS_<tag>.md")
    a = ap.parse_args()
    images = sorted(str(p) for p in (ROOT / "samples/ppm").glob("*.ppm"))
    assert images, "run tools/jpg_to_ppm.py samples/images samples/ppm first"
    n = len(images)
    all_imgs = images * a.passes

    results, reference = [], None
    for engine, mode, threads, lang, cmd, py_engine in configs(a.ncnn):
        best = None
        for _ in range(a.repeats):
            r = run(expand(cmd, all_imgs))
            infer = statistics.median(times(r["stdout"])[n:])  # drop warm-up pass
            if best is None or infer < best["infer"]:
                best = {"infer": infer, "wall": r["seconds"], "rss": r["rss_mib"], "stdout": r["stdout"]}
        one = min((run(expand(cmd, images[:1])) for _ in range(a.repeats)), key=lambda r: r["seconds"])
        dets = parse(best["stdout"])  # last pass wins per image name; all passes are identical
        row = {
            "engine": engine, "mode": mode, "threads": threads, "lang": lang,
            "infer_ms": best["infer"] * 1e3,
            "e2e_ms": (best["wall"] - one["seconds"]) / (len(all_imgs) - 1) * 1e3,
            "cold_start_s": one["seconds"],
            "peak_rss_mib": best["rss"],
            "detections": dets,
        }
        if py_engine:
            base_cmd = [PY, str(ROOT / "bench/py_runner.py"), py_engine, "x", "--baseline"]
            row["runtime_rss_mib"] = run(base_cmd)["rss_mib"]
        if engine.startswith("Ultralytics") and threads == 4:
            reference = dets
        results.append(row)
        print(f"{engine:22s} {mode:30s} T={threads}  infer {row['infer_ms']:7.1f} ms  e2e {row['e2e_ms']:7.1f} ms  "
              f"RSS {row['peak_rss_mib']:7.1f} MiB  cold {row['cold_start_s']:.2f} s", flush=True)

    for row in results:
        total, matched, extra, worst_iou, worst_score = compare(reference, row["detections"])
        row["accuracy"] = {"total": total, "matched": matched, "extra": extra, "worst_iou": worst_iou, "max_score_diff": worst_score}

    (ROOT / f"bench/results_{a.tag}.json").write_text(json.dumps(
        {"machine": f"{platform.machine()} {platform.platform()}", "images": n, "passes": a.passes, "results": results}, indent=1))
    write_markdown(results, n, a.passes, a)


def write_markdown(results, n, passes, a):
    cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip() \
        if sys.platform == "darwin" else f"Linux ({os.cpu_count()} CPUs)"
    lines = [
        "# Benchmark results", "",
        f"{cpu}, {n} sample images (1328×1328 → 640×640) × {passes} passes, same cherry YOLOv8n weights in every engine.",
        "Reference for detections: Ultralytics fp32, 4 threads.", "",
        "| Engine | Mode | Threads | Inference / img | End-to-end / img | Peak RSS | Runtime-only RSS | Cold start | Detections vs. Ultralytics |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        acc = r["accuracy"]
        det = f"{acc['matched']}/{acc['total']}, IoU ≥ {acc['worst_iou']:.4f}, Δscore ≤ {acc['max_score_diff']:.4f}"
        if acc.get("extra"):
            det += f", +{acc['extra']} extra"
        base = f"{r['runtime_rss_mib']:.0f} MiB" if "runtime_rss_mib" in r else "–"
        lines.append(f"| {r['engine']} ({r['lang']}) | {r['mode']} | {r['threads']} | {r['infer_ms']:.1f} ms | {r['e2e_ms']:.1f} ms | "
                     f"**{r['peak_rss_mib']:.1f} MiB** | {base} | {r['cold_start_s']:.2f} s | {det} |")
    lines += ["", "Runtime-only RSS = the same Python process with its libraries imported but no model loaded."]
    (ROOT / f"bench/RESULTS_{a.tag}.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
