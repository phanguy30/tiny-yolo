#!/usr/bin/env python3
"""Convert images (JPG, PNG, ...) to binary PPM (P6), the input format of tiny_yolo.

Usage: python3 tools/jpg_to_ppm.py INPUT_DIR OUTPUT_DIR
Requires Pillow (pip install pillow).
"""
import sys
from pathlib import Path

from PIL import Image

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def convert(src_dir, dst_dir):
    dst_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for src in sorted(src_dir.iterdir()):
        if src.suffix.lower() not in EXTENSIONS:
            continue
        dst = dst_dir / (src.stem + ".ppm")
        if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
            Image.open(src).convert("RGB").save(dst, format="PPM")
        written.append(dst)
    return written


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    for path in convert(Path(sys.argv[1]), Path(sys.argv[2])):
        print(path)
