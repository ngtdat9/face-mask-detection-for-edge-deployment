"""Compare Haar Cascade and YuNet face detection on the test set.

Every test image is a single face, so a detector "finds" the face when it
returns at least one box. The script reports, per class, the detection
rate and the average number of boxes, plus average detection time.

Images are first scaled so their longest side is --canvas_size pixels,
then padded with a grey border (--pad_ratio of the image size on every
side). Both detectors see exactly the same input, with the same
parameters live_camera.py uses.

Run from the project root::

    python src/compare_detectors.py
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edge_utils import list_labeled_images  # noqa: E402
from live_camera import HaarFaceDetector, YuNetFaceDetector, ensure_yunet_model  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLASS_TO_IDX = {"with_mask": 0, "without_mask": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Haar vs YuNet detection rate on the test set.")
    parser.add_argument("--test_dir", type=str, default=str(PROJECT_ROOT / "data" / "test"))
    parser.add_argument("--results_dir", type=str, default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--canvas_size", type=int, default=320,
                        help="Longest image side before padding (default: 320).")
    parser.add_argument("--pad_ratio", type=float, default=0.25,
                        help="Grey border on each side, as a fraction of image size.")
    parser.add_argument("--min_face_size", type=int, default=40,
                        help="Haar minimum face size, same as live_camera.py (default: 40).")
    parser.add_argument("--score_threshold", type=float, default=0.7,
                        help="YuNet score threshold, same as live_camera.py (default: 0.7).")
    return parser.parse_args()


def prepare(image: np.ndarray, canvas_size: int, pad_ratio: float) -> np.ndarray:
    h, w = image.shape[:2]
    scale = canvas_size / max(h, w)
    resized = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    pad_y = round(resized.shape[0] * pad_ratio)
    pad_x = round(resized.shape[1] * pad_ratio)
    return cv2.copyMakeBorder(resized, pad_y, pad_y, pad_x, pad_x,
                              cv2.BORDER_CONSTANT, value=(128, 128, 128))


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    detectors = {
        "Haar Cascade": HaarFaceDetector(None, args.min_face_size),
        "YuNet (DNN)": YuNetFaceDetector(ensure_yunet_model(None),
                                         score_threshold=args.score_threshold),
    }
    items = list_labeled_images(Path(args.test_dir), CLASS_TO_IDX)
    idx_to_class = {v: k for k, v in CLASS_TO_IDX.items()}
    frames = [(prepare(cv2.imread(str(p)), args.canvas_size, args.pad_ratio), label)
              for p, label in items]

    rows = []
    for name, detector in detectors.items():
        found = {c: 0 for c in CLASS_TO_IDX}
        boxes = {c: 0 for c in CLASS_TO_IDX}
        totals = {c: 0 for c in CLASS_TO_IDX}
        elapsed = 0.0
        for frame, label in frames:
            cls = idx_to_class[label]
            start = time.perf_counter()
            detections = detector.detect(frame)
            elapsed += time.perf_counter() - start
            totals[cls] += 1
            boxes[cls] += len(detections)
            found[cls] += int(len(detections) > 0)
        for cls in CLASS_TO_IDX:
            rows.append({
                "detector": name,
                "class": cls,
                "images": totals[cls],
                "detected": found[cls],
                "detection_rate": found[cls] / totals[cls],
                "avg_boxes_per_image": boxes[cls] / totals[cls],
                "avg_ms_per_image": elapsed / len(frames) * 1000,
            })

    csv_path = results_dir / "detector_comparison.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'detector':14} {'class':13} {'detected':>12} {'rate':>8} "
          f"{'boxes/img':>10} {'ms/img':>8}")
    print("-" * 70)
    for r in rows:
        print(f"{r['detector']:14} {r['class']:13} "
              f"{r['detected']:>5}/{r['images']:<6} {r['detection_rate'] * 100:>7.1f}% "
              f"{r['avg_boxes_per_image']:>10.2f} {r['avg_ms_per_image']:>8.2f}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    classes = list(CLASS_TO_IDX)
    width = 0.38
    for i, name in enumerate(detectors):
        rates = [next(r["detection_rate"] for r in rows
                      if r["detector"] == name and r["class"] == c) * 100 for c in classes]
        bars = ax.bar(np.arange(len(classes)) + (i - 0.5) * width, rates, width, label=name)
        ax.bar_label(bars, fmt="%.1f%%", padding=2)
    ax.set_xticks(np.arange(len(classes)), classes)
    ax.set_ylabel("faces detected (%)")
    ax.set_ylim(0, 110)
    ax.set_title(f"Face detection rate on the test set (n={len(frames)})")
    ax.legend(loc="lower right")
    fig.tight_layout()
    chart_path = results_dir / "detector_comparison.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    print(f"\nwrote {csv_path}")
    print(f"wrote {chart_path}")


if __name__ == "__main__":
    main()
