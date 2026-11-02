"""Benchmark every exported model variant for edge deployment.

For each model file and backend configuration this measures:

* file size and parameter count
* test-set accuracy with the OpenCV/NumPy preprocessing used on device
* single-face classifier latency (batch of 1): median and 95th percentile
* end-to-end pipeline FPS: YuNet face detection + crop + preprocess +
  classify + draw, timed over a fixed set of frames (camera capture and
  display are excluded so the numbers are repeatable)

Backends: PyTorch on CPU (1 and 4 threads), PyTorch on MPS (reference
only, edge devices have no Apple GPU), and ONNX Runtime on CPU (1 and 4
threads). The 4-thread CPU numbers are the closest stand-in for a
4-core edge board such as a Raspberry Pi; they are not measurements on
one.

By default the pipeline frames are synthetic: 640x480 canvases with 1-3
real test-set faces pasted in. Pass --video to use a recorded clip.

Run from the project root::

    python src/benchmark.py
    python src/benchmark.py --video my_webcam_clip.mp4
"""

from __future__ import annotations

import argparse
import csv
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edge_utils import (  # noqa: E402
    OnnxClassifier,
    TorchClassifier,
    list_labeled_images,
    preprocess_bgr,
    read_onnx_metadata,
    softmax,
)
from live_camera import YuNetFaceDetector, draw_label, ensure_yunet_model  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Edge benchmark for the mask classifier.")
    parser.add_argument("--test_dir", type=str, default=str(PROJECT_ROOT / "data" / "test"))
    parser.add_argument("--results_dir", type=str, default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4],
                        help="CPU thread counts to test (default: 1 4).")
    parser.add_argument("--no_mps", action="store_true", help="Skip the MPS reference runs.")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--runs", type=int, default=500,
                        help="Timed single-face inferences per configuration.")
    parser.add_argument("--frames", type=int, default=300,
                        help="Synthetic frames for the pipeline benchmark.")
    parser.add_argument("--video", type=str, default=None,
                        help="Use frames from this video instead of synthetic frames.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


# --------------------------------------------------------------------------- #
#  Model discovery                                                            #
# --------------------------------------------------------------------------- #
def describe(head: str, image_size: int) -> str:
    return f"{head}-{image_size}"


def discover_models() -> list[dict]:
    """Find every classifier checkpoint and its ONNX exports in models/."""
    import torch

    from model import build_model_from_checkpoint, count_parameters

    models = []
    for pth in sorted(MODELS_DIR.glob("mask_classifier*.pth")):
        checkpoint = torch.load(pth, map_location="cpu", weights_only=False)
        params = count_parameters(build_model_from_checkpoint(checkpoint))
        variant = describe(checkpoint.get("head", "flatten"),
                           int(checkpoint.get("image_size", 128)))
        models.append({"variant": variant, "format": "PyTorch fp32", "path": pth,
                       "params": params})
        for onnx_path, fmt in ((pth.with_suffix(".onnx"), "ONNX fp32"),
                               (pth.with_name(pth.stem + "_int8.onnx"), "ONNX INT8")):
            if onnx_path.is_file():
                meta = read_onnx_metadata(onnx_path)
                if meta.get("source_checkpoint") != pth.name:
                    continue
                models.append({"variant": variant, "format": fmt, "path": onnx_path,
                               "params": params})
    return models


def build_configs(models: list[dict], threads: list[int], use_mps: bool) -> list[dict]:
    configs = []
    for m in models:
        if m["format"].startswith("PyTorch"):
            for t in threads:
                configs.append({**m, "backend": "PyTorch CPU", "device": "cpu", "threads": t})
            if use_mps:
                configs.append({**m, "backend": "PyTorch MPS (reference)", "device": "mps",
                                "threads": max(threads)})
        else:
            for t in threads:
                configs.append({**m, "backend": "ONNX Runtime CPU", "device": "cpu",
                                "threads": t})
    return configs


def make_runner(cfg: dict):
    if cfg["format"].startswith("PyTorch"):
        return TorchClassifier(cfg["path"], device=cfg["device"], num_threads=cfg["threads"])
    return OnnxClassifier(cfg["path"], num_threads=cfg["threads"])


# --------------------------------------------------------------------------- #
#  Frames for the pipeline benchmark                                          #
# --------------------------------------------------------------------------- #
def synthetic_frames(items: list[tuple[Path, int]], n: int, seed: int) -> list[np.ndarray]:
    """640x480 frames, each with 1-3 real test faces on a plain background."""
    rng = np.random.default_rng(seed)
    w, h = 640, 480
    frames = []
    for _ in range(n):
        base = rng.integers(60, 200, size=3)
        gradient = np.linspace(0.8, 1.2, w, dtype=np.float32)[None, :, None]
        frame = np.clip(np.ones((h, w, 3), np.float32) * base * gradient, 0, 255)
        frame = (frame + rng.normal(0, 6, frame.shape)).clip(0, 255).astype(np.uint8)
        n_faces = int(rng.integers(1, 4))
        slot_w = w // n_faces
        for i in range(n_faces):
            face = cv2.imread(str(items[int(rng.integers(len(items)))][0]))
            target_h = int(rng.integers(140, 220))
            scale = target_h / face.shape[0]
            fw = min(int(face.shape[1] * scale), slot_w - 10)
            face = cv2.resize(face, (fw, target_h))
            x = i * slot_w + int(rng.integers(0, max(1, slot_w - fw)))
            y = int(rng.integers(20, h - target_h - 20))
            frame[y:y + target_h, x:x + fw] = face
        frames.append(frame)
    return frames


def video_frames(path: Path, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < n:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, (640, 480)) if frame.shape[1] != 640 else frame)
    cap.release()
    if not frames:
        raise SystemExit(f"ERROR: could not read frames from {path}")
    return frames


# --------------------------------------------------------------------------- #
#  Measurements                                                               #
# --------------------------------------------------------------------------- #
def measure_accuracy(runner, items, batch_size: int = 64) -> float:
    correct = 0
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        batch = np.stack([preprocess_bgr(cv2.imread(str(p)), runner.image_size,
                                         runner.mean, runner.std) for p, _ in chunk])
        labels = np.array([label for _, label in chunk])
        correct += int((runner(batch).argmax(1) == labels).sum())
    return correct / len(items)


def measure_latency(runner, face_bgr: np.ndarray, warmup: int, runs: int) -> dict:
    x = preprocess_bgr(face_bgr, runner.image_size, runner.mean, runner.std)[None]
    for _ in range(warmup):
        runner(x)
    times = np.empty(runs)
    for i in range(runs):
        start = time.perf_counter()
        runner(x)
        times[i] = time.perf_counter() - start
    times *= 1000
    return {"latency_p50_ms": float(np.percentile(times, 50)),
            "latency_p95_ms": float(np.percentile(times, 95))}


def measure_pipeline(runner, detector, frames: list[np.ndarray], idx_to_class) -> dict:
    """Detect -> crop -> preprocess -> classify -> draw, per frame."""
    for frame in frames[:10]:
        detector.detect(frame)
    detect_s = classify_s = 0.0
    faces = 0
    start_all = time.perf_counter()
    for original in frames:
        frame = original.copy()
        t0 = time.perf_counter()
        boxes = detector.detect(frame)
        t1 = time.perf_counter()
        crops, kept = [], []
        fh, fw = frame.shape[:2]
        for x, y, w, h in boxes:
            x1, y1, x2, y2 = max(0, x), max(0, y), min(fw, x + w), min(fh, y + h)
            if x2 > x1 and y2 > y1:
                crops.append(preprocess_bgr(frame[y1:y2, x1:x2], runner.image_size,
                                            runner.mean, runner.std))
                kept.append((x, y, w, h))
        if crops:
            probs = softmax(runner(np.stack(crops)))
            for box, p in zip(kept, probs):
                draw_label(frame, box, idx_to_class[int(p.argmax())], float(p.max()))
        t2 = time.perf_counter()
        detect_s += t1 - t0
        classify_s += t2 - t1
        faces += len(kept)
    total = time.perf_counter() - start_all
    n = len(frames)
    return {"pipeline_fps": n / total,
            "detect_ms_per_frame": detect_s / n * 1000,
            "classify_ms_per_frame": classify_s / n * 1000,
            "faces_per_frame": faces / n}


# --------------------------------------------------------------------------- #
#  Reporting                                                                  #
# --------------------------------------------------------------------------- #
def system_info() -> dict:
    import onnxruntime
    import torch

    try:
        chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"],
                                       text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        chip = platform.processor() or "unknown"
    return {"chip": chip, "os": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(), "torch": torch.__version__,
            "onnxruntime": onnxruntime.__version__, "opencv": cv2.__version__}


def write_markdown(rows: list[dict], info: dict, frame_source: str, path: Path) -> None:
    lines = [
        "# Edge benchmark results",
        "",
        f"- Machine: {info['chip']}, {info['os']}",
        f"- Software: Python {info['python']}, PyTorch {info['torch']}, "
        f"ONNX Runtime {info['onnxruntime']}, OpenCV {info['opencv']}",
        f"- Pipeline frames: {frame_source}",
        "- Accuracy uses OpenCV resizing (the on-device path), so it can differ "
        "slightly from evaluate.py, which resizes with PIL.",
        "- CPU thread limits approximate a small edge board; these are not "
        "measurements on an edge device.",
        "",
        "| Variant | Format | Backend | Threads | Size (MB) | Params | Test acc | "
        "p50 latency (ms) | p95 latency (ms) | Pipeline FPS | Detect (ms) | Classify (ms) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['variant']} | {r['format']} | {r['backend']} | {r['threads']} | "
            f"{r['size_mb']:.2f} | {r['params']:,} | {r['test_accuracy'] * 100:.2f}% | "
            f"{r['latency_p50_ms']:.2f} | {r['latency_p95_ms']:.2f} | "
            f"{r['pipeline_fps']:.1f} | {r['detect_ms_per_frame']:.2f} | "
            f"{r['classify_ms_per_frame']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def plot_latency(rows: list[dict], threads: int, path: Path) -> None:
    subset = [r for r in rows if r["backend"] != "PyTorch MPS (reference)"
              and r["threads"] == threads]
    if not subset:
        return
    labels = [f"{r['variant']}\n{r['format']}" for r in subset]
    values = [r["latency_p50_ms"] for r in subset]
    fig, ax = plt.subplots(figsize=(max(7, len(subset) * 1.1), 4.5))
    bars = ax.bar(range(len(subset)), values, color="#4C72B0")
    ax.bar_label(bars, fmt="%.2f", padding=2)
    ax.set_xticks(range(len(subset)), labels, fontsize=8)
    ax.set_ylabel("median latency per face (ms)")
    ax.set_title(f"Classifier latency on CPU, {threads} thread(s), batch of 1")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    models = discover_models()
    if not models:
        raise SystemExit("ERROR: no models/mask_classifier*.pth checkpoints found.")
    use_mps = not args.no_mps and __import__("torch").backends.mps.is_available()
    configs = build_configs(models, args.threads, use_mps)

    class_to_idx = TorchClassifier(models[0]["path"]).class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    items = list_labeled_images(Path(args.test_dir), class_to_idx)
    sample_face = cv2.imread(str(items[0][0]))

    if args.video:
        frames = video_frames(Path(args.video), args.frames)
        frame_source = f"{len(frames)} frames from {Path(args.video).name}"
    else:
        frames = synthetic_frames(items, args.frames, args.seed)
        frame_source = (f"{len(frames)} synthetic 640x480 frames with 1-3 test-set "
                        f"faces each (seed {args.seed})")
    yunet_path = ensure_yunet_model(None)

    accuracy_cache: dict[Path, float] = {}
    rows = []
    print(f"Benchmarking {len(configs)} configurations on {frame_source}\n")
    for cfg in configs:
        runner = make_runner(cfg)
        if cfg["path"] not in accuracy_cache:
            accuracy_cache[cfg["path"]] = measure_accuracy(runner, items)
        cv2.setNumThreads(cfg["threads"])
        detector = YuNetFaceDetector(yunet_path)
        row = {
            "variant": cfg["variant"], "format": cfg["format"], "backend": cfg["backend"],
            "threads": cfg["threads"], "size_mb": cfg["path"].stat().st_size / 1e6,
            "params": cfg["params"], "test_accuracy": accuracy_cache[cfg["path"]],
            **measure_latency(runner, sample_face, args.warmup, args.runs),
            **measure_pipeline(runner, detector, frames, idx_to_class),
        }
        rows.append(row)
        print(f"{row['variant']:12} {row['format']:13} {row['backend']:24} "
              f"t={row['threads']}  acc={row['test_accuracy'] * 100:6.2f}%  "
              f"p50={row['latency_p50_ms']:6.2f} ms  p95={row['latency_p95_ms']:6.2f} ms  "
              f"fps={row['pipeline_fps']:6.1f}")

    csv_path = results_dir / "benchmark.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    info = system_info()
    write_markdown(rows, info, frame_source, results_dir / "benchmark.md")
    plot_latency(rows, max(args.threads), results_dir / "benchmark_latency.png")

    sample = frames[0].copy()
    runner = make_runner(configs[-1])
    for x, y, w, h in YuNetFaceDetector(yunet_path).detect(sample):
        crop = sample[max(0, y):y + h, max(0, x):x + w]
        if crop.size:
            p = softmax(runner(preprocess_bgr(crop, runner.image_size,
                                              runner.mean, runner.std)[None]))[0]
            draw_label(sample, (x, y, w, h), idx_to_class[int(p.argmax())], float(p.max()))
    cv2.imwrite(str(results_dir / "benchmark_sample_frame.jpg"), sample)

    print(f"\nwrote {csv_path}")
    print(f"wrote {results_dir / 'benchmark.md'}")
    print(f"wrote {results_dir / 'benchmark_latency.png'}")


if __name__ == "__main__":
    main()
