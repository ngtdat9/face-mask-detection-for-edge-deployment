"""Export a trained MaskCNN checkpoint to ONNX (and optionally INT8).

Steps:

1. Rebuild the model from the checkpoint and export it to ONNX with a
   dynamic batch dimension. Class order, image size, mean and std are
   stored as ONNX metadata so the .onnx file is self-contained.
2. Parity check: run PyTorch and ONNX Runtime on every test image and
   report the largest logit difference and whether predictions match.
3. With --quantize, write an INT8 copy using static post-training
   quantization calibrated on validation images, then report its test
   accuracy next to the fp32 model.

Run from the project root::

    python src/export_onnx.py --model_path models/mask_classifier_gap.pth --quantize
"""

from __future__ import annotations

import argparse
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edge_utils import (  # noqa: E402
    OnnxClassifier,
    TorchClassifier,
    list_labeled_images,
    preprocess_bgr,
    write_onnx_metadata,
)

import cv2  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OPSET = 18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MaskCNN to ONNX / INT8 ONNX.")
    parser.add_argument("--model_path", type=str,
                        default=str(PROJECT_ROOT / "models" / "mask_classifier.pth"),
                        help="PyTorch checkpoint to export.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .onnx path (default: same name as the checkpoint).")
    parser.add_argument("--test_dir", type=str, default=str(PROJECT_ROOT / "data" / "test"),
                        help="Test set used for the parity and accuracy checks.")
    parser.add_argument("--quantize", action="store_true",
                        help="Also write an INT8 model (<name>_int8.onnx).")
    parser.add_argument("--calib_dir", type=str, default=str(PROJECT_ROOT / "data" / "val"),
                        help="Images used to calibrate INT8 quantization.")
    parser.add_argument("--calib_size", type=int, default=300,
                        help="Number of calibration images (default: 300).")
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


def load_batches(items: list[tuple[Path, int]], runner, batch_size: int):
    """Yield (batch, labels) using the NumPy/OpenCV preprocessing."""
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        batch = np.stack([
            preprocess_bgr(cv2.imread(str(p)), runner.image_size, runner.mean, runner.std)
            for p, _ in chunk
        ])
        yield batch, np.array([label for _, label in chunk])


def export(checkpoint_path: Path, onnx_path: Path) -> TorchClassifier:
    runner = TorchClassifier(checkpoint_path, device="cpu")
    dummy = torch.zeros(1, 3, runner.image_size, runner.image_size)
    torch.onnx.export(
        runner.model,
        (dummy,),
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=OPSET,
        dynamo=True,
        external_data=False,
    )
    write_onnx_metadata(onnx_path, {
        "class_to_idx": runner.class_to_idx,
        "image_size": runner.image_size,
        "mean": runner.mean,
        "std": runner.std,
        "head": runner.model.head,
        "source_checkpoint": checkpoint_path.name,
    })
    return runner


def parity_and_accuracy(torch_runner, onnx_runner, items, batch_size: int) -> dict:
    max_diff = 0.0
    agree = 0
    torch_correct = 0
    onnx_correct = 0
    for batch, labels in load_batches(items, torch_runner, batch_size):
        t_logits = torch_runner(batch)
        o_logits = onnx_runner(batch)
        max_diff = max(max_diff, float(np.abs(t_logits - o_logits).max()))
        t_pred = t_logits.argmax(1)
        o_pred = o_logits.argmax(1)
        agree += int((t_pred == o_pred).sum())
        torch_correct += int((t_pred == labels).sum())
        onnx_correct += int((o_pred == labels).sum())
    n = len(items)
    return {
        "max_abs_logit_diff": max_diff,
        "prediction_agreement": agree / n,
        "torch_accuracy": torch_correct / n,
        "onnx_accuracy": onnx_correct / n,
    }


def accuracy(runner, items, batch_size: int) -> float:
    correct = 0
    for batch, labels in load_batches(items, runner, batch_size):
        correct += int((runner(batch).argmax(1) == labels).sum())
    return correct / len(items)


def quantize_int8(fp32_path: Path, int8_path: Path, runner, calib_items) -> None:
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self._iter = iter(calib_items)

        def get_next(self):
            item = next(self._iter, None)
            if item is None:
                return None
            x = preprocess_bgr(cv2.imread(str(item[0])), runner.image_size,
                               runner.mean, runner.std)
            return {"input": x[None]}

    with tempfile.TemporaryDirectory() as tmp:
        prepped = Path(tmp) / "prepped.onnx"
        quant_pre_process(str(fp32_path), str(prepped))
        quantize_static(
            str(prepped),
            str(int8_path),
            Reader(),
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
        )


def size_mb(path: Path) -> float:
    return path.stat().st_size / 1e6


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.model_path)
    if not checkpoint_path.is_file():
        print(f"ERROR: checkpoint not found: {checkpoint_path}", file=sys.stderr)
        raise SystemExit(1)
    onnx_path = Path(args.output) if args.output else checkpoint_path.with_suffix(".onnx")

    print(f"Exporting {checkpoint_path.name} -> {onnx_path.name} (opset {OPSET})")
    torch_runner = export(checkpoint_path, onnx_path)
    onnx_runner = OnnxClassifier(onnx_path)

    test_items = list_labeled_images(Path(args.test_dir), torch_runner.class_to_idx)
    report = parity_and_accuracy(torch_runner, onnx_runner, test_items, args.batch_size)

    print(f"\nParity check on {len(test_items)} test images (OpenCV preprocessing):")
    print(f"  max |logit diff|      : {report['max_abs_logit_diff']:.2e}")
    print(f"  prediction agreement  : {report['prediction_agreement'] * 100:.2f}%")
    print(f"  PyTorch accuracy      : {report['torch_accuracy'] * 100:.2f}%")
    print(f"  ONNX fp32 accuracy    : {report['onnx_accuracy'] * 100:.2f}%")
    print(f"  checkpoint size       : {size_mb(checkpoint_path):.2f} MB")
    print(f"  ONNX fp32 size        : {size_mb(onnx_path):.2f} MB")

    if args.quantize:
        int8_path = onnx_path.with_name(onnx_path.stem + "_int8.onnx")
        calib_all = list_labeled_images(Path(args.calib_dir), torch_runner.class_to_idx)
        calib_items = random.Random(0).sample(calib_all, min(args.calib_size, len(calib_all)))
        print(f"\nQuantizing to INT8 with {len(calib_items)} calibration images...")
        quantize_int8(onnx_path, int8_path, torch_runner, calib_items)
        write_onnx_metadata(int8_path, {
            "class_to_idx": torch_runner.class_to_idx,
            "image_size": torch_runner.image_size,
            "mean": torch_runner.mean,
            "std": torch_runner.std,
            "head": torch_runner.model.head,
            "source_checkpoint": checkpoint_path.name,
            "quantization": "int8 static, per-channel, QDQ",
        })
        int8_acc = accuracy(OnnxClassifier(int8_path), test_items, args.batch_size)
        print(f"  ONNX INT8 accuracy    : {int8_acc * 100:.2f}% "
              f"({(int8_acc - report['onnx_accuracy']) * 100:+.2f} points vs fp32)")
        print(f"  ONNX INT8 size        : {size_mb(int8_path):.2f} MB")
        print(f"  wrote {int8_path}")

    print(f"\nwrote {onnx_path}")


if __name__ == "__main__":
    main()
