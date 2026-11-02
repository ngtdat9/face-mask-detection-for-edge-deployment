"""Shared helpers for edge deployment: preprocessing and model runners.

Everything here avoids torchvision and PIL so the same code path can run
on a device that only has OpenCV, NumPy, and ONNX Runtime installed.
The PyTorch runner exists only so benchmarks compare backends on
identical inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
METADATA_KEY = "mask_classifier"


# --------------------------------------------------------------------------- #
#  Preprocessing                                                              #
# --------------------------------------------------------------------------- #
def preprocess_bgr(
    face_bgr: np.ndarray,
    image_size: int,
    mean: list[float],
    std: list[float],
) -> np.ndarray:
    """BGR uint8 face crop -> normalized float32 tensor of shape (3, H, W)."""
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    # INTER_LINEAR aliases badly when shrinking large crops; INTER_AREA
    # matches the antialiased PIL resize the model was trained with.
    shrinking = max(rgb.shape[:2]) > image_size
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    resized = cv2.resize(rgb, (image_size, image_size), interpolation=interpolation)
    x = resized.astype(np.float32) / 255.0
    x = (x - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def list_labeled_images(root: Path, class_to_idx: dict[str, int]) -> list[tuple[Path, int]]:
    """Return (path, label) pairs for an ImageFolder-style directory, sorted."""
    items: list[tuple[Path, int]] = []
    for name, idx in sorted(class_to_idx.items(), key=lambda kv: kv[1]):
        class_dir = root / name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        for p in sorted(class_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                items.append((p, idx))
    return items


# --------------------------------------------------------------------------- #
#  ONNX metadata                                                              #
# --------------------------------------------------------------------------- #
def write_onnx_metadata(onnx_path: Path, metadata: dict) -> None:
    """Store class order and preprocessing settings inside the .onnx file."""
    import onnx

    model = onnx.load(str(onnx_path))
    for prop in list(model.metadata_props):
        if prop.key == METADATA_KEY:
            model.metadata_props.remove(prop)
    entry = model.metadata_props.add()
    entry.key = METADATA_KEY
    entry.value = json.dumps(metadata)
    onnx.save(model, str(onnx_path))


def read_onnx_metadata(onnx_path: Path) -> dict:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    raw = session.get_modelmeta().custom_metadata_map.get(METADATA_KEY)
    if raw is None:
        raise ValueError(
            f"{onnx_path} has no '{METADATA_KEY}' metadata. "
            "Re-export it with `python src/export_onnx.py`."
        )
    return json.loads(raw)


# --------------------------------------------------------------------------- #
#  Model runners                                                              #
# --------------------------------------------------------------------------- #
class OnnxClassifier:
    """Run an exported mask classifier with ONNX Runtime on CPU."""

    backend = "onnxruntime"

    def __init__(self, onnx_path: Path, num_threads: int | None = None) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # By default idle worker threads busy-wait after each run, which
        # starves the OpenCV face detector sharing the same CPU cores.
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        if num_threads is not None:
            options.intra_op_num_threads = num_threads
            options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        meta = json.loads(
            self.session.get_modelmeta().custom_metadata_map[METADATA_KEY]
        )
        self.class_to_idx: dict[str, int] = meta["class_to_idx"]
        self.image_size: int = int(meta["image_size"])
        self.mean: list[float] = meta["mean"]
        self.std: list[float] = meta["std"]

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: batch})[0]


class TorchClassifier:
    """Run a PyTorch checkpoint on CPU or MPS with NumPy in and out."""

    backend = "pytorch"

    def __init__(self, checkpoint_path: Path, device: str = "cpu",
                 num_threads: int | None = None) -> None:
        import torch

        from model import build_model_from_checkpoint

        if num_threads is not None:
            torch.set_num_threads(num_threads)
        self._torch = torch
        self.device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model = build_model_from_checkpoint(checkpoint)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval().to(self.device)
        self.class_to_idx = checkpoint["class_to_idx"]
        self.image_size = int(checkpoint.get("image_size", 128))
        self.mean = list(checkpoint.get("mean", [0.485, 0.456, 0.406]))
        self.std = list(checkpoint.get("std", [0.229, 0.224, 0.225]))

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        with self._torch.no_grad():
            x = self._torch.from_numpy(batch).to(self.device)
            logits = self.model(x)
            if self.device.type == "mps":
                self._torch.mps.synchronize()
            return logits.cpu().numpy()
