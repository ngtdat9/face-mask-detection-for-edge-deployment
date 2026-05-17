"""Real-time face mask detection from a webcam.

This script wires the two halves of the project together:

    OpenCV face detector  ->  crop each face  ->  MaskCNN classifier

For every frame from the webcam we:

1. Run an OpenCV face detector (YuNet by default, Haar as a fallback).
   The detector finds *where* the faces are and returns bounding boxes.
2. Crop each face from the original BGR frame, convert it to RGB,
   resize to 128x128, and normalize it with the same mean/std the
   model was trained on.
3. Run the PyTorch MaskCNN classifier on the batch of face crops to
   get one prediction per face.
4. Draw a colored bounding box and a label like ``with_mask 96.4%``
   around each face.

Important
---------
The classifier only runs *after* a face is detected. If the face
detector cannot find the face, the classifier is never called.

Haar Cascades work well for fully visible frontal faces but **often
fail when the lower half of the face is covered by a mask**, because
they were trained on examples that include the mouth and chin. We
therefore use OpenCV's **YuNet** DNN face detector by default, which
is far more robust to occluded faces (masks, glasses, hands).

Press ``q`` (with the camera window focused) to quit.

Run from the project root::

    python src/live_camera.py --camera 0 --detector dnn
    python src/live_camera.py --camera 0 --detector haar
    python src/live_camera.py --camera 0 --detector dnn --debug
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

try:
    from model import MaskCNN
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from model import MaskCNN  # type: ignore[no-redef]
    except ImportError as exc:
        print(
            "ERROR: could not import MaskCNN from src/model.py.\n"
            "Run this script from the project root with `python src/live_camera.py`.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_IMAGE_SIZE = 128
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]

# OpenCV uses BGR for colors. Pick green for "good" (with_mask) and
# red for "bad" (without_mask). Other classes fall back to yellow.
COLORS_BGR = {
    "with_mask": (0, 200, 0),       # green
    "without_mask": (0, 0, 220),    # red
}
DEFAULT_COLOR_BGR = (0, 200, 220)   # yellow-ish
DEBUG_BOX_COLOR = (200, 200, 200)   # grey for "face detected, no label yet"

# YuNet (the default DNN detector) ships as a small ONNX file. The
# script downloads it to `models/` on first run if it isn't there.
YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"


# --------------------------------------------------------------------------- #
#  Argument parsing                                                           #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time face mask detection (OpenCV + MaskCNN).",
    )
    parser.add_argument("--camera", type=int, default=0,
                        help="Webcam index (default: 0). "
                             "Try 1 or 2 if the wrong camera opens.")
    parser.add_argument("--detector", type=str, default="dnn",
                        choices=["dnn", "haar"],
                        help="Face detector to use (default: dnn). "
                             "'dnn' = YuNet (robust to masks); "
                             "'haar' = classical Haar Cascade.")
    parser.add_argument("--model_path", type=str,
                        default=str(PROJECT_ROOT / "models" / "mask_classifier.pth"),
                        help="Path to the trained classifier checkpoint.")
    parser.add_argument("--cascade_path", type=str, default=None,
                        help="Optional path to a Haar Cascade XML file "
                             "(used when --detector haar).")
    parser.add_argument("--dnn_model_path", type=str, default=None,
                        help="Optional path to the YuNet ONNX file "
                             "(used when --detector dnn). Auto-downloaded "
                             "to models/ on first run if missing.")
    parser.add_argument("--min_face_size", type=int, default=40,
                        help="Minimum face size in pixels (default: 40).")
    parser.add_argument("--score_threshold", type=float, default=0.7,
                        help="YuNet detection score threshold (default: 0.7).")
    parser.add_argument("--debug", action="store_true",
                        help="Print per-frame face counts and draw "
                             "detection boxes even before classification.")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
#  Device                                                                     #
# --------------------------------------------------------------------------- #
def select_device() -> torch.device:
    """Pick the best device: MPS (Apple Silicon) > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
#  Loading classifier checkpoint                                              #
# --------------------------------------------------------------------------- #
def load_model_from_checkpoint(
    model_path: Path,
    device: torch.device,
) -> tuple[MaskCNN, dict[int, str], int, list[float], list[float]]:
    """Load checkpoint and return (model, idx_to_class, image_size, mean, std)."""
    if not model_path.is_file():
        print(f"ERROR: model checkpoint not found: {model_path}", file=sys.stderr)
        print("       Train the model first with `python src/train.py`.",
              file=sys.stderr)
        raise SystemExit(1)

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as exc:
        print(f"ERROR: failed to load checkpoint {model_path}: {exc}",
              file=sys.stderr)
        raise SystemExit(1) from exc

    class_to_idx = checkpoint.get("class_to_idx")
    if class_to_idx is None:
        print("ERROR: checkpoint is missing 'class_to_idx'. "
              "Re-run training with the current train.py to regenerate it.",
              file=sys.stderr)
        raise SystemExit(1)

    image_size = int(checkpoint.get("image_size", DEFAULT_IMAGE_SIZE))
    mean = list(checkpoint.get("mean", DEFAULT_MEAN))
    std = list(checkpoint.get("std", DEFAULT_STD))
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    model = MaskCNN(num_classes=len(class_to_idx)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, idx_to_class, image_size, mean, std


# --------------------------------------------------------------------------- #
#  Face detectors                                                             #
# --------------------------------------------------------------------------- #
class HaarFaceDetector:
    """Classical OpenCV Haar Cascade face detector.

    Note: Haar cascades are trained on fully visible frontal faces and
    tend to **miss faces whose mouth and chin are covered by a mask**,
    because the cascade relies on intensity patterns around the lower
    face. Use the YuNet detector for masked-face robustness.
    """

    def __init__(self, cascade_path: str | None, min_face_size: int) -> None:
        if cascade_path is None:
            cascade_path = (
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
        self._detector = cv2.CascadeClassifier(cascade_path)
        if self._detector.empty():
            print(f"ERROR: could not load Haar Cascade XML: {cascade_path}",
                  file=sys.stderr)
            raise SystemExit(1)
        # More sensitive parameters than the textbook defaults so we
        # catch borderline / partially occluded faces a bit better.
        self._scale_factor = 1.05
        self._min_neighbors = 3
        self._min_size = (min_face_size, min_face_size)

    def detect(self, frame_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._detector.detectMultiScale(
            gray,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=self._min_size,
        )
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


class YuNetFaceDetector:
    """OpenCV YuNet DNN face detector.

    YuNet is a small, modern face detector (~230 KB ONNX) built into
    OpenCV via ``cv2.FaceDetectorYN_create``. It is significantly more
    robust to **occluded faces** (masks, sunglasses, hands) than Haar
    Cascades, which is exactly the case we hit in this project.
    """

    def __init__(
        self,
        model_path: Path,
        score_threshold: float = 0.7,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        if not hasattr(cv2, "FaceDetectorYN_create"):
            print("ERROR: your OpenCV does not support YuNet "
                  "(cv2.FaceDetectorYN_create is missing).\n"
                  "       Upgrade with `pip install --upgrade opencv-python`, "
                  "or fall back to `--detector haar`.", file=sys.stderr)
            raise SystemExit(1)

        self._detector = cv2.FaceDetectorYN_create(
            str(model_path),
            "",
            (320, 320),  # placeholder; we set the real size per-frame below
            score_threshold,
            nms_threshold,
            top_k,
        )
        self._size: tuple[int, int] | None = None

    def detect(self, frame_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = frame_bgr.shape[:2]
        # YuNet expects to know the input size up front. We update it
        # only when the frame size changes (which is essentially never
        # during a single run).
        if self._size != (w, h):
            self._detector.setInputSize((w, h))
            self._size = (w, h)

        _retval, faces = self._detector.detect(frame_bgr)
        if faces is None:
            return []

        boxes: list[tuple[int, int, int, int]] = []
        for f in faces:
            # YuNet returns: [x, y, w, h, 5x landmarks (x,y), score].
            x, y, fw, fh = int(round(f[0])), int(round(f[1])), int(round(f[2])), int(round(f[3]))
            if fw <= 0 or fh <= 0:
                continue
            boxes.append((x, y, fw, fh))
        return boxes


def ensure_yunet_model(custom_path: str | None) -> Path:
    """Locate (or download) the YuNet ONNX file and return its path."""
    if custom_path is not None:
        path = Path(custom_path)
        if not path.is_file():
            print(f"ERROR: YuNet ONNX file not found: {path}", file=sys.stderr)
            raise SystemExit(1)
        return path

    target = PROJECT_ROOT / "models" / YUNET_MODEL_FILENAME
    if target.is_file():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"YuNet model not found locally; downloading to {target} ...")
    try:
        urllib.request.urlretrieve(YUNET_MODEL_URL, target)
    except (urllib.error.URLError, OSError) as exc:
        print(f"ERROR: could not download YuNet model from {YUNET_MODEL_URL}\n"
              f"       {exc}\n"
              f"       Either reconnect to the internet or download the file "
              f"manually to {target}, or fall back to `--detector haar`.",
              file=sys.stderr)
        raise SystemExit(1) from exc
    print("Download complete.")
    return target


def build_face_detector(args: argparse.Namespace) -> HaarFaceDetector | YuNetFaceDetector:
    """Build the face detector requested on the command line."""
    if args.detector == "haar":
        return HaarFaceDetector(args.cascade_path, args.min_face_size)
    if args.detector == "dnn":
        model_path = ensure_yunet_model(args.dnn_model_path)
        return YuNetFaceDetector(model_path, score_threshold=args.score_threshold)
    raise ValueError(f"Unknown detector: {args.detector}")


# --------------------------------------------------------------------------- #
#  Per-face preprocessing + drawing                                           #
# --------------------------------------------------------------------------- #
def build_preprocess(image_size: int, mean: list[float], std: list[float]) -> transforms.Compose:
    """Same recipe used during training validation / single-image prediction."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def crop_to_tensor(
    frame_bgr: np.ndarray,
    box: tuple[int, int, int, int],
    preprocess: transforms.Compose,
) -> torch.Tensor | None:
    """Crop one face, convert BGR->RGB, resize+normalize, return (3,H,W)."""
    x, y, w, h = box
    frame_h, frame_w = frame_bgr.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame_w, x + w)
    y2 = min(frame_h, y + h)
    if x2 <= x1 or y2 <= y1:
        return None

    face_bgr = frame_bgr[y1:y2, x1:x2]
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

    pil_face = Image.fromarray(face_rgb)
    return preprocess(pil_face)


def draw_label(
    frame_bgr: np.ndarray,
    box: tuple[int, int, int, int],
    label: str,
    confidence: float,
) -> None:
    """Draw the bounding box and ``label conf%`` text on the frame."""
    x, y, w, h = box
    color = COLORS_BGR.get(label, DEFAULT_COLOR_BGR)

    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)

    text = f"{label} {confidence * 100:.1f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    y_text_top = y - text_h - baseline - 4
    if y_text_top < 0:
        y_text_top = y + 2
    cv2.rectangle(
        frame_bgr,
        (x, y_text_top),
        (x + text_w + 6, y_text_top + text_h + baseline + 4),
        color, cv2.FILLED,
    )
    cv2.putText(
        frame_bgr, text,
        (x + 3, y_text_top + text_h + 2),
        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
    )


def draw_debug_box(
    frame_bgr: np.ndarray,
    box: tuple[int, int, int, int],
) -> None:
    """Draw a thin grey rectangle for raw detector output (debug mode)."""
    x, y, w, h = box
    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), DEBUG_BOX_COLOR, 1)


# --------------------------------------------------------------------------- #
#  Main loop                                                                  #
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()

    device = select_device()
    print(f"Using device: {device}")

    # Load the trained classifier first so a missing checkpoint fails
    # fast, before we touch the camera or download the YuNet model.
    model, idx_to_class, image_size, mean, std = load_model_from_checkpoint(
        Path(args.model_path), device,
    )
    print(f"Loaded checkpoint: {args.model_path}")
    print(f"Classes: {idx_to_class}")

    # Build the face detector chosen by --detector.
    face_detector = build_face_detector(args)
    detector_name = "YuNet (DNN)" if args.detector == "dnn" else "Haar Cascade"
    print(f"Face detector: {detector_name}")
    if args.debug:
        print("Debug mode: ON (per-frame face counts + raw detection boxes).")

    preprocess = build_preprocess(image_size, mean, std)

    # Open the webcam. cv2.VideoCapture(0) is usually the default camera.
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: could not open camera index {args.camera}.",
              file=sys.stderr)
        print("       Try a different index with `--camera 1` (or 2).",
              file=sys.stderr)
        print("       On macOS, also check System Settings -> Privacy & "
              "Security -> Camera and grant access to your terminal/IDE.",
              file=sys.stderr)
        raise SystemExit(1)

    print("Camera opened. Press 'q' in the camera window to quit.")
    window_name = "Face Mask Detection (press q to quit)"

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("WARNING: failed to read frame from camera.",
                      file=sys.stderr)
                break

            # 1. Detect faces. The classifier only runs after this step,
            #    so if no faces are detected, no labels are drawn.
            boxes = face_detector.detect(frame)
            frame_idx += 1

            if args.debug:
                print(f"Frame {frame_idx}: {len(boxes)} face(s) detected")
                if not boxes:
                    print("  WARNING: no faces detected this frame.",
                          file=sys.stderr)
                # Draw raw detection rectangles in grey so the user can
                # see whether the detector is firing even when the
                # classifier hasn't been called yet.
                for box in boxes:
                    draw_debug_box(frame, box)

            # 2. Build a batch of preprocessed face tensors.
            tensors: list[torch.Tensor] = []
            valid_boxes: list[tuple[int, int, int, int]] = []
            for box in boxes:
                tensor = crop_to_tensor(frame, box, preprocess)
                if tensor is not None:
                    tensors.append(tensor)
                    valid_boxes.append(box)

            # 3. Classify -- one forward pass for all faces in this frame.
            if tensors:
                batch = torch.stack(tensors, dim=0).to(device)
                with torch.no_grad():
                    logits = model(batch)
                    # Softmax converts logits into probabilities. We
                    # apply it here at inference time only -- never
                    # inside the model -- so training with
                    # CrossEntropyLoss stays numerically stable.
                    probabilities = F.softmax(logits, dim=1)
                    confidence_values, predicted_indices = torch.max(
                        probabilities, dim=1,
                    )

                # 4. Draw one labeled bounding box per face. Boxes are
                #    redrawn every frame, so they follow moving faces.
                for box, idx, conf in zip(
                    valid_boxes,
                    predicted_indices.cpu().tolist(),
                    confidence_values.cpu().tolist(),
                ):
                    label = idx_to_class.get(int(idx), str(idx))
                    draw_label(frame, box, label, float(conf))

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
