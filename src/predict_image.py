"""Single-image prediction with the trained MaskCNN classifier.

This script does **not** detect faces. It is the classifier half of the
two-stage face mask detection pipeline:

    OpenCV face detector  ->  crop each face  ->  MaskCNN classifier
                                                 ^^^^^^^^^^^^^^^^^^
                                                 (this script tests it)

Given a path to a single image (ideally an already-cropped face), the
script loads the trained checkpoint, runs the model, and prints the
predicted label, the confidence, the true label (if known), and
whether the prediction is correct.

Run from the project root::

    python src/predict_image.py --image data/test/with_mask/example.jpg

    python src/predict_image.py --image test_images/person1.jpg \
                                --true_label without_mask
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

# Robust import of MaskCNN whether the script is launched as
# `python src/predict_image.py` (from the project root) or as
# `python predict_image.py` from inside `src/`.
try:
    from model import MaskCNN
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from model import MaskCNN  # type: ignore[no-redef]
    except ImportError as exc:
        print(
            "ERROR: could not import MaskCNN from src/model.py.\n"
            "Run this script from the project root:\n"
            "    python src/predict_image.py --image <path>",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


PROJECT_ROOT = Path(__file__).resolve().parent.parent

VALID_LABELS = ("with_mask", "without_mask")

# Sensible fallbacks if the checkpoint is missing any of these keys.
DEFAULT_IMAGE_SIZE = 128
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------- #
#  Argument parsing                                                           #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the trained MaskCNN classifier on a single image.",
    )
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the image to classify.")
    parser.add_argument("--model_path", type=str,
                        default=str(PROJECT_ROOT / "models" / "mask_classifier.pth"),
                        help="Path to the trained checkpoint.")
    parser.add_argument("--true_label", type=str, default=None,
                        choices=list(VALID_LABELS),
                        help="Optional ground-truth label for the image.")
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
#  True label inference                                                       #
# --------------------------------------------------------------------------- #
def infer_true_label_from_path(image_path: Path) -> str:
    """Infer the true label from the parent folder name.

    Examples
    --------
        data/test/with_mask/img.jpg     -> "with_mask"
        data/test/without_mask/img.jpg  -> "without_mask"
        any/other/folder/img.jpg        -> "Unknown"
    """
    parent_name = image_path.parent.name
    if parent_name in VALID_LABELS:
        return parent_name
    return "Unknown"


# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()

    image_path = Path(args.image)
    model_path = Path(args.model_path)

    if not image_path.is_file():
        print(f"ERROR: image not found: {image_path}", file=sys.stderr)
        raise SystemExit(1)

    if not model_path.is_file():
        print(f"ERROR: model checkpoint not found: {model_path}", file=sys.stderr)
        print("       Train the model first with `python src/train.py`.",
              file=sys.stderr)
        raise SystemExit(1)

    device = select_device()
    print(f"Using device: {device}")

    # ---- Load checkpoint --------------------------------------------------
    # The checkpoint stores everything we need to recreate the exact
    # preprocessing the model was trained on, plus the class order.
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as exc:
        print(f"ERROR: failed to load checkpoint {model_path}: {exc}",
              file=sys.stderr)
        raise SystemExit(1) from exc

    if "model_state_dict" not in checkpoint:
        print("ERROR: checkpoint is missing 'model_state_dict'.", file=sys.stderr)
        raise SystemExit(1)

    class_to_idx = checkpoint.get("class_to_idx")
    if class_to_idx is None:
        print("ERROR: checkpoint is missing 'class_to_idx'. "
              "Re-run training with the current train.py to regenerate it.",
              file=sys.stderr)
        raise SystemExit(1)

    image_size = int(checkpoint.get("image_size", DEFAULT_IMAGE_SIZE))
    mean = list(checkpoint.get("mean", DEFAULT_MEAN))
    std = list(checkpoint.get("std", DEFAULT_STD))

    # idx_to_class is the inverse mapping. We rely on it (instead of any
    # hardcoded order) so the predicted index always maps to the right
    # class name.
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    # ---- Build and load the model ----------------------------------------
    try:
        model = MaskCNN(num_classes=len(class_to_idx)).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
    except Exception as exc:
        print(f"ERROR: failed to load model weights: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    # `eval()` disables Dropout and uses BatchNorm running statistics.
    model.eval()

    # ---- Preprocess the image --------------------------------------------
    # We mirror the validation transform from train.py: Resize -> ToTensor
    # -> Normalize. No augmentation here -- predictions must be deterministic.
    try:
        pil_image = Image.open(image_path).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        print(f"ERROR: could not open image {image_path}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # `unsqueeze(0)` adds a batch dimension: (3, H, W) -> (1, 3, H, W)
    input_tensor = preprocess(pil_image).unsqueeze(0).to(device)

    # ---- Prediction -------------------------------------------------------
    # The model returns RAW LOGITS. We apply softmax HERE (at inference
    # time) to turn the logits into probabilities so we can show a
    # human-friendly confidence percentage. Softmax is intentionally NOT
    # part of the model so training with CrossEntropyLoss stays correct.
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1)
        confidence_value, predicted_index = torch.max(probabilities, dim=1)

    predicted_idx = int(predicted_index.item())
    predicted_label = idx_to_class[predicted_idx]
    confidence_pct = float(confidence_value.item()) * 100.0

    # ---- True label & correctness ----------------------------------------
    if args.true_label is not None:
        true_label = args.true_label
    else:
        true_label = infer_true_label_from_path(image_path)

    if true_label == "Unknown":
        correctness = "Not available"
    elif true_label == predicted_label:
        correctness = "Yes"
    else:
        correctness = "No"

    # ---- Output -----------------------------------------------------------
    print()
    print(f"Image: {image_path}")
    print(f"True label: {true_label}")
    print(f"Predicted label: {predicted_label}")
    print(f"Confidence: {confidence_pct:.1f}%")
    print(f"Correct: {correctness}")


if __name__ == "__main__":
    main()
