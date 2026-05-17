"""Evaluate the trained MaskCNN classifier on the held-out test set.

This script:

1. Loads the checkpoint saved by ``train.py``
   (``models/mask_classifier.pth``).
2. Wraps ``data/test`` with ``ImageFolder`` and the same preprocessing
   used at validation time.
3. Runs the model on every test image, computes:
       - accuracy
       - precision (per class + macro)
       - recall    (per class + macro)
       - F1-score  (per class + macro)
4. Saves two plots:
       - ``results/confusion_matrix.png``
       - ``results/classification_report.png``

Run from the project root::

    python src/evaluate.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

try:
    from model import MaskCNN
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from model import MaskCNN  # type: ignore[no-redef]
    except ImportError as exc:
        print(
            "ERROR: could not import MaskCNN from src/model.py.\n"
            "Run this script from the project root with `python src/evaluate.py`.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Sensible fallbacks if the checkpoint does not contain these keys.
DEFAULT_IMAGE_SIZE = 128
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------- #
#  Argument parsing                                                           #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the trained MaskCNN on the test set.",
    )
    parser.add_argument("--test_dir", type=str,
                        default=str(PROJECT_ROOT / "data" / "test"),
                        help="Folder with the test set (ImageFolder layout).")
    parser.add_argument("--model_path", type=str,
                        default=str(PROJECT_ROOT / "models" / "mask_classifier.pth"),
                        help="Path to the trained checkpoint.")
    parser.add_argument("--results_dir", type=str,
                        default=str(PROJECT_ROOT / "results"),
                        help="Folder where evaluation plots are written.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Mini-batch size for evaluation (default: 32).")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers. Use 0 on macOS (default: 0).")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
#  Device                                                                     #
# --------------------------------------------------------------------------- #
def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
#  Data + transforms                                                          #
# --------------------------------------------------------------------------- #
def build_eval_transform(image_size: int, mean: list[float], std: list[float]) -> transforms.Compose:
    """Same preprocessing as the validation transform in train.py."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# --------------------------------------------------------------------------- #
#  Plotting helpers                                                           #
# --------------------------------------------------------------------------- #
def save_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    """Render a confusion matrix as a heatmap with counts in each cell."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=15, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    # Annotate each cell with its raw count. Choose text color based
    # on cell value so the number stays readable on dark backgrounds.
    threshold = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            color = "white" if value > threshold else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_classification_report(
    report_dict: dict,
    class_names: list[str],
    output_path: Path,
) -> None:
    """Render sklearn's classification_report (dict form) as a table image.

    Rows: each class + ``accuracy`` + ``macro avg`` + ``weighted avg``.
    Columns: precision, recall, f1-score, support.
    """
    columns = ["precision", "recall", "f1-score", "support"]
    row_keys = list(class_names) + ["accuracy", "macro avg", "weighted avg"]

    cell_text: list[list[str]] = []
    for key in row_keys:
        if key == "accuracy":
            # In sklearn's dict, "accuracy" is a single float, not a sub-dict.
            acc = report_dict.get("accuracy", float("nan"))
            total_support = int(report_dict.get("macro avg", {}).get("support", 0))
            row = ["", "", f"{acc:.4f}", str(total_support)]
        else:
            row_dict = report_dict.get(key, {})
            row = [
                f"{row_dict.get('precision', float('nan')):.4f}",
                f"{row_dict.get('recall', float('nan')):.4f}",
                f"{row_dict.get('f1-score', float('nan')):.4f}",
                str(int(row_dict.get('support', 0))),
            ]
        cell_text.append(row)

    fig, ax = plt.subplots(figsize=(8, 0.6 * (len(row_keys) + 1) + 1))
    ax.axis("off")
    ax.set_title("Classification Report", pad=12)

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_keys,
        colLabels=columns,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Evaluation loop                                                            #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model on every batch and return (y_true, y_pred) as numpy arrays."""
    model.eval()
    all_true: list[int] = []
    all_pred: list[int] = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()

        all_true.extend(labels.numpy().tolist())
        all_pred.extend(preds.tolist())

    return np.array(all_true), np.array(all_pred)


# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()

    test_dir = Path(args.test_dir)
    model_path = Path(args.model_path)
    results_dir = Path(args.results_dir)

    if not test_dir.is_dir():
        print(f"ERROR: test folder not found: {test_dir}", file=sys.stderr)
        print("       Did you run `python src/prepare_data.py` first?",
              file=sys.stderr)
        raise SystemExit(1)

    if not model_path.is_file():
        print(f"ERROR: trained model not found: {model_path}", file=sys.stderr)
        print("       Did you run `python src/train.py` first?",
              file=sys.stderr)
        raise SystemExit(1)

    results_dir.mkdir(parents=True, exist_ok=True)

    device = select_device()
    print(f"Using device: {device}")

    # Load the checkpoint. We use the values it stored at training time
    # (image_size, mean, std, class_to_idx) so the test pipeline matches
    # exactly what the model was trained on.
    print(f"Loading checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    image_size = int(checkpoint.get("image_size", DEFAULT_IMAGE_SIZE))
    mean = list(checkpoint.get("mean", DEFAULT_MEAN))
    std = list(checkpoint.get("std", DEFAULT_STD))
    class_to_idx = checkpoint.get("class_to_idx")
    best_val_acc = checkpoint.get("best_val_accuracy")

    # Class names sorted by their integer index so that index 0 -> name 0, etc.
    if class_to_idx is None:
        class_names = ["with_mask", "without_mask"]
        print("WARNING: checkpoint has no class_to_idx; using default order.")
    else:
        class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]
    print(f"Classes: {class_names}")
    if best_val_acc is not None:
        print(f"Checkpoint best val accuracy: {best_val_acc:.4f}")

    # Build dataset + loader using the SAME transform as validation in train.py.
    eval_transform = build_eval_transform(image_size, mean, std)
    test_dataset = ImageFolder(root=str(test_dir), transform=eval_transform)

    if not test_dataset.classes:
        print("ERROR: no class subfolders found in the test directory.",
              file=sys.stderr)
        raise SystemExit(1)
    if class_to_idx is not None and test_dataset.class_to_idx != class_to_idx:
        print("ERROR: test class_to_idx does not match the checkpoint.",
              file=sys.stderr)
        print(f"  checkpoint: {class_to_idx}", file=sys.stderr)
        print(f"  test set  : {test_dataset.class_to_idx}", file=sys.stderr)
        raise SystemExit(1)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print(f"Test images: {len(test_dataset)}")

    # Build the model, load weights, move to device.
    model = MaskCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Collect predictions over the entire test set.
    y_true, y_pred = collect_predictions(model, test_loader, device)

    # Metrics. ``average="macro"`` treats both classes equally, which is
    # the right choice for a roughly balanced dataset like ours. We
    # also print the per-class numbers via classification_report.
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print()
    print("=" * 50)
    print("Test set results")
    print("=" * 50)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}  (macro average)")
    print(f"Recall   : {recall:.4f}  (macro average)")
    print(f"F1-score : {f1:.4f}  (macro average)")
    print()

    print("Per-class report:")
    print(classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0,
    ))

    # Plots.
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_path = results_dir / "confusion_matrix.png"
    save_confusion_matrix(cm, class_names, cm_path)
    print(f"Confusion matrix saved to    : {cm_path}")

    report_dict = classification_report(
        y_true, y_pred, target_names=class_names,
        output_dict=True, zero_division=0,
    )
    report_path = results_dir / "classification_report.png"
    save_classification_report(report_dict, class_names, report_path)
    print(f"Classification report saved to: {report_path}")


if __name__ == "__main__":
    main()
