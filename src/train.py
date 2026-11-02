"""Train the MaskCNN classifier on the prepared dataset.

This script is the *training stage* of the face mask detection project.
The pipeline is split as follows:

    OpenCV face detector  ->  crop each face  ->  MaskCNN classifier

Here we only train the classifier half. We do NOT train any face
detector -- OpenCV handles that at runtime.

Run from the project root::

    python src/train.py --epochs 30 --batch_size 32 --lr 0.001 --num_workers 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Optional progress bar -- fall back to a no-op wrapper if tqdm is missing.
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

# `python src/train.py` puts `src/` on sys.path, so a flat import works.
# We add an explicit fallback for safety in case the script is launched
# from somewhere unusual.
try:
    from model import HEADS, MaskCNN, count_parameters
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from model import HEADS, MaskCNN, count_parameters  # type: ignore[no-redef]
    except ImportError as exc:
        print(
            "ERROR: could not import MaskCNN from src/model.py.\n"
            "Make sure src/model.py exists and that you are running this "
            "script from the project root with `python src/train.py`.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_SIZE = 128
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------- #
#  Argument parsing                                                           #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the MaskCNN classifier (mask vs no mask).",
    )
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs (default: 30).")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Mini-batch size (default: 32).")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate for Adam (default: 0.001).")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers. Use 0 on macOS (default: 0).")
    parser.add_argument("--train_dir", type=str,
                        default=str(PROJECT_ROOT / "data" / "train"),
                        help="Folder containing the training set.")
    parser.add_argument("--val_dir", type=str,
                        default=str(PROJECT_ROOT / "data" / "val"),
                        help="Folder containing the validation set.")
    parser.add_argument("--model_path", type=str,
                        default=str(PROJECT_ROOT / "models" / "mask_classifier.pth"),
                        help="Where to save the best model checkpoint.")
    parser.add_argument("--results_dir", type=str,
                        default=str(PROJECT_ROOT / "results"),
                        help="Where to save loss / accuracy plots.")
    parser.add_argument("--head", type=str, default="flatten", choices=list(HEADS),
                        help="Classifier head: 'flatten' (original, ~4.6M params) "
                             "or 'gap' (edge-friendly, ~0.39M params).")
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE,
                        help="Square input size (default: 128). The 'flatten' "
                             "head only supports 128.")
    parser.add_argument("--plot_prefix", type=str, default="",
                        help="Prefix for plot filenames, e.g. 'gap_' writes "
                             "gap_loss_plot.png (default: no prefix).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for weight init and shuffling (default: 42).")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
#  Device selection                                                           #
# --------------------------------------------------------------------------- #
def select_device() -> torch.device:
    """Pick the best available device: MPS (Apple Silicon) > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
#  Data wrappers                                                              #
# --------------------------------------------------------------------------- #
def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    """Return (train_transform, val_transform).

    The train transform adds light data augmentation so the model sees a
    slightly different version of each image every epoch. The val
    transform is deterministic: only resize + normalize, so validation
    metrics are comparable across epochs.

    The val transform is also the one we will reuse later in
    `evaluate.py`, `predict_image.py`, and `live_camera.py`, so the
    classifier always sees images preprocessed in exactly the same way.
    """
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

    return train_transform, val_transform


def build_dataloaders(
    train_dir: Path,
    val_dir: Path,
    batch_size: int,
    num_workers: int,
    image_size: int,
) -> tuple[DataLoader, DataLoader, ImageFolder, ImageFolder]:
    """Wrap the train/val folders in ImageFolder + DataLoader.

    ImageFolder reads `<root>/<class>/<image>` directly, so we don't need
    a custom Dataset class for this layout.
    """
    if not train_dir.is_dir():
        print(f"ERROR: training folder not found: {train_dir}", file=sys.stderr)
        print("       Did you run `python src/prepare_data.py` first?",
              file=sys.stderr)
        raise SystemExit(1)
    if not val_dir.is_dir():
        print(f"ERROR: validation folder not found: {val_dir}", file=sys.stderr)
        print("       Did you run `python src/prepare_data.py` first?",
              file=sys.stderr)
        raise SystemExit(1)

    train_transform, val_transform = build_transforms(image_size)

    train_dataset = ImageFolder(root=str(train_dir), transform=train_transform)
    val_dataset = ImageFolder(root=str(val_dir), transform=val_transform)

    if not train_dataset.classes or not val_dataset.classes:
        print("ERROR: no classes found. Each split must contain class "
              "subfolders (e.g. with_mask/, without_mask/).", file=sys.stderr)
        raise SystemExit(1)

    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        print("ERROR: train and val class_to_idx mappings differ. "
              "Re-run `python src/prepare_data.py` to rebuild the splits.",
              file=sys.stderr)
        raise SystemExit(1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, train_dataset, val_dataset


# --------------------------------------------------------------------------- #
#  Train / Validate one epoch                                                 #
# --------------------------------------------------------------------------- #
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    epochs: int,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass: model returns RAW LOGITS (no softmax inside the model).
        logits = model(images)

        # Sanity check on the very first batch.
        if total == 0 and logits.shape[1] != 2:
            raise RuntimeError(
                f"Unexpected model output: got shape {tuple(logits.shape)}, "
                f"expected (batch, 2) for a 2-class classifier."
            )

        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    epochs: int,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs} [val]  ", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# --------------------------------------------------------------------------- #
#  Plotting                                                                   #
# --------------------------------------------------------------------------- #
def save_curve(
    epochs_x: list[int],
    train_values: list[float],
    val_values: list[float],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_x, train_values, marker="o", label=f"train {ylabel}")
    plt.plot(epochs_x, val_values, marker="o", label=f"val {ylabel}")
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()

    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)
    model_path = Path(args.model_path)
    results_dir = Path(args.results_dir)

    if args.head == "flatten" and args.image_size != 128:
        print("ERROR: the 'flatten' head is wired for 128x128 inputs. "
              "Use --head gap for other image sizes.", file=sys.stderr)
        raise SystemExit(1)

    torch.manual_seed(args.seed)

    # Make sure the output folders exist before we start training.
    model_path.parent.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Device selection: MPS > CUDA > CPU.
    device = select_device()
    print(f"Using device: {device}")

    # Data wrappers.
    train_loader, val_loader, train_dataset, _val_dataset = build_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )
    print(f"Found classes: {train_dataset.class_to_idx}")
    print(f"Train images : {len(train_dataset)}")
    print(f"Val images   : {len(_val_dataset)}")

    # Model + loss + optimizer.
    # CrossEntropyLoss expects raw logits, so the model must NOT apply
    # softmax in its forward method (and MaskCNN does not).
    model = MaskCNN(num_classes=len(train_dataset.classes), head=args.head).to(device)
    print(f"Model head   : {args.head}  ({count_parameters(model):,} parameters)")
    print(f"Image size   : {args.image_size}x{args.image_size}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_losses: list[float] = []
    val_losses: list[float] = []
    train_accuracies: list[float] = []
    val_accuracies: list[float] = []

    best_val_acc = 0.0
    best_epoch = -1

    print(f"\nStarting training for {args.epochs} epochs...\n")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch=epoch, epochs=args.epochs,
        )
        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, device,
            epoch=epoch, epochs=args.epochs,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        improved = val_acc > best_val_acc
        marker = "  *new best*" if improved else ""
        print(
            f"Epoch {epoch:>3}/{args.epochs} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}{marker}"
        )

        # Save the best model based on validation accuracy.
        # We save class_to_idx, image_size and normalization stats so
        # evaluate.py / predict_image.py / live_camera.py can use the
        # exact same preprocessing and label mapping at inference time.
        if improved:
            best_val_acc = val_acc
            best_epoch = epoch
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "class_to_idx": train_dataset.class_to_idx,
                "head": args.head,
                "image_size": args.image_size,
                "mean": NORM_MEAN,
                "std": NORM_STD,
                "best_val_accuracy": best_val_acc,
            }
            torch.save(checkpoint, model_path)

    print("\nTraining finished.")
    if best_epoch == -1:
        print("WARNING: no best model was saved (validation never improved).")
    else:
        print(f"Best epoch: {best_epoch}  best val_accuracy: {best_val_acc:.4f}")
        print(f"Best model saved to: {model_path}")

    # Save plots showing how training and validation evolved over epochs.
    epochs_x = list(range(1, args.epochs + 1))
    loss_path = results_dir / f"{args.plot_prefix}loss_plot.png"
    accuracy_path = results_dir / f"{args.plot_prefix}accuracy_plot.png"
    save_curve(
        epochs_x, train_losses, val_losses,
        title="Training and Validation Loss",
        ylabel="loss",
        output_path=loss_path,
    )
    save_curve(
        epochs_x, train_accuracies, val_accuracies,
        title="Training and Validation Accuracy",
        ylabel="accuracy",
        output_path=accuracy_path,
    )
    print(f"Loss plot saved to    : {loss_path}")
    print(f"Accuracy plot saved to: {accuracy_path}")


if __name__ == "__main__":
    main()
