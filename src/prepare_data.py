"""Prepare the face mask dataset.

Converts the raw dataset layout::

    raw_data/
        with_mask/
        without_mask/

into a train / val / test split that ``torchvision.datasets.ImageFolder``
can read directly::

    data/
        train/
            with_mask/
            without_mask/
        val/
            with_mask/
            without_mask/
        test/
            with_mask/
            without_mask/

Run from the project root::

    python src/prepare_data.py
"""

from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "raw_data"
DATA_DIR = PROJECT_ROOT / "data"

CLASSES = ("with_mask", "without_mask")
SPLITS = ("train", "val", "test")
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

RANDOM_SEED = 42


def list_images(class_dir: Path) -> list[Path]:
    """Return all image files inside ``class_dir`` (non-recursive, sorted)."""
    if not class_dir.is_dir():
        raise FileNotFoundError(f"Expected class folder not found: {class_dir}")

    images = [
        p
        for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    images.sort()
    return images


def split_indices(n: int, ratios: dict[str, float]) -> dict[str, slice]:
    """Compute slice ranges for each split given ``n`` items.

    Uses floor for train/val so that any rounding remainder ends up in the
    test split. This guarantees the splits cover every item exactly once.
    """
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])
    n_test = n - n_train - n_val
    return {
        "train": slice(0, n_train),
        "val": slice(n_train, n_train + n_val),
        "test": slice(n_train + n_val, n_train + n_val + n_test),
    }


def reset_data_dir(data_dir: Path) -> None:
    """Delete ``data_dir`` if it exists, then recreate the split skeleton."""
    if data_dir.exists():
        shutil.rmtree(data_dir)

    for split in SPLITS:
        for cls in CLASSES:
            (data_dir / split / cls).mkdir(parents=True, exist_ok=True)


def copy_files(files: list[Path], dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        shutil.copy2(src, dest_dir / src.name)


def prepare() -> dict[str, dict[str, int]]:
    """Run the full preparation pipeline and return per-split / per-class counts."""
    if not RAW_DATA_DIR.is_dir():
        print(f"ERROR: raw data folder not found: {RAW_DATA_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Raw data dir : {RAW_DATA_DIR}")
    print(f"Output dir   : {DATA_DIR}")
    print(f"Random seed  : {RANDOM_SEED}")
    print()

    print("Resetting output directory...")
    reset_data_dir(DATA_DIR)

    rng = random.Random(RANDOM_SEED)
    counts: dict[str, dict[str, int]] = {split: {} for split in SPLITS}
    counts["total"] = {}

    for cls in CLASSES:
        class_dir = RAW_DATA_DIR / cls
        images = list_images(class_dir)

        if not images:
            print(f"WARNING: no images found in {class_dir}")

        rng.shuffle(images)
        slices = split_indices(len(images), SPLIT_RATIOS)

        counts["total"][cls] = len(images)
        for split in SPLITS:
            split_files = images[slices[split]]
            copy_files(split_files, DATA_DIR / split / cls)
            counts[split][cls] = len(split_files)

    return counts


def print_summary(counts: dict[str, dict[str, int]]) -> None:
    classes = list(CLASSES)

    header = f"{'split':<8}" + "".join(f"{cls:>16}" for cls in classes) + f"{'total':>10}"
    print()
    print(header)
    print("-" * len(header))

    for split in SPLITS:
        row = f"{split:<8}"
        split_total = 0
        for cls in classes:
            n = counts[split].get(cls, 0)
            split_total += n
            row += f"{n:>16}"
        row += f"{split_total:>10}"
        print(row)

    print("-" * len(header))
    total_row = f"{'total':<8}"
    grand_total = 0
    for cls in classes:
        n = counts["total"].get(cls, 0)
        grand_total += n
        total_row += f"{n:>16}"
    total_row += f"{grand_total:>10}"
    print(total_row)
    print()


def main() -> None:
    counts = prepare()
    print_summary(counts)
    print(f"Done. Dataset written to: {DATA_DIR}")


if __name__ == "__main__":
    main()
