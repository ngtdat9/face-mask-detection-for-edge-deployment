# Phase 2 — Dataset Preparation

In Phase 1 we set up an empty project skeleton. In Phase 2 we turn the
raw images in `raw_data/` into a clean, properly split dataset inside
`data/` that PyTorch can read directly.

This document explains *why* we do each step before describing *how* to
run the script.

---

## 1. Why convert `raw_data/` at all?

When you collect images, the result usually looks like this:

```
raw_data/
├── with_mask/
└── without_mask/
```

That layout is fine for **storing** images, but it is not enough for
**training a model**. There are three problems:

1. **No separation between training and evaluation.**
   If we train on every image we own, we cannot tell whether the model
   actually learned to recognize masks or just memorized the pictures.
2. **No reproducibility.**
   Every time someone shuffles the data differently they get different
   results, which makes experiments impossible to compare.
3. **The framework expects a specific shape.**
   PyTorch's `torchvision.datasets.ImageFolder` needs a folder per
   *class* under a single root. We need that structure for `train/`,
   `val/`, and `test/` separately.

The `prepare_data.py` script solves all three problems at once.

---

## 2. Why split into train, validation, and test?

A machine-learning model sees data in three different roles. Mixing
them is one of the most common (and most damaging) beginner mistakes.

| Split | Used by                          | Purpose                                                                                          |
| ----- | -------------------------------- | ------------------------------------------------------------------------------------------------ |
| Train | The optimizer (gradient descent) | Adjust the model's weights so it learns to predict the correct label.                            |
| Val   | The developer, between epochs    | Tune things we choose by hand (learning rate, model size, when to stop training).                |
| Test  | The developer, only at the end   | Get an honest estimate of how the model will perform on new, unseen images.                      |

Why three sets and not two?

- If we pick hyperparameters by looking at the **test set**, we are
  effectively training on it. Our final accuracy will look great but
  the model will disappoint in the real world.
- The **validation** set acts as a buffer: we can look at it many
  times during development without "spending" the test set.
- The **test** set is opened **once**, at the very end, to report the
  final number.

A common ratio for medium datasets is **70 / 15 / 15**, which is what we
use here. The training set is large enough for the model to learn, and
the val/test sets are large enough to give meaningful metrics.

We also fix a **random seed** (`42`) so the same split is produced every
time. Re-running the script always yields the same train/val/test
images, which makes experiments reproducible.

---

## 3. Why are the class folder names important?

The class folder names are **not** decoration — they are the **labels**.

`torchvision.datasets.ImageFolder` works like this: it scans the root
folder, sorts the subfolder names alphabetically, and assigns each one
an integer index.

For our project the mapping becomes:

| Folder name      | Label index | Meaning                              |
| ---------------- | ----------- | ------------------------------------ |
| `with_mask`      | `0`         | Face is wearing a mask               |
| `without_mask`   | `1`         | Face is not wearing a mask           |

If we rename, mistype, or duplicate a folder, the labels change
silently and the model trains on the wrong target. That is why
`prepare_data.py` uses a fixed list of class names (`with_mask`,
`without_mask`) and writes them into every split with exactly those
spellings.

---

## 4. What `ImageFolder` expects

`ImageFolder` is the most convenient way to load image classification
data in PyTorch. It expects this exact layout:

```
<root>/
├── <class_a>/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── <class_b>/
    ├── image1.jpg
    └── ...
```

Rules:

- One **subfolder per class**, directly under the root.
- Each image file lives **inside its class folder** — no nesting.
- Supported file extensions include `.jpg`, `.jpeg`, `.png`, `.bmp`,
  `.webp`, etc. (Our script also restricts itself to that set.)

We will create one `ImageFolder` per split, pointing at
`data/train/`, `data/val/`, and `data/test/`. Each of those three
folders has the structure shown above.

---

## 5. What `prepare_data.py` does, step by step

The script lives in `src/prepare_data.py`. When you run it, it:

1. Locates `raw_data/` and `data/` from the project root.
2. **Deletes `data/` if it already exists**, so reruns always start
   from a clean slate (no leftovers from a previous split).
3. Recreates the empty skeleton:
   `data/{train,val,test}/{with_mask,without_mask}/`.
4. For each class:
   - Lists every file with a supported image extension
     (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`).
   - Sorts the list (so the starting order is deterministic).
   - Shuffles it with a **fixed random seed** (`42`).
   - Splits it into 70 % train / 15 % val / 15 % test (any rounding
     remainder goes into `test/`, so every image is used exactly once).
   - Copies the images (`shutil.copy2`, preserves metadata) into the
     matching destination folder.
5. Prints a summary table with counts per class, per split, and a grand
   total.

The original `raw_data/` is **never modified** — we only ever read from
it. That way you can re-run `prepare_data.py` as many times as you want
without losing the source images.

---

## 6. How to run it

From the **project root** (the folder that contains `src/`,
`raw_data/`, `data/`, etc.) run:

```bash
python src/prepare_data.py
```

You should see output similar to this:

```
Project root : .../face-mask-detection
Raw data dir : .../face-mask-detection/raw_data
Output dir   : .../face-mask-detection/data
Random seed  : 42

Resetting output directory...

split          with_mask    without_mask     total
--------------------------------------------------
train               2607            2679      5286
val                  558             574      1132
test                 560             575      1135
--------------------------------------------------
total               3725            3828      7553

Done. Dataset written to: .../face-mask-detection/data
```

(Your numbers will depend on how many images are in your `raw_data/`
folders, but the proportions should always be roughly 70 / 15 / 15.)

> Always run the script **from the project root**, not from inside
> `src/`. The script computes the project root from its own file path,
> so the absolute paths it uses are correct, but running it from the
> project root keeps your shell consistent with the rest of the
> tutorial commands.

---

## 7. What the final `data/` folder looks like

After Phase 2 succeeds, `data/` looks exactly like this:

```
data/
├── train/
│   ├── with_mask/
│   │   ├── img_0001.jpg
│   │   ├── img_0002.jpg
│   │   └── ...
│   └── without_mask/
│       ├── img_0001.jpg
│       ├── img_0002.jpg
│       └── ...
├── val/
│   ├── with_mask/
│   │   └── ...
│   └── without_mask/
│       └── ...
└── test/
    ├── with_mask/
    │   └── ...
    └── without_mask/
        └── ...
```

Properties you should expect:

- Every image from `raw_data/` appears in **exactly one** of the three
  splits — no duplicates and no images left behind.
- Each split contains both classes, with roughly the same class
  proportions as the original dataset (because we split each class
  independently, not the whole dataset at once).
- The split is **stable** across runs, thanks to the fixed seed.

---

## 8. Troubleshooting

- **`raw data folder not found`** — you are probably running the
  script from inside `src/`. Go back to the project root and try
  again.
- **`no images found in ...`** — check that the class folder really
  contains files with one of the supported extensions
  (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`).
- **Train/val/test counts look off by 1** — that's expected. When the
  number of images per class is not perfectly divisible by the split
  ratios, the rounding remainder is added to the test set so that
  every image is used exactly once.

---

## 9. What's next

In Phase 3 we will define the **CNN architecture** in PyTorch and write
the **training loop** that consumes the `data/train/` and `data/val/`
folders we just created via `ImageFolder` and `DataLoader`.
