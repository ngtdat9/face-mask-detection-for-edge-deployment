# Real-Time Face Mask Detection Using OpenCV and PyTorch

> **Note for the student:** this document is the *final report*
> template for the project. Sections that depend on actual training
> results (final accuracy, confusion matrix counts, classification
> report numbers) use placeholders such as `[Insert final accuracy
> here]`. Replace those placeholders with the numbers from your most
> recent `python src/evaluate.py` run before submitting. The image
> placeholders refer to the PNG files saved in `results/` — embed
> those images into your printed/PDF report at the marked positions.

---

## 1. Introduction

Face masks became a part of everyday life during global health
events such as the COVID-19 pandemic. In schools, hospitals, public
transport, and offices, organisations need a quick way to check
whether visitors are wearing a mask. Doing this manually is slow and
unreliable.

**Computer vision** is a natural fit for this task. A camera already
sees every person who enters a room; a small program can read those
camera frames and decide, automatically, whether each visible face is
wearing a mask. This project shows how to build such a program from
scratch using two well-known tools:

- **OpenCV**, which is excellent at finding the location of faces in
  an image.
- **PyTorch**, which is excellent at training a small Convolutional
  Neural Network (CNN) that recognises visual patterns — in our case,
  the difference between a face with a mask and a face without one.

The result is a real-time desktop application that draws a coloured
box and a label like `with_mask 96.4%` around each face the camera
sees.

---

## 2. Problem Statement

The system must **automatically detect whether a person is wearing a
mask or not** from images or live video. More precisely, given an
image (or a video frame), the program should:

1. Locate every face in the image.
2. For each face, decide whether the person is wearing a mask.
3. Show that decision visually (a bounding box and a label) so a
   human can verify it at a glance.

The system must also:

- Work in **real time** on a normal laptop webcam.
- Handle **multiple faces** in the same frame.
- Continue working even when the person is wearing a mask, which
  hides the lower half of the face.

---

## 3. Objectives

- Build a **CNN classifier** (from scratch in PyTorch) that decides
  `with_mask` vs `without_mask` from a single cropped face image.
- Use **OpenCV** to detect faces and feed cropped face regions to the
  classifier.
- Apply a clean, repeatable **preprocessing pipeline** (resize +
  normalize) to every face crop, identical at training and inference
  time.
- Evaluate model performance with **standard metrics** (accuracy,
  precision, recall, F1-score, confusion matrix).
- Wrap everything into a **real-time live-camera application** that
  draws bounding boxes, labels, and confidence scores on screen.

---

## 4. System Design

The system is split into two clearly separated stages. This keeps
each stage simple and lets us improve them independently.

### Stage 1 — Face Detection

The OpenCV DNN face detector (YuNet) takes a frame as input and
returns a **list of bounding boxes**, one per detected face. It does
not look at mask features at all; it only answers *"where are the
faces?"*.

YuNet is used because it was trained on a dataset that includes
occluded faces (masks, sunglasses, hands), making it much more
reliable than the classical Haar Cascade detector when the lower
half of the face is covered.

### Stage 2 — Mask Classification

For each bounding box returned by Stage 1, the system:

1. Crops the face region from the original frame.
2. Resizes it to **128 × 128** pixels.
3. Normalizes the pixel values with the same mean/std used during
   training.
4. Feeds the result to a small **CNN (`MaskCNN`)** trained from
   scratch.

The CNN returns one of two predictions, `with_mask` or
`without_mask`, plus a probability that we display as a confidence
percentage.

### End-to-End Pipeline

```
Input frame
    -> face detection (OpenCV DNN / YuNet)
    -> face crop
    -> preprocessing (resize 128x128 + normalize)
    -> CNN classifier (MaskCNN)
    -> prediction (label + confidence)
    -> display result (bounding box + label on the frame)
```

---

## 5. Dataset Preparation

The starting point is a **raw dataset** with one folder per class:

```
raw_data/
├── with_mask/
└── without_mask/
```

Training a model directly on `raw_data/` would be a mistake: we'd
have no way to tell whether the model actually learned to recognise
masks or just memorised the training images. We therefore split the
data into three sets, each with a specific role:

- **Training set** — used by the optimizer to update model weights.
- **Validation set** — used to choose the best model and tune
  hyperparameters during development.
- **Test set** — used only at the very end to honestly estimate how
  the model would perform on unseen data.

Split ratio used in this project:

- **70 %** training
- **15 %** validation
- **15 %** testing

The split is done by `src/prepare_data.py`. It uses a fixed random
seed so the split is reproducible across runs and never modifies
`raw_data/`. The processed dataset has this layout:

```
data/
├── train/
│   ├── with_mask/
│   └── without_mask/
├── val/
│   ├── with_mask/
│   └── without_mask/
└── test/
    ├── with_mask/
    └── without_mask/
```

This is exactly what `torchvision.datasets.ImageFolder` expects, so
no custom dataset class is needed.

---

## 6. Model Architecture

The classifier is a small CNN built from scratch in
`src/model.py` (`class MaskCNN`). It has two clearly separated parts.

### Feature Extractor (`self.features`)

Four convolutional blocks, each made of:

- `Conv2d`
- `BatchNorm2d`
- `ReLU`
- `MaxPool2d` (halves the spatial size)

| Block        | Channels (in → out) | Spatial size after block |
|--------------|---------------------|--------------------------|
| Conv Block 1 | 3 → 32              | 64 × 64                  |
| Conv Block 2 | 32 → 64             | 32 × 32                  |
| Conv Block 3 | 64 → 128            | 16 × 16                  |
| Conv Block 4 | 128 → 256           | 8 × 8                    |

Input shape is `(batch, 3, 128, 128)`; after the four blocks the
feature map shape is `(batch, 256, 8, 8)`.

### Classifier (`self.classifier`)

- `Flatten` — turns each `(256, 8, 8)` tensor into a 16384-long
  vector.
- `Linear(16384 → 256)`
- `ReLU`
- `Dropout(p = 0.5)` — regularization.
- `Linear(256 → 2)` — final output: 2 logits, one per class.

### Why this split?

The **CNN feature extractor** learns visual patterns from the face
image (edges, textures, shapes such as a mask strap or a covered
mouth). The **fully connected classifier** uses those features to
decide between `with_mask` and `without_mask`.

The model intentionally returns **raw logits**, not probabilities.
This is the right format for `nn.CrossEntropyLoss` during training.
At inference time we apply `softmax` outside the model to obtain
human-friendly probabilities.

---

## 7. Training Process

The training script (`src/train.py`) does the following:

- **Resize** every image to **128 × 128** pixels.
- Apply **data augmentation** to the training set:
  `RandomHorizontalFlip`, `RandomRotation(±10°)`,
  `ColorJitter(brightness ±20 %, contrast ±20 %)`. The validation set
  uses no augmentation, so its metrics are stable across epochs.
- Use **`nn.CrossEntropyLoss`** as the loss function.
- Use the **Adam optimizer** with `lr = 0.001`.
- Track training loss/accuracy and validation loss/accuracy each
  epoch.
- Save the **best model** (highest validation accuracy) to
  `models/mask_classifier.pth`. The checkpoint includes
  `class_to_idx`, `image_size`, `mean`, `std`, and
  `best_val_accuracy`, so downstream scripts can reproduce the exact
  preprocessing.
- Save plots:
  - `results/loss_plot.png` — training and validation loss.
  - `results/accuracy_plot.png` — training and validation accuracy.

Recommended command:

```bash
python src/train.py --epochs 30 --batch_size 32 --lr 0.001 --num_workers 0
```

---

## 8. Evaluation

After training, we run `src/evaluate.py` on the test set to get an
honest estimate of how the model would perform on unseen data.

### Metrics

- **Accuracy** — fraction of all test images classified correctly.
- **Precision** *(per class)* — of the predictions for this class,
  how many were correct.
- **Recall** *(per class)* — of the true examples of this class, how
  many were caught.
- **F1-score** *(per class)* — harmonic mean of precision and recall.
- **Confusion matrix** — counts of `(true class, predicted class)`
  pairs.

### Final results

> **TODO (student):** fill these in from your final
> `python src/evaluate.py` run.

- **Final test accuracy:** [Insert final accuracy here]
- **Macro precision:** [Insert macro precision here]
- **Macro recall:** [Insert macro recall here]
- **Macro F1-score:** [Insert macro F1-score here]
- Per-class numbers: [Insert per-class precision/recall/F1 here]

### Result images

> Insert the four plots saved in the `results/` folder here.

- **`results/loss_plot.png`** — shows how training and validation
  loss evolved over the epochs. Healthy training: both curves go
  down together. Overfitting: train down, val up.
  → [Insert `loss_plot.png` here]

- **`results/accuracy_plot.png`** — shows how training and
  validation accuracy evolved over the epochs. The peak of the
  validation curve is the epoch the best model was saved at.
  → [Insert `accuracy_plot.png` here]

- **`results/confusion_matrix.png`** — a 2 × 2 grid where each cell
  is `(true class, predicted class)`. Strong diagonal = many correct
  predictions; off-diagonal = mistakes.
  → [Insert `confusion_matrix.png` here]

- **`results/classification_report.png`** — a table with precision,
  recall, F1-score, and support per class plus the macro and
  weighted averages.
  → [Insert `classification_report.png` here]

---

## 9. Live Camera Implementation

The real-time application lives in `src/live_camera.py`. It does
exactly what was described in *Section 4 — System Design*, but at
**video rate** (typically 20–30 FPS):

- The webcam **captures frames continuously** through
  `cv2.VideoCapture`.
- The **OpenCV DNN face detector (YuNet)** is run on every frame and
  returns the bounding boxes of all visible faces.
- Each face is **cropped, preprocessed (BGR→RGB → resize 128 × 128
  → normalize), and stacked** into a single batch tensor.
- The CNN classifier is run **once per frame** on the entire batch
  (one forward pass for *all* faces in the frame), producing one
  prediction per face.
- For every face the script draws a coloured **bounding box** and a
  label like `with_mask 96.4%` directly on the frame.
- **Multiple faces** are handled because each detected box is
  processed independently, with its own crop and its own label.
- The bounding box **follows moving faces** because the entire
  detect-classify-draw chain re-runs on every new frame. There is no
  multi-object tracker; we rely on the per-frame detector to find
  the face's new location each time.

Recommended command:

```bash
python src/live_camera.py --camera 0 --detector dnn
```

Press `q` in the camera window to quit.

---

## 10. Debugging and Improvement

### A real bug we encountered

During testing, the live camera worked correctly when the user was
**not** wearing a mask but **failed silently** when the user put on
a real mask: no bounding box was drawn at all. Curiously, the same
program could detect a masked face shown on a phone screen.

**Diagnosis:** the original face detector was an **OpenCV Haar
Cascade**. Haar Cascades work by checking simple intensity patterns
on the face — *"are the eyes darker than the cheeks?"*, *"is the
mouth lighter than the upper lip?"* and so on. These tests assume
the **whole face** is visible. When a real mask covers the mouth and
chin, several of these checks fail, the cascade rejects the
candidate, and **no face** is reported. Because the classifier only
runs *after* a face is detected, no prediction can be made and no
label appears on the frame.

A photo on a phone screen has high-contrast, sharp pixel patterns
around the eyes, which is enough for the early stages of the cascade
to pass. Live, slightly-shadowed mask fabric in a real room does not
have that contrast, so the cascade gives up.

### Fix: use OpenCV DNN face detection (YuNet)

We replaced the Haar Cascade with **OpenCV's YuNet DNN face detector**
(via `cv2.FaceDetectorYN_create`). YuNet was trained on a dataset
that **includes occluded faces** (masks, glasses, partial views), so
it learned to recognise a face from features that are still visible
when a mask is on — the eyes, eyebrows, nose bridge, and overall
face shape.

To verify the fix, both detectors were run on the same set of test
images. YuNet detected a face in every image, including masked ones;
the Haar Cascade missed several of them. The live camera now draws
the green/red bounding box reliably, even when the user is wearing
a real mask.

The script keeps both detectors selectable via `--detector`:

```bash
python src/live_camera.py --detector dnn       # default, robust to masks
python src/live_camera.py --detector haar      # fallback / comparison
```

It also exposes a `--debug` flag that prints how many faces were
detected each frame and draws raw detector boxes in grey before
classification, which makes it easy to tell whether a missing label
is a *detection* problem or a *classification* problem.

---

## 11. Limitations

- The classifier only runs when a face is detected. If detection
  fails (very small face, extreme angle, very bad lighting), no
  prediction can be made for that face.
- Performance depends on lighting and camera quality. Dim or
  back-lit scenes are harder.
- Very small or strongly side-view faces may be missed by the
  detector.
- The model only supports **two classes**: `with_mask` and
  `without_mask`. It does not currently flag masks worn under the
  nose or under the chin.
- Dataset quality directly affects accuracy. Mislabeled or
  low-resolution training images will produce a less reliable
  classifier.

---

## 12. Future Improvements

- **Add an `incorrect_mask` class** so the system distinguishes a
  correctly-worn mask from a mask under the nose / under the chin.
  Because `num_classes` is read from `class_to_idx` in the
  checkpoint, the rest of the pipeline can absorb a third class with
  almost no code changes.
- **Use a larger and more diverse dataset** (more ages, ethnicities,
  mask types, lighting conditions).
- **Use a pretrained model** (ResNet, MobileNet) fine-tuned on the
  mask dataset to push accuracy further.
- **Compare detection backbones**: Haar vs YuNet vs DNN SSD vs
  RetinaFace, both qualitatively and quantitatively (FPS, detection
  rate, false-positive rate).
- **Add face tracking** so each bounding box has a stable ID across
  frames and label flicker is reduced.
- **Improve the deployment interface**: package the live camera as a
  small desktop app or a web app (FastAPI + browser-side video).

---

## 13. Conclusion

This project successfully built a **real-time face mask detection
system** using OpenCV for face detection and a from-scratch PyTorch
CNN for mask vs no-mask recognition. The pipeline is split into two
clean stages (detection and classification), each with its own
script, its own evaluation, and its own documentation in
`docs/phase_*.md`. The system handles multiple faces in real time,
draws labelled bounding boxes with confidence scores, and was made
robust to masked faces by switching from a classical Haar Cascade
detector to OpenCV's modern DNN (YuNet) detector during the
debugging phase.

The final classifier achieves
**[Insert final test accuracy here]** test accuracy with a macro F1
of **[Insert macro F1 here]** on the held-out test set. Training and
evaluation artifacts (`loss_plot.png`, `accuracy_plot.png`,
`confusion_matrix.png`, `classification_report.png`) are saved in
the `results/` folder for inclusion in the final report.

The two-stage design also makes the project easy to extend: a better
detector or a richer set of mask classes can be plugged in without
rewriting the rest of the system.
