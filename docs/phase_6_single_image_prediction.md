# Phase 6 — Single Image Prediction

In Phase 5 we measured how well the trained model performs across the
whole test set. In Phase 6 we zoom in to **one image at a time**:
load it, run the classifier, and print the predicted class with a
confidence score.

The script lives in `src/predict_image.py`. This document explains
what it does and why.

---

## 1. What does Phase 6 do?

`predict_image.py` is a small command-line tool that:

1. Loads the trained checkpoint from `models/mask_classifier.pth`.
2. Opens a single image file you point it at.
3. Resizes, normalizes, and feeds that image to the model.
4. Prints the **predicted label** (`with_mask` / `without_mask`), the
   **confidence** (a probability percentage), the **true label**
   (when known), and whether the prediction is **correct**.

It does **not** detect faces, draw anything, or process video. It is
purely a sanity-check tool for the classifier.

---

## 2. Why single image prediction is useful before live camera

Jumping straight from training to a webcam demo is risky: if anything
is wrong (preprocessing, label mapping, model weights), the camera
window is the worst place to debug it. Frames flash by, OpenCV
windows are hard to inspect, and you can't tell whether the bug is in
the **detector** (face cropping) or the **classifier** (mask vs no
mask).

Single-image prediction is a **controlled experiment**:

- One known image, one expected answer.
- No camera, no face detector — just the classifier.
- You can reproduce the run a thousand times and the answer will be
  the same.

If `predict_image.py` works correctly on test images, we know the
classifier and its preprocessing are correct. Then in Phase 7 we can
focus 100 % on plumbing OpenCV face detection on top of it.

---

## 3. Why this script does not detect faces

The whole project deliberately splits the problem in two:

```
OpenCV face detector  ->  crop each face  ->  MaskCNN classifier
                                              ^^^^^^^^^^^^^^^^^^
                                              Phase 6 lives here
```

`predict_image.py` only runs the classifier half. It assumes the
image you give it already shows roughly **one face filling the
frame** — exactly what the OpenCV detector will hand to the
classifier in Phase 7.

This separation is intentional: it keeps each piece simple,
debuggable, and independently improvable.

---

## 4. Why the input should ideally be a cropped face image

The model was trained on the dataset in `data/train/`, which is full
of **already-cropped faces** (the original `raw_data/` images). It
never saw a full street scene during training, so:

- If you feed it `data/test/with_mask/img.jpg` (a tight face crop),
  it will work very well. That is what we test in Phase 6.
- If you feed it a wide photo with several people and a lot of
  background, the model will resize the entire scene to 128×128 and
  classify the *whole image* as if it were one face. Results in this
  case are not meaningful.

In short: this script is for **face crops**. Anything else is outside
the model's training distribution.

---

## 5. How the model checkpoint is loaded

`train.py` saved a Python dictionary that contains everything we need
to reproduce the training-time pipeline:

```python
{
    "model_state_dict": ...,          # the learned weights
    "class_to_idx": {"with_mask": 0, "without_mask": 1},
    "image_size": 128,
    "mean": [0.485, 0.456, 0.406],
    "std":  [0.229, 0.224, 0.225],
    "best_val_accuracy": 0.985,
}
```

`predict_image.py` does:

```python
checkpoint = torch.load(model_path, map_location=device)
model = MaskCNN(num_classes=len(checkpoint["class_to_idx"]))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

`map_location=device` makes the load work on whatever device we have
(MPS / CUDA / CPU). `model.eval()` switches off Dropout and uses
running statistics for BatchNorm — both important so predictions are
deterministic.

---

## 6. Why `class_to_idx` is saved and reused

`ImageFolder` assigns label indices alphabetically. For this project
that gives:

```
{"with_mask": 0, "without_mask": 1}
```

If we ever rebuilt the dataset with different folder names, that
mapping could change. By **saving it inside the checkpoint** and
reading it back here, we eliminate any guessing:

```python
class_to_idx = checkpoint["class_to_idx"]
idx_to_class = {idx: name for name, idx in class_to_idx.items()}
```

`idx_to_class` is the inverse mapping. After the model returns logits,
the highest-scoring index is looked up in `idx_to_class` to get the
human-readable label. We **never** hardcode `0 = with_mask` anywhere
in the code — that would be brittle and easy to break later.

---

## 7. How the image is preprocessed

To get a consistent, comparable prediction, the input must be
preprocessed in *exactly* the same way the validation images were
during training. The script applies:

```python
preprocess = transforms.Compose([
    transforms.Resize((image_size, image_size)),  # from checkpoint, e.g. 128
    transforms.ToTensor(),                        # uint8 [0,255] -> float [0,1]
    transforms.Normalize(mean=mean, std=std),     # from checkpoint
])

input_tensor = preprocess(pil_image).unsqueeze(0).to(device)
```

Why each step?

- `Resize((128, 128))` — the model has a fixed input shape.
- `ToTensor()` — converts a PIL image into a `(3, H, W)` float tensor
  with values in `[0, 1]`.
- `Normalize(mean, std)` — re-centers pixel values to match the
  distribution the model was trained on.
- `unsqueeze(0)` — adds a batch dimension: `(3, H, W)` becomes
  `(1, 3, H, W)`. PyTorch models always expect a batch axis.
- `.to(device)` — moves the tensor to MPS / CUDA / CPU so it lives
  on the same device as the model.

There is **no augmentation** here (no random flip, rotation, or
ColorJitter). Predictions must be deterministic.

---

## 8. What are logits?

The model returns **logits** — raw, unbounded numbers, one per class.
For a 2-class model the output shape is `(1, 2)`:

```
logits = [[ 4.71, -3.05]]
```

The numbers can be any real value. They are *not* probabilities; you
can't just read them as percentages. Their **relative magnitudes**
encode the model's preference: the higher logit "wins".

We deliberately keep the model output in logit form so training with
`CrossEntropyLoss` stays numerically stable.

---

## 9. What does softmax do?

`softmax` turns a vector of logits into a probability distribution
that sums to 1:

```
prob_i = exp(logit_i) / sum(exp(logit_j) for all j)
```

For our example logits:

```
softmax([4.71, -3.05]) ≈ [0.9996, 0.0004]
```

So we now have a real probability per class. We apply softmax **at
inference time only** — never inside the model — because the loss
function during training already includes a softmax internally.

In code:

```python
probabilities = F.softmax(logits, dim=1)
confidence_value, predicted_index = torch.max(probabilities, dim=1)
```

`torch.max(..., dim=1)` returns both the highest probability and the
index where it occurred — exactly what we need.

---

## 10. What does *confidence* mean?

The **confidence** is the probability the model assigns to its own
prediction:

```
confidence = max(softmax(logits))
```

We display it as a percentage (`96.4%`). High confidence means the
model is very sure; low confidence (around 50 % for two classes)
means the image is borderline.

> Important: a high confidence does **not** guarantee the answer is
> correct. The model can be confidently wrong. Always sanity-check
> with the test set, not with cherry-picked single images.

---

## 11. How the script gets the true label from the parent folder

If you don't pass `--true_label`, the script looks at the **parent
folder** of the image:

```
data/test/with_mask/img_001.jpg     ->  parent name = "with_mask"
data/test/without_mask/img_002.jpg  ->  parent name = "without_mask"
random/folder/photo.jpg              ->  parent name = "folder"
```

The rule is:

```python
if parent_name in {"with_mask", "without_mask"}:
    true_label = parent_name
else:
    true_label = "Unknown"
```

This means it works automatically on every image inside `data/test/`
without you typing the label by hand.

---

## 12. How to manually provide the true label

If your image is somewhere else (e.g. a photo you took with your
phone, or a downloaded picture), you can pass the true label
explicitly with `--true_label`:

```bash
python src/predict_image.py \
    --image my_photos/dad_with_mask.jpg \
    --true_label with_mask
```

Allowed values are `with_mask` and `without_mask`. A user-provided
label always **overrides** the parent-folder inference, which is
useful if a folder happens to be named oddly.

---

## 13. How `Correct: Yes / No / Not available` is calculated

After we have the predicted label and the true label:

| True label                        | Logic                              | Output line             |
| --------------------------------- | ---------------------------------- | ----------------------- |
| `"Unknown"`                       | nothing to compare against         | `Correct: Not available`|
| matches predicted label           | `true_label == predicted_label`    | `Correct: Yes`          |
| differs from predicted label      | `true_label != predicted_label`    | `Correct: No`           |

So you can use the script in two modes:

- **Diagnostic mode**: feed it images from `data/test/` and
  immediately see whether each one is correctly classified.
- **Demo mode**: feed it any new photo, pass the right
  `--true_label`, and verify the model still works in the wild.

---

## 14. Example commands

Test on a known masked face from the test set (true label inferred):

```bash
python src/predict_image.py --image data/test/with_mask/with_mask_100.jpg
```

Test on a known unmasked face from the test set:

```bash
python src/predict_image.py --image data/test/without_mask/without_mask_100.jpg
```

Test on an image outside the dataset, with the true label given by
hand:

```bash
python src/predict_image.py \
    --image my_photos/selfie.jpg \
    --true_label without_mask
```

Use a different checkpoint path (e.g. an experiment in another
folder):

```bash
python src/predict_image.py \
    --image data/test/with_mask/with_mask_100.jpg \
    --model_path experiments/run_2/best.pth
```

---

## 15. Example outputs

Correct prediction on a test-set image (true label inferred from
folder):

```
Using device: mps

Image: data/test/with_mask/with_mask_100.jpg
True label: with_mask
Predicted label: with_mask
Confidence: 86.9%
Correct: Yes
```

Correct prediction with very high confidence:

```
Using device: mps

Image: data/test/without_mask/without_mask_100.jpg
True label: without_mask
Predicted label: without_mask
Confidence: 100.0%
Correct: Yes
```

Incorrect prediction (manually wrong `--true_label` to demonstrate
how mismatches are reported):

```
Using device: mps

Image: data/test/with_mask/with_mask_100.jpg
True label: without_mask
Predicted label: with_mask
Confidence: 86.9%
Correct: No
```

Image outside the dataset with no true label given:

```
Using device: mps

Image: /tmp/random.jpg
True label: Unknown
Predicted label: with_mask
Confidence: 86.9%
Correct: Not available
```

---

## 16. Common problems and fixes

- **`ERROR: image not found: <path>`**
  The path you passed does not exist or is misspelled. Use
  tab-completion or `ls` to confirm the file is there.

- **`ERROR: model checkpoint not found`**
  You haven't trained yet. Run `python src/train.py` (Phase 4) first
  so `models/mask_classifier.pth` is created.

- **`ERROR: checkpoint is missing 'class_to_idx'`**
  The checkpoint was saved by an older version of `train.py`. Re-run
  Phase 4 so the new checkpoint includes `class_to_idx`,
  `image_size`, `mean`, and `std`.

- **`ERROR: could not open image`**
  The file is not a supported image (e.g. it's a `.heic`,
  `.svg`, `.gif` animation, or a corrupt file). Convert it to
  `.jpg` / `.png` first.

- **The label is wrong but confidence is very high.**
  The image is probably outside the model's training distribution
  (very small face, lots of background, unusual lighting). Try a
  tighter crop. In Phase 7 the OpenCV detector will handle the
  cropping automatically.

- **`could not import MaskCNN from src/model.py`**
  Run the script from the **project root**:
  `python src/predict_image.py --image ...`
  not from inside `src/`.

---

## 17. What's next

Phase 7 wraps everything together: an OpenCV-based webcam app that
detects faces in real time, crops each one, sends it through this
exact same classifier, and draws bounding boxes with predicted
labels and confidence scores.
