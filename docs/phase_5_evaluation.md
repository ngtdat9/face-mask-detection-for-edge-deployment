# Phase 5 — Evaluating the Mask Classifier

In Phase 4 we trained `MaskCNN` and saved the best checkpoint based on
validation accuracy. In Phase 5 we **measure how well that saved model
actually works on data it has never seen** — the test set we created
in Phase 2 (`data/test/`).

The script lives in `src/evaluate.py`. This document explains every
metric and plot it produces.

---

## 1. Why evaluate on the test set?

While training, we used:

- the **train** set to update weights, and
- the **val** set to choose the best checkpoint and tune
  hyperparameters.

The model has therefore been "exposed" to both sets — directly (train)
or indirectly (val, because we picked the best model based on it).

The **test set** is the one set we never looked at during development.
Running the model on it gives us a single, *honest* number: what the
model would do on completely new data, like images it sees during the
live demo.

> Rule of thumb: train tells you whether the model **can learn**, val
> tells you when it **stops improving**, test tells you how well it
> will **work in the real world**.

Because we trust this number, we open the test set **only once**, at
the end. If we tweak hyperparameters based on test results, we lose
that honesty and effectively turn the test set into another val set.

---

## 2. What does *accuracy* mean?

```
accuracy = number of correct predictions / total number of predictions
```

It answers the question: *"Out of all images, what fraction did the
model classify correctly?"*

Accuracy is intuitive but it can be **misleading on imbalanced
datasets**. If 95 % of images are `without_mask`, a model that always
predicts `without_mask` would have 95 % accuracy without learning
anything useful. Our dataset is roughly balanced (≈ 49 % vs 51 %), so
accuracy is meaningful — but we still report precision / recall / F1
as a safety net.

---

## 3. What does *precision* mean?

For one class (say, `with_mask`):

```
precision = true positives / (true positives + false positives)
```

In words: *"When the model predicts `with_mask`, how often is it
right?"*

- High precision ⇒ the model rarely cries wolf.
- Low precision ⇒ the model often labels things `with_mask` that are
  not.

Precision is the metric you care about when **a false positive is
expensive** (e.g. you don't want to falsely tell someone they are
wearing a mask when they are not).

---

## 4. What does *recall* mean?

For one class:

```
recall = true positives / (true positives + false negatives)
```

In words: *"Out of all the actual `with_mask` images, how many did
the model catch?"*

- High recall ⇒ the model finds almost every positive.
- Low recall ⇒ the model misses a lot of them.

Recall is the metric you care about when **a missed positive is
expensive** (e.g. in a hospital you don't want to miss anyone who is
unmasked).

> Precision = "of the alarms I rang, how many were real?"
> Recall    = "of the real cases, how many alarms did I ring?"

---

## 5. What does *F1-score* mean?

F1 is the harmonic mean of precision and recall:

```
F1 = 2 * (precision * recall) / (precision + recall)
```

It's a single number that is high **only when both** precision and
recall are high. If either one is low, F1 drops sharply.

We typically report **macro F1** for our two classes: compute F1 per
class, then average. That treats both classes equally regardless of
how many samples each has, which is what we want for a roughly
balanced two-class problem.

---

## 6. What is a confusion matrix?

A **confusion matrix** is a small table that shows how predictions
line up with the true labels. For 2 classes it looks like this:

```
                 Predicted
                 with_mask    without_mask
True with_mask   [   TP_w  ]  [   FN_w   ]
     without_mask[   FN_w' ]  [   TP_w'  ]
```

Where:

- The **diagonal** (top-left, bottom-right) holds the **correct**
  predictions.
- The **off-diagonal** cells hold the **mistakes**.

Every metric above (accuracy, precision, recall, F1) can be computed
directly from these four numbers.

---

## 7. How to read `results/confusion_matrix.png`

The image shows the same table as a heatmap, with raw counts in each
cell. To read it, look at:

| Cell                      | Meaning                                                     |
| ------------------------- | ----------------------------------------------------------- |
| top-left (`with_mask` → `with_mask`)         | The model correctly recognized a masked face. |
| top-right (`with_mask` → `without_mask`)     | The model **missed** a mask (false negative for `with_mask`). |
| bottom-left (`without_mask` → `with_mask`)   | The model **falsely** said someone was wearing a mask. |
| bottom-right (`without_mask` → `without_mask`) | The model correctly recognized an unmasked face. |

A good model has **dark diagonal** (lots of correct predictions) and
**light off-diagonal** (few mistakes), exactly what you should see in
the saved heatmap.

For our run on the held-out test set, the matrix shows e.g.:

```
                 Predicted
                 with_mask    without_mask
True with_mask        545           15
     without_mask       8          567
```

So out of 1135 test images:

- 1112 were classified correctly (545 + 567).
- 15 masked faces were misread as unmasked.
- 8 unmasked faces were misread as masked.

---

## 8. What does `results/classification_report.png` show?

This image is a small table generated from sklearn's
`classification_report`. It lists, **per class**, the four numbers we
care about:

| Column        | What it tells you                                             |
| ------------- | ------------------------------------------------------------- |
| `precision`   | Of predictions for this class, how many were correct.         |
| `recall`      | Of true examples of this class, how many were caught.         |
| `f1-score`    | Harmonic mean of the two — single quality score per class.    |
| `support`     | How many real examples of this class are in the test set.     |

The bottom three rows summarize across classes:

- **`accuracy`** — overall accuracy on the whole test set
  (only filled in the `f1-score` column to keep the table compact).
- **`macro avg`** — mean of per-class precision/recall/F1, treating
  both classes equally.
- **`weighted avg`** — mean weighted by `support`. Differs from
  macro average only when classes are imbalanced.

`support` doubles as a sanity check that the test set has the right
number of images per class.

---

## 9. How to run `evaluate.py`

From the **project root**:

```bash
python src/evaluate.py
```

All flags have sensible defaults; common ones if you need them:

```bash
python src/evaluate.py \
    --test_dir data/test \
    --model_path models/mask_classifier.pth \
    --results_dir results \
    --batch_size 32 \
    --num_workers 0
```

What the script does, in order:

1. Picks the best device (`mps` > `cuda` > `cpu`) and prints it.
2. Loads `models/mask_classifier.pth`. The checkpoint includes
   `class_to_idx`, `image_size`, `mean`, and `std` from training, so
   evaluation uses **exactly the same preprocessing** the model was
   trained on.
3. Wraps `data/test/` with `ImageFolder` + the validation transform
   (Resize → ToTensor → Normalize).
4. Runs the model on every test image and collects predictions.
5. Computes accuracy + macro precision / recall / F1, and prints the
   per-class report.
6. Saves `results/confusion_matrix.png` and
   `results/classification_report.png`.

Expected console output looks like:

```
Using device: mps
Loading checkpoint: .../models/mask_classifier.pth
Classes: ['with_mask', 'without_mask']
Checkpoint best val accuracy: 0.9850
Test images: 1135

==================================================
Test set results
==================================================
Accuracy : 0.9797
Precision: 0.9799  (macro average)
Recall   : 0.9797  (macro average)
F1-score : 0.9797  (macro average)

Per-class report:
              precision    recall  f1-score   support

   with_mask     0.9855    0.9732    0.9793       560
without_mask     0.9742    0.9861    0.9801       575

    accuracy                         0.9797      1135
   macro avg     0.9799    0.9797    0.9797      1135
weighted avg     0.9798    0.9797    0.9797      1135

Confusion matrix saved to    : .../results/confusion_matrix.png
Classification report saved to: .../results/classification_report.png
```

(Your numbers may differ slightly depending on training randomness.)

---

## 10. Quick troubleshooting

- **`trained model not found`** — you have not trained yet. Run
  `python src/train.py` first.
- **`test folder not found`** — you have not built the splits yet.
  Run `python src/prepare_data.py` first.
- **`test class_to_idx does not match the checkpoint`** — the test
  folder has different class subfolders than the ones used during
  training. Re-run `prepare_data.py` to rebuild `data/`.

---

## 11. What's next

In Phase 6 we connect the dots into a real-time application:

- A single-image script (`predict_image.py`) that takes a photo path
  and prints the predicted class + probability.
- A live webcam script (`live_camera.py`) that uses **OpenCV** to
  detect faces in each frame, crops them, runs the trained `MaskCNN`
  on each crop, and draws bounding boxes with labels and
  probabilities — finally giving us the full two-stage face mask
  detection system.
