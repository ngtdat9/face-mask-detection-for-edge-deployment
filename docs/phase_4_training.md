# Phase 4 — Training the Mask Classifier

In Phase 3 we built the `MaskCNN` model. In Phase 4 we *teach* that
model to tell mask from no-mask using the dataset prepared in Phase 2.

The script lives in `src/train.py`. This document explains the *ideas*
behind every step before showing how to run it.

---

## 1. What is "training"?

**Training** is the process of adjusting the weights of a neural
network so that its predictions get closer to the correct answers.

It works in a loop:

1. Show the model a batch of labeled images.
2. Ask the model to predict each label.
3. Compare predictions to the ground-truth labels with a **loss
   function**. The loss is a single number that tells us how wrong
   the model was.
4. Compute how each weight contributed to that loss
   (**backpropagation**).
5. Nudge each weight a little in the direction that would reduce the
   loss (**optimizer step**).

Repeat for many batches and many epochs until the loss is low and the
model predicts well.

---

## 2. What is "validation"?

**Validation** is checking the model on data it has **never trained
on**, after each training epoch. We do *not* update the weights during
validation; we only measure performance.

Validation gives us an honest signal of how well the model is
generalizing. If the training loss keeps going down but the validation
loss starts going up, the model is overfitting (memorizing) and we
should stop training or change something.

---

## 3. Why split data into train and validation sets?

A model that has seen an image during training can almost always
predict it correctly — that does not prove it learned anything useful.
Real generalization is measured on **unseen** data.

That is why Phase 2 produced a separate `data/val/` folder. During
training:

- The **train set** updates the weights.
- The **val set** only measures performance.

The two sets must not overlap, otherwise the validation score is
misleading.

---

## 4. What does the "data wrapper stage" mean?

The data wrapper stage is the part of the script that turns folders of
image files into Python objects that a PyTorch training loop can
consume.

There are two layers:

| Layer       | Class                      | What it provides                                 |
| ----------- | -------------------------- | ------------------------------------------------ |
| **Dataset** | `ImageFolder`              | A way to ask "give me image #i and its label."   |
| **Loader**  | `DataLoader`               | Mini-batches images, shuffles, runs in workers.  |

So `ImageFolder` knows about *single images*; `DataLoader` knows about
*batches*. Training loops always read from a `DataLoader`.

---

## 5. Why `ImageFolder`?

`torchvision.datasets.ImageFolder` is built for exactly the layout we
already have:

```
<root>/
  with_mask/
    img_001.jpg
    ...
  without_mask/
    img_001.jpg
    ...
```

It scans the subfolders, treats each one as a class, applies the
transforms we pass, and returns `(image_tensor, label_int)` pairs.
Because Phase 2 created `data/train` and `data/val` in this exact
shape, we don't need a custom `Dataset` class — `ImageFolder` is
enough.

It also exposes `class_to_idx`, e.g. `{"with_mask": 0, "without_mask": 1}`,
which we later save into the checkpoint.

---

## 6. Why `DataLoader`?

`torch.utils.data.DataLoader` wraps a dataset and gives us all the
plumbing a training loop needs:

- **Batching**: groups individual images into mini-batches of size
  `batch_size`.
- **Shuffling**: `shuffle=True` for the train loader so each epoch
  sees the data in a different order. The val loader uses
  `shuffle=False` because we don't update weights from it.
- **Parallel loading** (`num_workers`): spawns helper processes that
  load and decode images while the GPU/CPU is busy with the previous
  batch.

In short: `ImageFolder` defines *what* the data is; `DataLoader`
defines *how* it is fed to the model.

---

## 7. Why `num_workers=0` on macOS?

On macOS, multiprocessing in PyTorch sometimes fights with the way
Python `multiprocessing` is initialized, especially under MPS (Apple
Silicon GPU) or when scripts are launched from an IDE. The simplest,
most reliable default is **0 workers**, meaning the main process
itself loads the data.

It's a little slower than `num_workers=4`, but:

- It works **everywhere** without surprises.
- For our dataset size (~5k train images, 128×128) the bottleneck is
  the GPU/MPS, not the data loader.

If you are on Linux or Windows and want to experiment, you can pass
`--num_workers 4`.

---

## 8. What is "batch size"?

**Batch size** is the number of images the model sees before each
weight update.

- Smaller batches: noisier updates, slower training, but uses less
  memory.
- Larger batches: smoother updates, faster training, but uses more
  memory.

`32` is a safe, common default and what we use here.

---

## 9. What is "learning rate"?

The **learning rate** controls how big a step the optimizer takes when
updating each weight.

- Too high: the optimizer overshoots and the loss bounces around or
  diverges.
- Too low: training is slow and may get stuck.

`0.001` is a standard starting value for the Adam optimizer.

---

## 10. What is an "epoch"?

One **epoch** is one full pass over the entire training set. After
one epoch the model has seen every training image exactly once.

We typically train for many epochs because each pass adjusts the
weights by a small amount; you need many small steps to converge.

---

## 11. What is "data augmentation"?

**Data augmentation** is the practice of applying small random
transformations to training images on the fly, so the model sees a
slightly different version of each image every epoch. It does **not**
change the underlying file on disk.

Augmentation:

- Multiplies the *effective* size of the dataset.
- Forces the model to learn features that are robust to small
  changes.
- Reduces overfitting.

We apply augmentation **only** to the training set, never to the
validation set, so val metrics remain comparable.

---

## 12. Why these specific augmentations?

| Transform                                 | What it does                                                                 | Why it helps for masks                                                                          |
| ----------------------------------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `RandomHorizontalFlip()`                  | Mirrors the image left-right with 50 % probability.                          | Faces are roughly symmetric, and people can be facing either way.                                |
| `RandomRotation(degrees=10)`              | Rotates the image by up to 10° in either direction.                          | A real face crop is rarely perfectly upright; webcams shake, heads tilt.                         |
| `ColorJitter(brightness=0.2, contrast=0.2)` | Randomly tweaks brightness/contrast by ±20 %.                              | Different lighting conditions (sunlight, indoor lamps, dark rooms) should not confuse the model.  |

We deliberately stay light on augmentation: aggressive transforms
(huge rotations, vertical flips, big color shifts) would create
unrealistic faces and hurt accuracy.

---

## 13. Why resize images to 128×128?

- The CNN in `model.py` is designed for a fixed input shape of
  `(3, 128, 128)`. Every image must be resized to that.
- 128×128 is small enough to train quickly on CPU/MPS but large
  enough to preserve mask details (color, edges, fabric texture).
- A consistent size lets us **batch** images together: every image
  in a batch must share the same shape.

The same `Resize((128, 128))` is applied at training, validation,
evaluation, and inference time so the model never sees an unexpected
input size.

---

## 14. Why normalize images?

`ToTensor()` converts pixel values from `[0, 255]` to `[0.0, 1.0]`.
`Normalize(mean, std)` then re-centers each channel:

```
out = (in - mean) / std
```

We use the standard ImageNet stats:

```
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Why normalize?

- Inputs end up centered around 0 with similar variance, which makes
  optimization more stable and faster.
- The same normalization must be applied at train, val, and inference
  time, otherwise the model sees a different distribution than what
  it learned on. That's why we save the `mean` / `std` into the
  checkpoint and reuse them in `evaluate.py` / `predict_image.py` /
  `live_camera.py`.

---

## 15. What does `CrossEntropyLoss` do?

`nn.CrossEntropyLoss` is the standard loss for multi-class
classification. Internally it does two things in a numerically stable
way:

1. Applies `log_softmax` to the model's logits to obtain log
   probabilities.
2. Computes the negative log-likelihood of the correct class.

The lower the loss, the more confident and correct the model is. Loss
is `0` when the model assigns probability `1` to the correct class.

---

## 16. Why does the model return logits?

The model produces **logits** — raw, unbounded numbers, one per
class. There is no softmax inside `MaskCNN`.

Two reasons:

1. `CrossEntropyLoss` already applies softmax internally. Adding
   another softmax in the model would compute it twice and break
   training.
2. Logits are easier to work with numerically. Softmax can be applied
   later, only when we want to display a probability.

---

## 17. Why is softmax not used during training?

Same reason: `CrossEntropyLoss` expects **logits**, not
probabilities. We only apply `softmax` at inference time, in
`predict_image.py` and `live_camera.py`, when we want to show a
probability such as `with_mask 0.97` next to a bounding box.

---

## 18. What does the Adam optimizer do?

**Adam** is the algorithm that actually updates the weights using the
gradients computed by backpropagation.

Why we use it:

- It keeps a per-parameter, **adaptive** learning rate.
- It works well out of the box on most problems with `lr=0.001`.
- It is more forgiving than plain SGD if the learning rate is not
  perfectly tuned.

`optim.Adam(model.parameters(), lr=args.lr)` plugs Adam into our
training loop.

---

## 19. What is "training loss"?

The average loss over the **training set** during the current epoch.
Training loss tells us how well the model fits the data it can update
its weights from. It should generally go down over time.

---

## 20. What is "validation loss"?

The average loss over the **validation set**, computed at the end of
each epoch with weights frozen. Validation loss tells us how well the
model generalizes to unseen images.

If validation loss starts going **up** while training loss keeps
going **down**, the model is overfitting.

---

## 21. What is "training accuracy"?

The fraction of training images the model classified correctly during
the current epoch:

```
training accuracy = correct predictions on train / total train images
```

It's an easier-to-read companion to training loss.

---

## 22. What is "validation accuracy"?

Same idea, computed on the **validation** set after the epoch's
training is done:

```
validation accuracy = correct predictions on val / total val images
```

This is the number we ultimately care about, because it estimates
real-world performance on unseen images.

---

## 23. Why save the best model based on validation accuracy?

Training accuracy is biased: the model has just seen those images.
Validation accuracy is a fair test, because the model never updated
its weights on those images.

So we keep the **single best checkpoint** — the one with the highest
validation accuracy across all epochs. Later epochs may overfit and
score worse on val, even if their train accuracy keeps improving;
saving by val accuracy automatically protects us from that.

---

## 24. Why save `class_to_idx` with the model checkpoint?

`ImageFolder` assigns label indices alphabetically:
`{"with_mask": 0, "without_mask": 1}`. If a future evaluation or
inference script ever uses a different mapping (for example because
class folders are renamed), it would silently swap predictions.

By saving `class_to_idx` inside the checkpoint, every downstream
script (`evaluate.py`, `predict_image.py`, `live_camera.py`) can
read it back and be **100 % sure** which logit corresponds to which
class.

We save the same idea for `image_size`, `mean`, and `std`: a
checkpoint is self-contained and can be deployed without any
external configuration.

---

## 25. What does `loss_plot.png` show?

Two curves over epochs (1 → N):

- **train loss** — average loss on the train set per epoch.
- **val loss** — average loss on the val set per epoch.

How to read it:

- Both curves going down ⇒ healthy training.
- Train down, val up ⇒ overfitting; consider stopping earlier or
  adding more augmentation/regularization.
- Both flat from the start ⇒ learning rate may be too high/low or
  data preprocessing is broken.

---

## 26. What does `accuracy_plot.png` show?

Two curves over epochs:

- **train accuracy** — fraction correct on the train set per epoch.
- **val accuracy** — fraction correct on the val set per epoch.

Generally both should rise. The peak of the val accuracy curve is
where the best checkpoint was saved.

---

## 27. How to run `train.py`

From the **project root**:

```bash
python src/train.py --epochs 30 --batch_size 32 --lr 0.001 --num_workers 0
```

All flags have defaults, so this also works:

```bash
python src/train.py
```

Common variations:

```bash
# Quick smoke test on 1 epoch
python src/train.py --epochs 1

# Bigger batch (if you have a GPU and enough RAM)
python src/train.py --batch_size 64

# Custom paths
python src/train.py \
    --train_dir data/train \
    --val_dir data/val \
    --model_path models/mask_classifier.pth \
    --results_dir results
```

What you should see while training:

```
Using device: mps
Found classes: {'with_mask': 0, 'without_mask': 1}
Train images : 5286
Val images   : 1132

Starting training for 30 epochs...

Epoch   1/30 | train_loss=0.4321 val_loss=0.2510 | train_acc=0.8123 val_acc=0.9123  *new best*
Epoch   2/30 | train_loss=0.2107 val_loss=0.1623 | train_acc=0.9234 val_acc=0.9456  *new best*
...
Training finished.
Best epoch: 24  best val_accuracy: 0.9821
Best model saved to: .../models/mask_classifier.pth
Loss plot saved to    : .../results/loss_plot.png
Accuracy plot saved to: .../results/accuracy_plot.png
```

(Exact numbers depend on your hardware and randomness.)

---

## 28. What's next

In Phase 5 we will:

- Load the saved checkpoint with `class_to_idx`, `image_size`, and
  normalization stats.
- Evaluate on `data/test/` to get a final, honest accuracy and a
  confusion matrix (`evaluate.py`).
- Wire the trained classifier into `predict_image.py` (single image)
  and `live_camera.py` (real-time webcam) — combining OpenCV face
  detection with `MaskCNN` mask classification.
