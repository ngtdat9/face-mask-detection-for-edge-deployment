# Phase 3 — Model Architecture (`MaskCNN`)

In Phase 2 we produced a clean, split dataset under `data/`. In Phase 3
we design the **brain** of our system: a Convolutional Neural Network
(CNN) that looks at a 128×128 cropped face and decides whether it is
`with_mask` or `without_mask`.

The implementation lives in `src/model.py`. This document explains the
*ideas* behind every layer in plain language, suitable for a school
project.

---

## 1. What is a CNN?

A **Convolutional Neural Network (CNN)** is a kind of neural network
that is specialized for images. Instead of treating an image as a flat
list of numbers, a CNN slides small filters over the image to look for
visual patterns — first simple ones (edges, colors), then more complex
ones (textures, shapes), and finally high-level concepts ("a piece of
fabric covering the lower half of a face").

Two ideas make a CNN different from a regular fully connected network:

1. **Local receptive fields.** Each filter looks at a small patch of
   the image at a time, not the whole image. That matches how visual
   information is structured.
2. **Weight sharing.** The *same* filter is applied at every location
   in the image. So a "vertical edge detector" learned in one corner
   automatically works in every other corner too.

The result: far fewer parameters than a fully connected network, and a
strong inductive bias that suits images.

---

## 2. Why CNNs are useful for image classification

For our problem, a CNN is a natural fit because:

- A mask is a **localized visual pattern** (fabric over nose and
  mouth). Convolutional filters are very good at detecting localized
  patterns.
- The pattern can appear at slightly different positions and scales
  inside the cropped face. CNNs are **translation-equivariant**: a
  feature in the top-left of the image is detected the same way as the
  same feature in the bottom-right.
- Stacking convolutional layers lets the network build up **hierarchy**:
  edges → textures → object parts → object. By the last block the
  network can reason about "mask vs no-mask" using rich features
  instead of raw pixels.

In short, CNNs match the structure of images, so they need much less
data and compute than a fully connected network would.

---

## 3. CNN feature extractor vs FCNN classifier

Modern image classifiers, including ours, are split into two clearly
separated parts:

| Part                        | Job                                                                  | Where in our code |
| --------------------------- | -------------------------------------------------------------------- | ----------------- |
| CNN **feature extractor**   | Turn raw pixels into a compact, *meaningful* numerical description.  | `self.features`   |
| FCNN **classifier head**    | Take that description and decide which class it belongs to.          | `self.classifier` |

Think of it as two specialists working together:

- The **feature extractor** is like an art student who carefully
  describes what they see in the picture: "I see a vertical edge here,
  a smooth fabric texture there, two eyes above a covered area...".
- The **classifier** is like a friend who hears the description and
  says: "Sounds like the person is wearing a mask."

The feature extractor is heavy (lots of convolutions, batch norms,
poolings) but generic. The classifier is small but is the part that
actually picks the answer.

---

## 4. Where is the FCNN part in this architecture?

In our code the FCNN classifier is everything inside `self.classifier`:

```
Flatten          # turn 256 x 8 x 8 into a 1D vector of length 16384
Linear(16384,256)  # fully connected layer
ReLU
Dropout(p=0.5)
Linear(256, 2)   # final fully connected layer -> 2 logits
```

Everything before that — the four `Conv -> BN -> ReLU -> MaxPool`
blocks — is the CNN feature extractor (`self.features`).

That separation is not just decoration. It mirrors what the network
*does* and makes the model easier to understand, debug, and later
modify (e.g., swapping the classifier head while keeping the features).

---

## 5. What does `Conv2d` do?

`nn.Conv2d` is the **convolutional layer**. It slides small filters
(also called kernels) over the input image and computes a dot product
at every position. Each filter learns to detect one specific kind of
visual pattern.

Key parameters in our model:

- `in_channels`: how many channels are in the input. Block 1 starts
  with `3` (RGB), the next blocks use the previous block's output
  channels.
- `out_channels`: how many filters this layer learns. More filters =
  the layer can detect more different patterns. We grow this from
  `32 → 64 → 128 → 256` as we go deeper, because deeper layers need to
  represent more abstract, more varied features.
- `kernel_size=3`: each filter is 3×3.
- `padding=1`: we add a 1-pixel border so the convolution itself does
  **not** change the spatial size. That way the only thing shrinking
  the image is the `MaxPool2d` at the end of each block — which makes
  the size flow easy to predict.

Output of `Conv2d` is a stack of "feature maps", one per filter.

---

## 6. What does `BatchNorm2d` do?

`nn.BatchNorm2d` is **Batch Normalization** for 2D feature maps. After
each convolution we re-center and re-scale the activations using the
mean and standard deviation computed over the current mini-batch.

Why bother?

- It keeps the numbers flowing through the network in a sensible
  range, instead of drifting toward very large or very small values.
- Networks with BatchNorm typically train **faster** and tolerate
  **higher learning rates**.
- It acts as a mild regularizer because the batch statistics add a
  small amount of noise during training.

We add it right after the `Conv2d` and before the activation, which is
the standard recipe.

---

## 7. What does `ReLU` do?

`nn.ReLU` is the **activation function**. It takes a number and
returns:

```
ReLU(x) = max(0, x)
```

So negative values become 0; positive values pass through unchanged.

Why we need it:

- Without an activation, stacking many `Linear`/`Conv` layers would
  collapse mathematically into a single linear function. A non-linear
  activation is what lets a deep network model **complex** patterns.
- ReLU is cheap, fast, and works very well in practice.

We use `inplace=True` to save a bit of memory; it modifies the tensor
in place instead of allocating a new one.

---

## 8. What does `MaxPool2d` do?

`nn.MaxPool2d(kernel_size=2, stride=2)` looks at every 2×2 block of the
feature map and keeps only the maximum value. This:

- **Halves the spatial size** at each block (`128 → 64 → 32 → 16 → 8`).
- Makes the network slightly **translation-invariant**: small shifts of
  the input do not change the output much.
- Forces the network to keep the **strongest** signals and drop the
  weakest, which is a form of "summarization".

That's why each block ends with a max-pool: the conv detects features,
the BN+ReLU shape them, and the max-pool compresses them before the
next block.

---

## 9. What does `Flatten` do?

After the final convolution block, each image has become a tensor of
shape `(256, 8, 8)`. A fully connected layer expects a **1D vector**
per sample, so we use `nn.Flatten` to reshape it:

```
(batch, 256, 8, 8)  -->  (batch, 256 * 8 * 8) = (batch, 16384)
```

That single 16384-long vector is the "summary" of the image that the
classifier head will work with.

---

## 10. What do `Linear` layers do?

`nn.Linear(in_features, out_features)` is a **fully connected layer**.
Mathematically it computes `y = x @ W.T + b`: every output neuron is a
weighted sum of every input neuron.

In our classifier head we use two of them:

- `Linear(16384, 256)`: compresses the big 16384-long feature vector
  into a smaller 256-long vector. The network learns *what to keep*.
- `Linear(256, 2)`: maps that compact vector to **2 numbers**, one per
  class.

Linear layers are how the network finally combines all the visual
evidence into a class decision.

---

## 11. What does `Dropout` do?

`nn.Dropout(p=0.5)` randomly sets half of the inputs to zero **during
training**. It is a **regularization** technique that fights
overfitting.

Why it helps:

- The network cannot rely on any single neuron, because that neuron
  might be dropped at any step. So it learns redundant, more robust
  features.
- The remaining activations are automatically scaled by `1/(1-p)` so
  the expected value of the output stays the same.

Dropout is **automatically disabled** when the model is in eval mode
(`model.eval()`), so it does not affect prediction at inference time.

---

## 12. Why does the final layer have 2 outputs?

We have **2 classes**: `with_mask` and `without_mask`. So the model
produces 2 numbers per image — one "score" per class. The class with
the higher score is the model's prediction.

If we ever extended the project to, say, also detect "mask worn
incorrectly", we would change `num_classes` to `3` and the final
`Linear(256, num_classes)` would produce 3 numbers. Nothing else in
the architecture has to change.

---

## 13. Why does the model return logits instead of probabilities?

The 2 numbers the model returns are called **logits**. They can be
any real number (negative, zero, positive) — they are *not*
probabilities. This is intentional, for two reasons:

1. **Training is stable in logit space.** PyTorch's
   `nn.CrossEntropyLoss` expects logits and applies `log_softmax`
   internally with a numerically stable implementation. If we applied
   softmax inside the model and then computed cross-entropy, we would
   take `log` of a very small number and lose precision.
2. **It separates concerns.** The model's job is to produce a score
   for each class. Whether we want to turn that into probabilities
   (using `softmax`) or just take the highest score (using `argmax`)
   is up to the *caller*.

---

## 14. Why softmax is used during prediction but not inside the model?

`softmax` turns a vector of logits into a vector of probabilities that
sum to 1:

```
probs = exp(logits) / sum(exp(logits))
```

When we display a prediction in the live demo, we want to show
something like `with_mask 0.97`. That number is a probability, so we
*do* need softmax — but only **at inference time**, on the model's
output.

We do **not** put softmax inside the `forward` method because:

- During training we use `nn.CrossEntropyLoss`, which already includes
  softmax internally. Adding another softmax would compute it twice
  and break training.
- Keeping softmax outside the model lets us use the same model for
  training (where we want logits) and for the live demo (where we
  apply softmax just before drawing on the frame).

So the rule is: **inside the model = logits; outside the model =
softmax for display, argmax for the predicted class**.

---

## 15. Why does this model only classify mask/no-mask?

This project uses a **two-stage pipeline**:

```
  Webcam frame
      |
      v
  +--------------------+
  | OpenCV face detector|  <-- finds *where* the faces are
  +--------------------+
      |
      v   (cropped 128x128 face)
  +--------------------+
  |     MaskCNN         |  <-- decides *what* the face is wearing
  +--------------------+
      |
      v   (logits -> probabilities -> label)
  Drawn bounding box + label
```

`MaskCNN` is **not** an object detector. It does not produce bounding
boxes, it does not localize anything, and it does not handle full
images with several faces. It only takes one already-cropped face and
classifies it.

Why split the pipeline this way?

- Face detection is a **solved problem**. OpenCV ships ready-to-use
  detectors that are fast and accurate, so building our own would be
  reinventing the wheel.
- Classification on a tightly cropped face is a **much simpler**
  problem than detection on a full image. A small CNN can learn it
  with a modest dataset and modest training time — perfect for a
  school project.
- The two stages are **independent**: we can improve the detector
  without touching the classifier and vice versa.

That's why this network has no anchors, no bounding-box regression,
and no objectness score — just `Conv` blocks for features and a small
FCNN head for the final yes/no decision.

---

## 16. Quick reference: shape flow

```
Input              :  (B, 3,   128, 128)

Conv Block 1       :  (B, 32,   64,  64)
Conv Block 2       :  (B, 64,   32,  32)
Conv Block 3       :  (B, 128,  16,  16)
Conv Block 4       :  (B, 256,   8,   8)

Flatten            :  (B, 16384)
Linear(16384, 256) :  (B, 256)
ReLU + Dropout(0.5):  (B, 256)
Linear(256, 2)     :  (B, 2)        <-- final logits
```

Where `B` is the batch size. For a single image (`B = 1`) the final
output tensor has shape `torch.Size([1, 2])`, which is exactly what
the test block in `src/model.py` checks at the end.

---

## 17. How to verify the model works

From the project root:

```bash
python src/model.py
```

You should see:

```
Input shape : (1, 3, 128, 128)
Output shape: (1, 2)
OK: output shape matches torch.Size([1, 2])
```

If you see that, every layer is wired correctly and we are ready for
Phase 4: writing the training loop.

---

## 18. What's next

In Phase 4 we will:

- Build PyTorch `Dataset` / `DataLoader` objects on top of our
  `data/train` and `data/val` folders (using `ImageFolder` from Phase 2).
- Train `MaskCNN` with `nn.CrossEntropyLoss` and an `Adam` optimizer.
- Track training/validation loss and accuracy per epoch.
- Save the best model to `models/` and the training curves to
  `results/`.
