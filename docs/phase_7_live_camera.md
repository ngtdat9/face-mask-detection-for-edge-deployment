# Phase 7 — Live Camera Face Mask Detection

In Phase 6 we tested the trained classifier on a single image. In
Phase 7 we put everything together into a **real-time application**:
the webcam streams frames, OpenCV finds every face in each frame, and
our trained `MaskCNN` decides whether each face is wearing a mask.

The script lives in `src/live_camera.py`. This document explains the
ideas behind it, how to use it, and why we now default to a **DNN**
face detector instead of the classical Haar Cascade.

---

## 1. The pipeline at a glance

```
Webcam frame (BGR)
        |
        v
  Face detector  (YuNet by default, Haar Cascade as a fallback)
        |
        v   list of (x, y, w, h) face boxes
  For each face:
      crop -> BGR->RGB -> resize 128x128 -> normalize
        |
        v   batch tensor (N, 3, 128, 128)
  MaskCNN classifier
        |
        v   logits -> softmax -> (label, confidence)
  Draw bounding box + label on the frame
        |
        v
  cv2.imshow -> screen
```

Two things are worth highlighting:

- **OpenCV** owns face *detection* (where the faces are).
- **PyTorch / MaskCNN** owns mask *classification* (what each face is
  wearing).

They are completely independent and can be improved separately —
which is exactly what we do in this phase: we keep the same
classifier and only swap the detector.

---

## 2. Why is the face detector so important?

The classifier is only as useful as the detector. Concretely:

> **The classifier only runs *after* a face is detected.**
> If the face detector cannot find a face, no crop is produced, no
> forward pass happens, and **no label is drawn** — the camera looks
> as if the model is doing nothing.

This is a real, very common failure mode in mask-detection projects.
You'll know you're hitting it if:

- Standing in front of the camera with **no mask** works fine and
  draws `without_mask`.
- Standing in front of the camera **with** a mask draws nothing — no
  red box, no green box, just the raw frame.

That isn't the classifier failing; it's the *detector* failing
silently before the classifier ever gets a chance.

---

## 3. Why Haar Cascade may fail when the lower face is covered by a mask

Haar Cascades (the classical OpenCV face detector) work by checking
for a fixed set of **intensity patterns**:

- "Are the eyes darker than the cheeks below them?"
- "Is there a dark vertical band where the nose should be?"
- "Are the corners of the mouth lighter than the upper lip?"

These tests assume the **whole face is visible**. They were trained
in 2001, on photographs of people who are not wearing masks. When the
mouth, chin, and lower nose are covered by a mask:

- The "mouth lighter than upper lip" check fails — there's a flat
  fabric region instead.
- The lower face has different intensity statistics from anything in
  the training data.
- Several of the cascade's stages reject the candidate before it
  ever becomes a returned face.

Result: the detector silently returns *zero* faces. The classifier
never runs. The camera looks broken.

A **printed photo of a masked face** on your phone screen sometimes
still works because the pixel statistics around the eyes are sharp
and high-contrast — enough for some early stages of the cascade to
pass — but live, soft-lit, slightly-shadowed mask fabric in real life
trips it up.

---

## 4. Why DNN face detection is better for this case

A "DNN face detector" is a face detector built as a **deep neural
network** — same family of model as the classifier, but tasked with
finding bounding boxes. The version we use is **YuNet**, which:

- Is bundled with modern OpenCV via `cv2.FaceDetectorYN_create`.
- Is small (~230 KB ONNX) and fast (CPU is fine).
- Was trained on a dataset that includes **occluded faces**: masks,
  glasses, hands, hair, partial views.

Because it was *trained* on data where the lower face is sometimes
covered, it learned to recognize a face from features that are still
visible: the eyes, eyebrows, nose bridge, hairline, and overall face
shape. That makes it robust to exactly the case where Haar fails.

**Concrete evidence from this project**

We re-ran both detectors on a small slice of `data/test/`:

| Image                | YuNet faces | Haar faces |
|----------------------|-------------|------------|
| `with_mask` (1)      | 1           | **0**      |
| `with_mask` (2)      | 1           | **0**      |
| `with_mask` (3)      | 1           | 2          |
| `without_mask` (1)   | 1           | 1          |
| `without_mask` (2)   | 1           | **0**      |
| `without_mask` (3)   | 1           | 1          |

YuNet finds the face in **all six** images, including every masked
one. Haar misses **three**, including two of the three masked images.
That is exactly the bug we hit with the live camera, reproduced on
the test set.

---

## 5. Detector options in this script

`live_camera.py` now supports two detectors. You pick one with the
`--detector` flag:

| Flag              | Detector       | Best for                                           |
|-------------------|----------------|----------------------------------------------------|
| `--detector dnn`  | YuNet (ONNX)   | The default. Robust to masked / occluded faces.    |
| `--detector haar` | Haar Cascade   | Fallback when a network download isn't possible.   |

When you choose `--detector dnn` for the first time, the script
automatically downloads the YuNet ONNX file from OpenCV's official
zoo (~230 KB) into `models/face_detection_yunet_2023mar.onnx`. After
that, no internet is needed.

When you choose `--detector haar`, the script uses OpenCV's bundled
`haarcascade_frontalface_default.xml` (no download required) with
slightly more sensitive parameters than the textbook defaults:

```
scaleFactor   = 1.05
minNeighbors  = 3
minSize       = (40, 40)
```

These tweaks help Haar catch borderline faces, but they cannot fix
its fundamental difficulty with masked lower faces — that's why
`dnn` is the recommended default.

---

## 6. How the classifier predicts mask / no-mask

Once a face is detected, the rest of the pipeline is identical to
Phase 6:

1. Crop the face region from the **original BGR frame**.
2. Convert the crop from **BGR to RGB** (OpenCV uses BGR; PyTorch /
   PIL expect RGB).
3. **Resize** to 128×128.
4. **`ToTensor`** — float tensor in `[0, 1]`.
5. **`Normalize`** with the same `mean` and `std` saved in the
   checkpoint.

All face tensors from the same frame are stacked into one batch
`(N, 3, 128, 128)` and run through the model in **one forward pass**.
The model returns logits, and `F.softmax` converts them into
probabilities. We pick the class with the highest probability and
show that probability as the **confidence**.

> Reminder: `softmax` is applied **outside** the model, only at
> inference. During training, `nn.CrossEntropyLoss` already includes
> `log_softmax` internally, so the model itself returns raw logits.

---

## 7. Why the boxes follow moving faces

There is no tracker — no IDs, no Kalman filter. Boxes "follow"
people simply because the **whole detect + classify + draw chain
re-runs on every frame** at 20–30 frames per second:

- Frame *t*: detect, classify, draw.
- Frame *t + 1*: detect again, classify again, draw again.
- ...

If you walk left, the next frame's detector will fire slightly to
the left, so the box appears to glide along.

This approach is robust to people entering or leaving the frame, and
avoids the complexity of multi-object tracking — exactly what we
want for a school-level real-time demo.

---

## 8. How multiple faces are handled

Both detectors return a **list** of bounding boxes. The script:

1. Crops every box into its own face tensor.
2. Stacks them into one batch of shape `(N, 3, 128, 128)`.
3. Runs **one** forward pass for all faces at once.
4. Iterates over the results to draw a colored box and label per face.

Boxes are colored:

- **Green** for `with_mask`.
- **Red** for `without_mask`.

If no faces are detected in a frame, no boxes are drawn — the script
just shows the raw camera image.

---

## 9. What does *confidence* mean?

The confidence is the probability the model gives to the class it
predicted:

```
probabilities = softmax(logits)              # e.g. [0.964, 0.036]
confidence    = max(probabilities)           # 0.964
predicted     = idx_to_class[argmax(probs)]  # "with_mask"
```

It is shown as a percentage (e.g. `96.4%`). High confidence means the
model is very sure; very low confidence (close to 50 % for two
classes) means the image is borderline. High confidence does **not**
guarantee correctness — it's just the model's own self-assessment.

---

## 10. Debug mode (`--debug`)

If something looks off — e.g. the camera shows your face but no box
appears — turn on debug mode:

```bash
python src/live_camera.py --camera 0 --detector dnn --debug
```

In debug mode the script:

- Prints the **number of faces detected per frame** in the terminal:
  `Frame 137: 1 face(s) detected`.
- Prints a **warning** when a frame contains zero faces, so you can
  see in real time when the detector is missing you.
- Draws a **thin grey rectangle** around every raw detector output,
  even before classification. So if you see a grey box but no green/
  red label, the detector worked and the classifier didn't (a bug
  somewhere in cropping/inference). If you see no grey box at all,
  the **detector** is the one not finding your face — switch to
  `--detector dnn` (if you were on `haar`), or move closer to the
  camera, improve lighting, etc.

This is the fastest way to tell *which* stage of the pipeline is at
fault.

---

## 11. Why the device is `mps` / `cuda` / `cpu` (in that order)

The script picks the fastest available backend automatically:

```python
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
```

- **`mps`** — Apple Silicon GPU acceleration (M1/M2/M3/M4 Macs).
- **`cuda`** — NVIDIA GPU acceleration.
- **`cpu`** — always works, just slower.

The selected device is printed once when the script starts so you
can confirm it. The face detector itself runs on CPU in OpenCV; only
the classifier benefits from GPU/MPS.

---

## 12. How to run `live_camera.py`

From the **project root**:

### Recommended: DNN (YuNet) detector

```bash
python src/live_camera.py --camera 0 --detector dnn
```

This is also the default, so you can drop the flag:

```bash
python src/live_camera.py
```

The first run will download the YuNet model (~230 KB) into `models/`.
Subsequent runs are offline-friendly.

### Fallback: Haar Cascade detector

```bash
python src/live_camera.py --camera 0 --detector haar
```

Use this only if the YuNet download fails (no internet) or you
specifically want to compare detectors. Be aware Haar will often
fail to detect masked faces.

### Debug mode (any detector)

```bash
python src/live_camera.py --camera 0 --detector dnn --debug
python src/live_camera.py --camera 0 --detector haar --debug
```

What you should see in either case:

1. The terminal prints `Using device: mps` (or cuda / cpu),
   `Loaded checkpoint:`, `Classes:`, `Face detector: YuNet (DNN)`
   or `Haar Cascade`, and finally
   `Camera opened. Press 'q' in the camera window to quit.`
2. A new window titled
   `Face Mask Detection (press q to quit)` opens with the live feed.
3. A green box (`with_mask`) or red box (`without_mask`) is drawn
   around each face, with a label like `with_mask 96.4%`.
4. Press **`q`** with the camera window focused to quit cleanly. The
   camera is released automatically (`finally` block).

> macOS only: the very first time you run this, your terminal/IDE
> needs camera permission. Open *System Settings → Privacy &
> Security → Camera* and grant access to whichever app you launched
> the script from.

---

## 13. How to change the camera index

If the wrong camera opens, just bump the index:

```bash
python src/live_camera.py --camera 1
python src/live_camera.py --camera 2
```

Quick tips:

- Index `0` is usually the built-in webcam.
- An external USB camera tends to be `1`.
- iPhone Continuity Camera often shows up as `1` or `2`.
- If a chosen index fails, the script prints a clear error
  suggesting another index and the macOS Privacy & Security setting
  to check.

---

## 14. Common problems and fixes

- **No box is drawn even though my face is in the frame.**
  This is the bug we just fixed. If you are on `--detector haar`,
  switch to `--detector dnn`. Use `--debug` to confirm whether the
  detector is the issue (no grey box ⇒ detector problem) or the
  classifier (grey box but no green/red label ⇒ pipeline problem).

- **`ERROR: could not download YuNet model`.**
  You're offline or behind a strict proxy. Either reconnect, manually
  download the file from
  `https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx`
  to `models/face_detection_yunet_2023mar.onnx`, or fall back to
  `--detector haar`.

- **`ERROR: your OpenCV does not support YuNet`.**
  Your `opencv-python` is older than 4.5.4. Run
  `pip install --upgrade opencv-python` and try again, or use
  `--detector haar`.

- **`ERROR: could not open camera index 0`.**
  Either no camera is available at that index, or your terminal/IDE
  does not have camera permission. Try `--camera 1`; on macOS, also
  check System Settings → Privacy & Security → Camera.

- **`ERROR: model checkpoint not found`.**
  You haven't trained yet. Run `python src/train.py` (Phase 4)
  first.

- **YuNet detects a face but the label is clearly wrong.**
  The detector is doing its job; the *classifier* sees a face it
  finds confusing (unusual angle, very dim lighting, partial mask).
  Try better lighting and a frontal pose. Confidence around 50 %
  means the model is genuinely uncertain.

- **The video looks frozen or very slow.**
  CPU-only inference is fine but slower. On Apple Silicon, make sure
  PyTorch is built with MPS support (`torch.backends.mps.is_available()`
  should return `True`). You can also raise `--min_face_size` so
  only large, clear faces are processed.

---

## 15. What's next

This is the last piece of the project. End-to-end you now have:

1. **Phase 1**: project skeleton.
2. **Phase 2**: dataset preparation (`data/{train,val,test}/`).
3. **Phase 3**: from-scratch CNN architecture (`MaskCNN`).
4. **Phase 4**: training loop + saved checkpoint + curves.
5. **Phase 5**: test-set evaluation (metrics + confusion matrix).
6. **Phase 6**: single-image prediction tool.
7. **Phase 7**: real-time webcam demo with a robust DNN face detector
   (this phase).

Optional next steps if you want to push further:

- Replace YuNet with a larger DNN face detector (e.g. RetinaFace) for
  even harder occlusion / extreme angles.
- Add an **"incorrect mask"** class (mask under chin / nose exposed)
  and retrain. The whole pipeline will pick it up automatically
  because `num_classes` is read from `class_to_idx` in the
  checkpoint.
- Record the live demo with `cv2.VideoWriter` so you can include the
  result in a project report.
