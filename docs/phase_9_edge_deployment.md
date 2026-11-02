# Phase 9 — Edge Optimization and Deployment

In Phases 1–8 we built a working mask detector that runs on a laptop
with PyTorch. In Phase 9 we prepare it for **edge devices**: small,
CPU-only computers such as a Raspberry Pi or a Jetson board that sit
next to a camera instead of in a data center.

We do not own an edge device, so everything here is measured on a
MacBook (Apple M1) **CPU with a limited number of threads** (1 and 4),
which is the closest honest approximation. The numbers describe this
CPU, not a real edge board, and should always be reported that way.

The phase has four steps:

1. Make the classifier smaller (a new model head).
2. Export it to **ONNX**, a format that runs without PyTorch.
3. Quantize it to **INT8** to shrink it further and speed it up.
4. Benchmark everything and prove the detector choice with data.

---

## 1. Why the original model is too big

`MaskCNN` has four convolutional blocks followed by a classifier
head. The original ("flatten") head flattens the final
`256 × 8 × 8` feature map into 16,384 numbers and feeds them to a
fully connected layer with 256 outputs:

```
16,384 × 256 ≈ 4.2 million weights
```

That one layer holds **about 92% of the model's 4.58 M parameters**,
and it only accepts 128×128 inputs because its size is tied to the
feature-map size.

### The fix: global average pooling (the "gap" head)

Instead of flattening, we **average each of the 256 feature maps
down to a single number**, then use a tiny linear layer:

```
256 × 8 × 8  --global average pool-->  256  --Dropout(0.2)-->  Linear(256, 2)
```

| Head | Parameters | Checkpoint size | Input size |
|---|---|---|---|
| `flatten` (original) | 4,584,450 | 18.35 MB | 128×128 only |
| `gap` (compact) | 389,890 | 1.57 MB | any size |

That is **11.8× fewer parameters**. Because pooling works at any
resolution, we also trained a 96×96 version, which needs about
half the computation of 128×128.

Train them with:

```bash
python src/train.py --head gap --model_path models/mask_classifier_gap.pth --plot_prefix gap_
python src/train.py --head gap --image_size 96 --model_path models/mask_classifier_gap96.pth --plot_prefix gap96_
```

The checkpoint stores `head` and `image_size`, so `evaluate.py`,
`predict_image.py` and `live_camera.py` rebuild the right model
automatically.

---

## 2. Exporting to ONNX

**ONNX** (Open Neural Network Exchange) is a standard file format for
neural networks. An `.onnx` file contains the whole computation graph
and the weights, and it can be run by **ONNX Runtime**, a small
inference engine available for Linux ARM boards, Android, iOS and
Windows. The device then needs only `onnxruntime`, `opencv-python`
and `numpy`, not PyTorch.

```bash
python src/export_onnx.py --model_path models/mask_classifier_gap.pth --quantize
```

`src/export_onnx.py` does three things:

1. **Exports** the model with a dynamic batch size, so it can classify
   several faces in one call.
2. **Stores the preprocessing inside the file** (class order, image
   size, mean and std) as ONNX metadata. The device reads these values
   from the model instead of hard-coding them, so a model and its
   preprocessing can never get out of sync.
3. **Runs a parity check**: it classifies all 1,135 test images with
   both PyTorch and ONNX Runtime and compares the results.

Parity check for the compact 128×128 model:

```
max |logit diff|      : 7.63e-06
prediction agreement  : 100.00%
PyTorch accuracy      : 98.06%
ONNX fp32 accuracy    : 98.06%
```

A logit difference of 0.0000076 is ordinary floating-point rounding:
the exported model is the same model.

---

## 3. INT8 quantization

Normally every weight is a 32-bit float (fp32). **Quantization** stores
weights and activations as 8-bit integers (INT8) instead, which makes
the file about 4× smaller and lets the CPU use fast integer
instructions.

We use **static quantization**: the script first feeds 300 validation
images through the model to measure the typical range of every
activation ("calibration"), then picks the integer scales from those
ranges. Weights are quantized **per channel**, which keeps accuracy
high for convolutional layers.

| Model | fp32 ONNX | INT8 ONNX | Accuracy change |
|---|---|---|---|
| Original (flatten-128) | 18.36 MB | 4.62 MB | 0.00 points |
| Compact (gap-128) | 1.58 MB | **0.42 MB** | 0.00 points |
| Compact (gap-96) | 1.58 MB | 0.42 MB | 0.00 points |

---

## 4. Benchmarking

```bash
python src/benchmark.py                    # synthetic frames
python src/benchmark.py --video clip.mp4   # a recorded webcam clip
```

`src/benchmark.py` finds every model in `models/` and tests each one
with PyTorch and ONNX Runtime on 1 and 4 CPU threads (plus Apple's GPU,
MPS, as a reference only). It measures:

- **Test accuracy**, using the same OpenCV preprocessing as the device.
- **Classifier latency**: the time to classify one face, repeated 500
  times. We report the median (p50) and the 95th percentile (p95),
  which shows how bad the slow cases get.
- **End-to-end FPS**: detection, cropping, preprocessing, classification
  and drawing on 300 synthetic 640×480 frames, each containing 1–3 real
  test faces. Camera capture and screen display are excluded, because
  they depend on the camera, not the model.

The results go to `results/benchmark.md`, `results/benchmark.csv` and
`results/benchmark_latency.png`.

### Results (Apple M1 CPU, 4 threads)

| Model | Size | Test accuracy | Latency p50 | End-to-end FPS |
|---|---|---|---|---|
| Original, PyTorch fp32 | 18.35 MB | 97.97% | 3.46 ms | 61.3 |
| Compact 128, ONNX INT8 | 0.42 MB | 98.06% | 0.72 ms | 99.9 |
| Compact 96, ONNX INT8 | 0.42 MB | 97.27% | 0.48 ms | 107.1 |

With 1 thread, the original model takes 4.75 ms per face (37.3 FPS)
and the compact INT8 model takes 1.87 ms (53.1 FPS).

### Reading the results

- **Size: 44× smaller** with no accuracy loss. The gap head gives
  11.8× and INT8 gives another 3.8×.
- **Classifier: 4.8× faster** on 4 threads and 2.5× faster on 1 thread.
- **Whole pipeline: 1.6× more FPS.** The gain is smaller than the
  classifier gain because, after optimization, **face detection
  dominates**: YuNet takes about 7.2 ms of the ~10 ms per frame, while
  classification takes about 2.8 ms. The next speed-up would have to
  come from the detector (for example, running it on a smaller frame
  or only every few frames), not from the classifier.
- **96×96 versus 128×128:** the 96 model is a third faster per face but
  loses 0.8 accuracy points. We deploy the 128 model by default.

---

## 5. Two bugs the benchmark caught

Measuring carefully exposed two problems that would also have affected
a real device.

**Resizing changed accuracy.** Training resizes images with PIL, which
smooths the image when shrinking it. OpenCV's default `INTER_LINEAR`
does not, so large photos became jagged ("aliasing") and the 96×96
model dropped to 96.92%. Using `cv2.INTER_AREA` when shrinking fixed
it (97.27%), and the 128×128 models now match `evaluate.py` exactly.

**ONNX Runtime slowed down the face detector.** By default, ONNX
Runtime's idle threads keep spinning on the CPU waiting for work,
which stole CPU time from OpenCV's detector and made YuNet go from
about 7 ms to 12 ms per frame. Setting
`session.intra_op.allow_spinning = 0` released those cores and raised
end-to-end FPS from about 64 to 100.

---

## 6. Proving the detector choice

Phase 7 switched from Haar Cascade to YuNet because Haar missed masked
faces. `src/compare_detectors.py` measures this on the whole test set:

```bash
python src/compare_detectors.py
```

| Detector | Masked faces found (n=560) | Unmasked faces found (n=575) | Time per image |
|---|---|---|---|
| Haar Cascade | 56.8% | 90.4% | 13.4 ms |
| YuNet (DNN) | **95.5%** | **99.8%** | **6.0 ms** |

Haar Cascade misses **43%** of masked faces, so the classifier never
even sees them. YuNet finds 95.5% and is also faster.

---

## 7. Running the edge model live

```bash
python src/live_camera.py --backend onnx --onnx_path models/mask_classifier_gap_int8.onnx --threads 4
```

The window shows a live FPS counter. `--backend torch` still runs the
PyTorch checkpoint for comparison.

To run on a real board later, copy `src/live_camera.py`,
`src/edge_utils.py`, the `.onnx` classifier and the YuNet model, and
install `onnxruntime`, `opencv-python` and `numpy`.
