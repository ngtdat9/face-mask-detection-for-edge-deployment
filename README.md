# Real-Time Face Mask Detection Using OpenCV and PyTorch

## 1. Project Title

**Real-Time Face Mask Detection Using OpenCV and PyTorch**

## 2. Project Overview

This project detects faces from a live camera feed (or from a still
image) and classifies each detected face as either **`with_mask`**
or **`without_mask`**. It is built as a **two-stage pipeline**:

1. **OpenCV** finds where the faces are in each frame.
2. A custom **PyTorch CNN** classifier trained from scratch decides,
   for each face, whether the person is wearing a mask.

The result is a real-time application that draws a colored bounding
box and a label such as `with_mask 96.4%` around every face it sees.

## 3. System Pipeline

```
Camera frame
    -> OpenCV DNN face detector (YuNet)
    -> face bounding box
    -> crop face region
    -> resize to 128x128 + normalize
    -> PyTorch CNN classifier (MaskCNN)
    -> with_mask / without_mask prediction + confidence
    -> display bounding box, label, and probability
```

The two stages are **independent**. OpenCV handles localization
("where is the face?") and the CNN handles classification ("what is
this face wearing?").
## 4. Results

