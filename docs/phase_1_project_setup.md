# Phase 1 — Project Setup

This document explains what we are building, how the project is organized,
and the *why* behind the main technical choices. It is written for beginners,
so terms are introduced gently before they are used.

---

## 1. Purpose of the Project

The goal of this project is to build a **real-time face mask detection
system**. Given a live video stream from a webcam, the program should:

1. Find every face that appears in the current frame.
2. Decide, for each face, whether the person is wearing a mask or not.
3. Draw a colored rectangle around each face and write the predicted label
   (e.g. `with_mask 0.97`) on top of the video.

This kind of system is a small but realistic example of **computer vision**:
teaching a computer to "see" and make decisions about images.

We split the problem into two independent steps because each step is much
easier to solve on its own than as one big problem:

- **Step A — Face Detection:** *Where* are the faces in the image?
- **Step B — Mask Classification:** *Is this face wearing a mask?*

---

## 2. Project Folder Structure

After Phase 1, the project looks like this:

```
face-mask-detection/
├── data/
├── docs/
├── models/
├── raw_data/
│   ├── with_mask/
│   └── without_mask/
├── results/
├── src/
├── requirements.txt
└── README.md
```

### 2.1 Role of Each Folder

| Folder        | What lives here                                                                 | Why it exists                                                                                          |
| ------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `raw_data/`   | The original images, grouped into `with_mask/` and `without_mask/` subfolders.  | This is the **untouched source of truth**. We never modify these files so we can always start over.    |
| `data/`       | Cleaned and split datasets, e.g. `train/`, `val/`, `test/`.                     | The model is trained from this folder. Keeping it separate from `raw_data/` makes experiments safe.    |
| `models/`     | Saved PyTorch model files (`.pt` or `.pth` checkpoints).                        | Training is expensive. Saving the trained weights lets us reload the model later without retraining.   |
| `results/`    | Training curves, confusion matrices, evaluation reports, demo screenshots.      | A place to collect every output we want to look at or include in a report.                             |
| `src/`        | All Python source code (data loading, model definition, training, inference).   | Keeps code separate from data and outputs, which is a standard machine-learning project layout.        |
| `docs/`       | Markdown notes for each phase of the project (this file lives here).            | Explains the *why* behind decisions so the project is understandable later (or by someone else).       |
| `requirements.txt` | A list of Python packages the project depends on.                          | Anyone can recreate the environment with one command: `pip install -r requirements.txt`.               |
| `README.md`   | Short, high-level description of the project.                                   | The first file a visitor reads on GitHub or when opening the folder.                                   |

This separation (raw data vs processed data vs code vs models vs results)
is a **best practice** in machine learning projects. It avoids the very
common mistake of mixing scripts, datasets, and trained weights in the same
folder, which becomes impossible to clean up later.

---

## 3. Why OpenCV for Face Detection?

**OpenCV** (Open Source Computer Vision Library) is a popular library for
working with images and video. We use it for two things:

1. **Reading frames from the webcam** in real time.
2. **Detecting faces** in those frames.

Why not train our own face detector?

- Face detection is a **solved problem**. OpenCV ships with detectors that
  already work very well, including the classical **Haar Cascade** detector
  and a more modern **DNN-based face detector**.
- Training a face detector from scratch requires a huge dataset of labeled
  bounding boxes and a lot of compute. That is out of scope for this
  project.
- By reusing OpenCV's detector we get a fast, reliable answer to the
  question *"where are the faces?"* with just a few lines of code.

In short: OpenCV gives us **face localization for free**, so we can spend
our effort on the part that is actually interesting for this project — the
mask classifier.

---

## 4. Why PyTorch for the Mask Classifier?

**PyTorch** is a deep-learning framework from Meta. It lets us define and
train neural networks in plain Python.

We use PyTorch because:

- It is **beginner-friendly**: models look like normal Python classes, and
  errors are easy to debug.
- It supports **automatic differentiation**, so we don't have to compute
  gradients by hand when training.
- It runs on **CPU and GPU** with the same code, so we can train on a
  laptop and later move to a GPU if needed.
- It has a rich ecosystem (`torchvision`) with image transforms, datasets,
  and pretrained models, which speeds up everyday tasks.

For this project we will **train a CNN from scratch** instead of using a
pretrained model. That is intentional: the goal is to *learn how a CNN
classifier is built and trained end-to-end*, not just to ship the highest
possible accuracy.

A **CNN (Convolutional Neural Network)** is a kind of neural network that
is very good at images. It learns small visual patterns (edges, textures,
shapes like a strap or a mouth) in early layers and combines them into
bigger concepts (like "a piece of fabric covering the lower face") in
deeper layers. That is exactly the kind of pattern we need to recognize a
mask.

---

## 5. Face Detection vs Mask Classification

These two tasks are easy to confuse but they answer very different
questions.

| Aspect            | Face Detection                                  | Mask Classification                                  |
| ----------------- | ----------------------------------------------- | ---------------------------------------------------- |
| Question answered | *Where* in the image is a face?                 | *What* is in this cropped image?                     |
| Input             | A full image / video frame.                     | A small image of a single face.                      |
| Output            | A list of bounding boxes (x, y, width, height). | A class label (`with_mask` / `without_mask`) + score.|
| Tool we use       | OpenCV (pretrained detector).                   | PyTorch (CNN trained by us).                         |
| Number of outputs | Zero, one, or many faces per frame.             | Exactly one label per cropped face.                  |

A useful mental model:

- The **detector** is like a highlighter that circles every face in the
  picture.
- The **classifier** is like a small expert that looks at one circled face
  at a time and says "mask" or "no mask".

The two are chained together: the detector tells the classifier *what to
look at*, and the classifier tells us *what it is*.

---

## 6. Expected Final Pipeline

When the project is finished, running the application will execute this
loop, many times per second:

```
┌──────────────────────────┐
│ 1. Read frame from webcam│  (OpenCV: cv2.VideoCapture)
└─────────────┬────────────┘
              ▼
┌──────────────────────────┐
│ 2. Detect faces in frame │  (OpenCV face detector)
└─────────────┬────────────┘
              ▼
┌──────────────────────────────────────┐
│ 3. For each detected face:           │
│    a. Crop the face region           │
│    b. Resize + normalize the crop    │  (preprocessing for the CNN)
│    c. Run the PyTorch CNN classifier │
│    d. Get label + probability        │
└─────────────┬────────────────────────┘
              ▼
┌──────────────────────────────────────┐
│ 4. Draw bounding box + label on frame│  (OpenCV drawing functions)
│    - Green for `with_mask`           │
│    - Red   for `without_mask`        │
└─────────────┬────────────────────────┘
              ▼
┌──────────────────────────┐
│ 5. Show the frame on     │
│    screen, repeat.       │
└──────────────────────────┘
```

So the **detector** answers *where*, the **classifier** answers *what*,
and OpenCV handles the I/O (reading from the camera and drawing on the
screen).

---

## 7. What Phase 1 Delivers

By the end of this phase we have:

- A clean, organized folder structure ready for code, data, and results.
- A `requirements.txt` that pins the Python libraries we will need.
- A `README.md` that explains the project at a glance.
- This document (`docs/phase_1_project_setup.md`) explaining the plan and
  the reasoning behind it.

We have **not** written any model code yet — that is intentional. A solid
project layout and a clear plan make every later phase much easier.

---

## 8. What's Next

Phase 2 will focus on **data preparation**: inspecting `raw_data/`,
checking class balance, splitting the images into `train/`, `val/`, and
`test/` subsets inside `data/`, and writing a PyTorch `Dataset` /
`DataLoader` to feed those images to the model efficiently.
