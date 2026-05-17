# Phase 8 — Final Reporting and Documentation

Phases 1 → 7 produced **a working system**: data preparation, a
from-scratch CNN, training, evaluation, single-image prediction, and
a real-time live-camera application with a robust DNN face detector.

Phase 8 produces the **artifacts you actually hand in**: a polished
`README.md`, a school-style final report, and this short note that
explains *how* to use those documents and what a fair, honest
project report looks like.

This document is intentionally short. It is meant to be read once,
right before you submit.

---

## 1. What Phase 8 does

Phase 8 has three deliverables, all in this repository:

1. **`README.md`** — the front page of the project.
2. **`docs/final_project_report.md`** — the long, structured report
   that goes into your school submission (printed or PDF).
3. **`docs/phase_8_reporting.md`** — this file.

Phase 8 does **not** add any new code. It only wraps up what the
earlier phases produced.

---

## 2. Why documentation is important

A school project is graded on three things, not one:

- **The system works.** The code runs and produces sensible
  predictions on a webcam.
- **You understand what you built.** You can explain *why* every
  piece is there, not just *what* it does.
- **Someone else can reproduce your results.** Given your code and
  your README, a classmate (or a future-you) should be able to set
  up the project and re-run the experiments.

Good documentation is the bridge between (1) and (2)/(3). Without
it, the grader has to guess what you did. With it, your project
*looks* finished — because it is.

---

## 3. What `README.md` is for

`README.md` is the **first thing anyone reads** when they open the
project folder, on GitHub, or in an IDE. It is short by design and
optimized for *quickly answering*:

- What is this project?
- What does it do?
- How do I install it?
- How do I run each script?
- Where do the results live?
- What are the known limitations?

Use the README to **orient** the reader. Do not try to fit your
entire project report into it. Anything long-form — methodology,
results commentary, future work — belongs in
`docs/final_project_report.md`.

---

## 4. What `docs/final_project_report.md` is for

`docs/final_project_report.md` is your **school-style report**. It
follows a standard academic structure:

1. Introduction
2. Problem Statement
3. Objectives
4. System Design
5. Dataset Preparation
6. Model Architecture
7. Training Process
8. Evaluation
9. Live Camera Implementation
10. Debugging and Improvement
11. Limitations
12. Future Improvements
13. Conclusion

It is written so that **someone who has never opened your code** can
follow your reasoning end-to-end. The Evaluation section contains
**placeholders** (e.g. `[Insert final accuracy here]`) — these are
where you paste the latest numbers and the four PNG plots from
`results/` before you submit.

This file is designed to be exported to PDF or printed as the formal
report.

---

## 5. How to include result images in your report

The four images in `results/` are the visual evidence that the
project actually trained and evaluated. They go inside the
**Evaluation** section of the final report at the marked positions:

| Placeholder in the report                | Image to insert                       | What the image shows                                                                  |
| ----------------------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------- |
| `[Insert loss_plot.png here]`             | `results/loss_plot.png`               | Training loss vs validation loss across epochs.                                       |
| `[Insert accuracy_plot.png here]`         | `results/accuracy_plot.png`           | Training accuracy vs validation accuracy across epochs.                               |
| `[Insert confusion_matrix.png here]`      | `results/confusion_matrix.png`        | 2 × 2 grid of (true class, predicted class) counts.                                    |
| `[Insert classification_report.png here]` | `results/classification_report.png`   | Per-class precision / recall / F1-score / support, plus macro and weighted averages.  |

Tips for embedding:

- In **Markdown**, use `![Loss plot](../results/loss_plot.png)` — the
  relative path resolves correctly when the report lives in `docs/`.
- In **PDF / Word** reports, paste each image **directly under** the
  paragraph that introduces it, with a short caption like
  *"Figure 3: Confusion matrix on the test set."*.
- Keep the **plot title visible**: do not crop the image so tightly
  that the axis labels disappear.
- After each image, write **one or two sentences** explaining what
  it shows in your specific run ("Both curves drop together with no
  sign of overfitting", "Most errors are…", etc.). Numbers without
  commentary look weak.

---

## 6. What should be submitted

A clean, complete submission usually includes:

1. **`docs/final_project_report.md`** (or its PDF / printed
   version), with all placeholders replaced and the four images
   embedded.
2. **`README.md`** as the project's front page.
3. The **source code** in `src/`:
   `prepare_data.py`, `model.py`, `train.py`, `evaluate.py`,
   `predict_image.py`, `live_camera.py`.
4. **`requirements.txt`** so the grader can install dependencies
   with one command.
5. The **trained checkpoint** at `models/mask_classifier.pth` and
   the auto-downloaded YuNet ONNX file in `models/`.
6. The **result images** in `results/`:
   `loss_plot.png`, `accuracy_plot.png`, `confusion_matrix.png`,
   `classification_report.png`.
7. *(Optional but recommended)* a **short demo video or
   screenshot** of the live camera in action, showing both
   `with_mask` and `without_mask` boxes.

You generally do **not** need to ship the dataset itself
(`raw_data/`, `data/`) — those are large and reproducible from the
scripts. Just describe the dataset in the report.

---

## 7. How to describe the Haar → DNN improvement

This story is one of the most valuable parts of the report, because
it demonstrates **engineering thinking**: you noticed a failure,
formed a hypothesis, tested it, and fixed it. Describe it in three
short stages:

1. **What you observed.**
   *"With the Haar Cascade detector, the live camera correctly
   classified faces without a mask, but no bounding box was drawn
   when the user wore a real mask. A masked face shown on a phone
   screen, however, was still detected."*

2. **Why it happened.**
   *"Haar Cascades rely on intensity patterns across the entire
   face — the eyes vs. the cheeks, the mouth vs. the upper lip. A
   mask covers the mouth and chin, so several of those checks fail
   and the cascade rejects the candidate. The classifier only runs
   after a face is detected, so when detection fails, no prediction
   is shown."*

3. **How you fixed it.**
   *"We switched the detector to OpenCV's DNN face detector
   (YuNet), which was trained on a dataset that includes occluded
   faces. YuNet detects masked faces reliably, so the classifier
   runs and the bounding box / label is drawn. Both detectors are
   still selectable via `--detector dnn` / `--detector haar` for
   comparison."*

Add a short paragraph noting the **evidence** you have for the fix
(detection counts on the same images, screenshots before / after, or
your live-camera observation). This makes the story concrete.

---

## 8. How to avoid claiming fake results

Reports lose marks fast when the numbers don't add up. A few simple
rules keep yours honest:

- **Re-run `python src/evaluate.py` *just before* you write the
  Evaluation section.** Take the numbers it prints. Use those exact
  values; do not round dramatically and do not "polish" them.
- **Do not invent or remember numbers.** A line like *"we achieved
  99 % accuracy"* with no plot to back it up is worse than a
  modest, real number.
- **Use placeholders during drafting.** This report intentionally
  uses `[Insert final accuracy here]` so you cannot accidentally
  ship a stale number from an earlier run. Replace each placeholder
  one by one when the artifacts are final.
- **Match every number to an artifact.** The accuracy you report
  must match the value printed by `evaluate.py` *and* the value
  visible on the confusion matrix.
- **Acknowledge limitations honestly** (Section 11 of the report).
  Saying "the model struggles with side-view faces" is a strength,
  not a weakness — it shows you understand your system.

If you re-train the model after writing the report, **redo the
Evaluation section**: re-run `evaluate.py`, replace the four
images, and update the numbers. Inconsistent text and figures are
the most common reason a project report loses points unnecessarily.

---

## 9. Final submission checklist

A short list to run through before you click "submit":

- [ ] `python src/train.py` and `python src/evaluate.py` were
      executed in the same session, on the latest code.
- [ ] `models/mask_classifier.pth` is the checkpoint that produced
      the numbers in the report.
- [ ] All four PNGs in `results/` were generated by that checkpoint
      and are embedded in the report.
- [ ] Every `[Insert ... here]` placeholder in
      `docs/final_project_report.md` has been replaced with a real
      number.
- [ ] The Haar → DNN debugging story is included with concrete
      evidence.
- [ ] `README.md` correctly describes how to install, prepare,
      train, evaluate, and run the live camera.
- [ ] Limitations and future improvements are discussed honestly.
- [ ] `python src/live_camera.py --camera 0 --detector dnn` works
      end-to-end and produces correct boxes for both classes.

When every box is ticked, the project is ready to ship.
