# Edge benchmark results

- Machine: Apple M1, Darwin 24.6.0
- Software: Python 3.11.7, PyTorch 2.11.0, ONNX Runtime 1.31.1, OpenCV 4.14.0
- Pipeline frames: 300 synthetic 640x480 frames with 1-3 test-set faces each (seed 0)
- Accuracy uses OpenCV resizing (the on-device path), so it can differ slightly from evaluate.py, which resizes with PIL.
- CPU thread limits approximate a small edge board; these are not measurements on an edge device.

| Variant | Format | Backend | Threads | Size (MB) | Params | Test acc | p50 latency (ms) | p95 latency (ms) | Pipeline FPS | Detect (ms) | Classify (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| flatten-128 | PyTorch fp32 | PyTorch CPU | 1 | 18.35 | 4,584,450 | 97.97% | 4.75 | 5.57 | 37.3 | 13.48 | 13.09 |
| flatten-128 | PyTorch fp32 | PyTorch CPU | 4 | 18.35 | 4,584,450 | 97.97% | 3.46 | 3.70 | 61.3 | 7.30 | 8.94 |
| flatten-128 | PyTorch fp32 | PyTorch MPS (reference) | 4 | 18.35 | 4,584,450 | 97.97% | 3.67 | 3.86 | 57.3 | 7.54 | 9.68 |
| flatten-128 | ONNX fp32 | ONNX Runtime CPU | 1 | 18.36 | 4,584,450 | 97.97% | 6.35 | 6.47 | 35.5 | 13.70 | 14.34 |
| flatten-128 | ONNX fp32 | ONNX Runtime CPU | 4 | 18.36 | 4,584,450 | 97.97% | 2.33 | 2.54 | 67.8 | 7.50 | 7.19 |
| flatten-128 | ONNX INT8 | ONNX Runtime CPU | 1 | 4.62 | 4,584,450 | 97.97% | 1.96 | 2.00 | 52.9 | 13.38 | 5.41 |
| flatten-128 | ONNX INT8 | ONNX Runtime CPU | 4 | 4.62 | 4,584,450 | 97.97% | 0.78 | 0.82 | 99.2 | 7.10 | 2.94 |
| gap-128 | PyTorch fp32 | PyTorch CPU | 1 | 1.57 | 389,890 | 98.06% | 4.07 | 4.68 | 37.7 | 13.92 | 12.43 |
| gap-128 | PyTorch fp32 | PyTorch CPU | 4 | 1.57 | 389,890 | 98.06% | 2.59 | 2.94 | 66.8 | 7.24 | 7.61 |
| gap-128 | PyTorch fp32 | PyTorch MPS (reference) | 4 | 1.57 | 389,890 | 98.06% | 3.16 | 3.31 | 70.5 | 7.14 | 6.86 |
| gap-128 | ONNX fp32 | ONNX Runtime CPU | 1 | 1.58 | 389,890 | 98.06% | 5.89 | 5.98 | 36.4 | 13.60 | 13.69 |
| gap-128 | ONNX fp32 | ONNX Runtime CPU | 4 | 1.58 | 389,890 | 98.06% | 1.96 | 2.11 | 73.3 | 7.21 | 6.39 |
| gap-128 | ONNX INT8 | ONNX Runtime CPU | 1 | 0.42 | 389,890 | 98.06% | 1.87 | 1.91 | 53.1 | 13.52 | 5.28 |
| gap-128 | ONNX INT8 | ONNX Runtime CPU | 4 | 0.42 | 389,890 | 98.06% | 0.72 | 0.75 | 99.9 | 7.19 | 2.79 |
| gap-96 | PyTorch fp32 | PyTorch CPU | 1 | 1.57 | 389,890 | 97.27% | 2.31 | 2.56 | 49.2 | 13.57 | 6.71 |
| gap-96 | PyTorch fp32 | PyTorch CPU | 4 | 1.57 | 389,890 | 97.27% | 1.55 | 1.72 | 87.4 | 7.10 | 4.31 |
| gap-96 | PyTorch fp32 | PyTorch MPS (reference) | 4 | 1.57 | 389,890 | 97.27% | 2.68 | 2.95 | 74.4 | 6.88 | 6.47 |
| gap-96 | ONNX fp32 | ONNX Runtime CPU | 1 | 1.58 | 389,890 | 97.27% | 3.36 | 3.51 | 45.9 | 13.45 | 8.19 |
| gap-96 | ONNX fp32 | ONNX Runtime CPU | 4 | 1.58 | 389,890 | 97.27% | 1.27 | 1.36 | 83.6 | 7.63 | 4.27 |
| gap-96 | ONNX INT8 | ONNX Runtime CPU | 1 | 0.42 | 389,890 | 97.27% | 1.07 | 1.10 | 58.9 | 13.60 | 3.34 |
| gap-96 | ONNX INT8 | ONNX Runtime CPU | 4 | 0.42 | 389,890 | 97.27% | 0.48 | 0.50 | 107.1 | 7.29 | 1.99 |
