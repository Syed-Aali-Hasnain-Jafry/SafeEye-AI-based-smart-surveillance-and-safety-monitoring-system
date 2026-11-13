# MOT17 — YOLO + ByteTrack Validation

## Purpose
Verify that SafEye can detect and track people in MOT17 video frames.

## Test Environment
- Python: 3.15.2
- Libraries: Ultralytics YOLO, OpenCV, PyTorch
- Dataset: MOT17

## Test Command
.\.venv\Scripts\python.exe .\test\test_mot17_tracking.py

## Observed Results
- Required library imports succeeded.
- The tracking script completed successfully.
- Tracking IDs were reported through frame 100.
- Multiple tracking IDs persisted across consecutive reported frames.
- Annotated video output: runs/mot17/tracking.avi

## Validation Status
PASS — Functional tracking smoke test.

## Limitations
- IDF1 and HOTA have not yet been measured.
- Identity switches and tracking accuracy require evaluation.
- Generated videos are kept locally and excluded from Git.
