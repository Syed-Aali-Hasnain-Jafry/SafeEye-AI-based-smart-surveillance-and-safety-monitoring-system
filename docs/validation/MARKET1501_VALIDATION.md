# Market-1501 — OSNet Person Re-Identification Validation

## Purpose
Verify that SafEye can load a pretrained OSNet model and retrieve images of the same person from different cameras.

## Test Environment
- Python: 3.11.9
- Libraries: PyTorch, OpenCV, NumPy
- Model: Pretrained OSNet
- Dataset: Market-1501

## Test Command
.\.venv-reid\Scripts\python.exe .\test\test_osnet_matching.py

## Observed Results
- Pretrained OSNet weights loaded successfully.
- Query image: 0001_c1s1_001051_00.jpg
- The model generated similarity scores and ranked five candidate images.
- The correct identity appeared at Rank 1.
- Correct match: 0001_c2s1_001976_01.jpg
- Correct match similarity: 0.6793
- Result image saved to: runs/osnet/top5_matches.jpg

## Validation Status
PASS — Functional person ReID retrieval test.

## Limitations
- This is a single-query demonstration.
- Dataset-wide Rank-1 and mAP have not been measured.
- Similarity scores are not calibrated identity probabilities.
- Cross-camera performance requires further evaluation.
- Model weights, dataset images and generated results remain local.
