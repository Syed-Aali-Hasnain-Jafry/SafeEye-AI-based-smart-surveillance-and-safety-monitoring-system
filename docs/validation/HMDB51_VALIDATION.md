# HMDB51 — VideoMAE Smoking Action Recognition Validation

## Purpose
Verify that SafEye can load HMDB51 videos and use a pretrained VideoMAE model to produce smoking-related action predictions.

## Test Environment
- Dataset: HMDB51
- Model: Pretrained VideoMAE
- Python: 3.11.9
- Libraries: PyTorch, Transformers, OpenCV
- Device: CPU

## Test Commands
.\.venv\Scripts\python.exe .\test\test_hmdb51.py

.\.venv-reid\Scripts\python.exe .\test\test_hmdb51_model.py

## Observed Results
- HMDB51 dataset validation completed successfully.
- VideoMAE model weights loaded successfully.
- Three sample videos were tested.
- 16 frames were selected from each video.
- Video 1: 60 frames read; top prediction smoking (0.0942).
- Video 2: 80 frames read; top prediction smoking hookah (0.3924).
- Video 3: 77 frames read; top prediction smoking (0.4566).
- All three samples had a smoking-related activity ranked first.

## Validation Status
PASS — Functional model inference and dataset-loading tests.

## Limitations
- These predictions use general pretrained action-recognition labels.
- The scores are not calibrated smoking probabilities.
- Only three smoking-related videos were examined.
- Smoking versus non-smoking classification has not been evaluated.
- Precision, recall, F1-score and false-alarm rates remain unmeasured.
- A deployment-ready smoking detector has not been established.

## Next Development Responsibility
Alisha will develop and evaluate the incident-detection module using separate smoking and non-smoking examples.

## Data Handling
Keep datasets, pretrained weights and generated outputs local.
