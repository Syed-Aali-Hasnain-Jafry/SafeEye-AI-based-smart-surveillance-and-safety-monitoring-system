# RWF-2000 — Fighting Detection Validation

## Purpose
Validate the RWF-2000 dataset and test pretrained VideoMAE inference on fighting and non-fighting footage.

## Dataset
- Total videos: 2,000
- Training Fight: 800
- Training NonFight: 800
- Validation Fight: 200
- Validation NonFight: 200

## Test Commands
.\.venv\Scripts\python.exe .\test\test_rwf2000.py

.\.venv-reid\Scripts\python.exe .\test\test_rwf2000_model.py

## Dataset Validation Results
- All four dataset folders contained their expected number of videos.
- Initial validation reported 1,989 videos.
- The dataset-count discrepancy was subsequently reported resolved.
- All 20 sampled videos were readable.
- No sampled video failed decoding.

## Pretrained Model Test
- Model: MCG-NJU/videomae-base-finetuned-kinetics
- Device: CPU
- Four validation videos tested: two Fight and two NonFight.
- Video 1, Fight: dining (0.0632).
- Video 2, Fight: swing dancing (0.1082).
- Video 3, NonFight: driving car (0.3844).
- Video 4, NonFight: shaving head (0.0656).

## Validation Status
Dataset counts and sample readability: PASS based on the final reported checks.
Pretrained VideoMAE inference: PASS — functional test only.
Binary Fight/NonFight classification: NOT IMPLEMENTED.

## Important Limitations
- The pretrained model predicts Kinetics activity classes.
- Its output cannot be treated as a reliable Fight/NonFight decision.
- No binary classification metrics have been measured.
- Dataset-wide video readability has not been verified.

## Alisha's Next Tasks
1. Confirm the corrected dataset manifest and split counts.
2. Develop or fine-tune a binary fighting classifier.
3. Evaluate precision, recall, F1-score and confusion matrix.
4. Investigate false positives and false negatives.
5. Integrate validated fighting predictions into SafEye incidents.

## Data Handling
Dataset videos and model weights remain local and must not be committed to Git.
