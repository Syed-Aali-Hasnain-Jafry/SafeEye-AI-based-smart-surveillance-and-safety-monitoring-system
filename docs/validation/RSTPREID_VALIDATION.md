# RSTPReid — CLIP Text-to-Person Retrieval Validation

## Purpose
Verify that SafEye can search for a person using a natural-language description and rank matching images.

## Test Environment
- Python: 3.11.9
- Model: CLIP using Hugging Face libraries
- Dataset: RSTPReid
- Candidate images: 50

## Test Command
.\.venv-reid\Scripts\python.exe .\test\test_rstpreid_retrieval.py

## Query Description
The man is wearing a black coat and black trousers. He is wearing glasses. He walked with his hands on each other.

## Observed Results
- CLIP model weights loaded successfully.
- 50 candidate images were processed.
- Similarity scores were generated.
- Correct identity appeared at Rank 2.
- Correct match: 3901_c14_0008.jpg
- Correct match score: 29.995
- Top-ranked image: 3949_c5_0006.jpg (incorrect identity)
- Result saved to: runs/rstpreid/top5_text_matches.jpg

## Validation Status
PASS — Functional text-to-person retrieval demonstration.
The correct identity was found within the top five results.

## Limitations
- The correct identity was not ranked first.
- This is a single-query test, not dataset-wide evaluation.
- Rank-1, Rank-5 and mAP metrics require systematic evaluation.
- CLIP scores should not be interpreted as calibrated probabilities.
- Dataset images, model weights and generated results remain local.
