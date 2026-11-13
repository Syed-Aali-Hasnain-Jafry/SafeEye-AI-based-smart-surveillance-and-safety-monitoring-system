
from pathlib import Path
import cv2
import numpy as np
import torch

from transformers import (
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification
)

BASE = Path(__file__).resolve().parent.parent
VIDEO_DIR = BASE / "datasets" / "hmdb51_org" / "smoke"

MODEL_ID = "MCG-NJU/videomae-base-finetuned-kinetics"

print("Loading VideoMAE...")

processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)
model = VideoMAEForVideoClassification.from_pretrained(MODEL_ID)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("Model loaded successfully!")
print("Device:", device)

# Check whether the pretrained model supports smoking
labels = model.config.id2label
smoke_labels = [
    label for label in labels.values()
    if "smok" in label.lower()
]

print("Smoking-related model labels:", smoke_labels)


def load_frames(video_path, count=16):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    # Read frames sequentially instead of seeking
    all_frames = []

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        all_frames.append(frame)

    cap.release()

    if not all_frames:
        raise RuntimeError(f"No readable frames: {video_path}")

    # Select 16 evenly spaced frames
    indices = np.linspace(
        0,
        len(all_frames) - 1,
        count,
        dtype=int
    )

    selected_frames = [all_frames[i] for i in indices]

    print(f"Read {len(all_frames)} frames successfully")
    print(f"Selected {len(selected_frames)} frames for VideoMAE")

    return selected_frames


videos = sorted(VIDEO_DIR.glob("*.avi"))[:3]

if not videos:
    raise FileNotFoundError(
        f"No smoking videos found in {VIDEO_DIR}"
    )

for video in videos:
    print("\nTesting:", video.name)

    frames = load_frames(video)

    inputs = processor(
        frames,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits, dim=-1
    )[0]

    top_values, top_indices = torch.topk(
        probabilities, 5
    )

    print("Top 5 predictions:")

    for score, index in zip(
        top_values, top_indices
    ):
        label = labels[int(index)]
        print(f"{label}: {score.item():.4f}")

print("\nHMDB51 VideoMAE test completed!")
