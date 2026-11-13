
from pathlib import Path
import csv
import cv2
import numpy as np
import torch

from transformers import (
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification,
)

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "RWF-2000"
MANIFEST = ROOT / "test" / "rwf2000_manifest.csv"

MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"

# Small initial test: 2 Fight and 2 NonFight clips
SAMPLES_PER_CLASS = 2
NUM_FRAMES = 16

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_video(video_path):
    """Read video sequentially and sample 16 frames."""

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None

    frames = []

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            # Convert OpenCV BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    finally:
        cap.release()

    if not frames:
        return None

    indices = np.linspace(
        0, len(frames) - 1, NUM_FRAMES,
        dtype=int,
    )

    return [frames[i] for i in indices]


def choose_samples():
    """Get two validation videos from each class."""

    selected = {0: [], 1: []}

    with MANIFEST.open(
        "r", newline="", encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            if row["split"] != "val":
                continue

            label = int(row["label"])

            if len(selected[label]) < SAMPLES_PER_CLASS:
                selected[label].append(
                    (
                        DATASET / row["path"],
                        label,
                    )
                )

    return selected[1] + selected[0]


def main():

    print("=" * 50)
    print("SAFEYE - RWF-2000 PRETRAINED MODEL TEST")
    print("=" * 50)

    print("Device:", device)

    if not MANIFEST.exists():
        print("Manifest missing:", MANIFEST)
        return

    print("\nLoading pretrained VideoMAE...")

    processor = VideoMAEImageProcessor.from_pretrained(
        MODEL_NAME
    )

    model = VideoMAEForVideoClassification.from_pretrained(
        MODEL_NAME
    )

    model.to(device)
    model.eval()

    print("Model loaded successfully!")

    samples = choose_samples()

    if len(samples) != SAMPLES_PER_CLASS * 2:
        print("Not enough validation samples.")
        return

    for index, (video_path, label) in enumerate(
        samples, start=1
    ):

        true_label = "Fight" if label == 1 else "NonFight"

        print(f"\n--- VIDEO {index} ---")
        print("Filename:", video_path.name)
        print("Actual RWF label:", true_label)

        frames = load_video(video_path)

        if frames is None:
            print("FAILED: Could not read video")
            continue

        inputs = processor(
            frames,
            return_tensors="pt",
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

        top_scores, top_indices = torch.topk(
            probabilities, k=5
        )

        print("\nTop 5 predicted activities:")

        for rank, (score, class_id) in enumerate(
            zip(top_scores, top_indices),
            start=1,
        ):

            name = model.config.id2label[
                class_id.item()
            ]

            print(
                f"{rank}. {name}: "
                f"{score.item():.4f}"
            )

    print("\n" + "=" * 50)
    print("PRETRAINED MODEL TEST COMPLETED")
    print("=" * 50)

    print(
        "NOTE: Predictions are Kinetics activity labels,"
        "\nnot binary Fight/NonFight decisions."
    )


if __name__ == "__main__":
    main()
