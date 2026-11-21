from pathlib import Path

import cv2

from src.safeye.tracking.person_tracker import PersonTracker


# Project directories
ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "yolo26n.pt"

IMAGE_DIR = (
    ROOT
    / "datasets"
    / "MOT17"
    / "train"
    / "MOT17-02-FRCNN"
    / "img1"
)


def main():
    # Load the first 10 MOT17 frames
    images = sorted(IMAGE_DIR.glob("*.jpg"))[:10]

    if not images:
        raise FileNotFoundError("MOT17 images not found.")

    # Create our reusable tracker
    tracker = PersonTracker(
        model_path=MODEL_PATH,
        camera_id="mot17_camera_01",
        confidence=0.25,
    )

    for image_path in images:
        frame = cv2.imread(str(image_path))

        if frame is None:
            print(f"Skipping unreadable frame: {image_path.name}")
            continue

        # Process one frame
        result = tracker.process_frame(frame)

        print(
            f"Camera: {result['camera_id']} | "
            f"Frame: {result['frame_number']} | "
            f"People tracked: {len(result['tracks'])}"
        )

        for person in result["tracks"]:
            print(
                f"  Track ID: {person['track_id']} | "
                f"Confidence: {person['confidence']}"
            )


if __name__ == "__main__":
    main()