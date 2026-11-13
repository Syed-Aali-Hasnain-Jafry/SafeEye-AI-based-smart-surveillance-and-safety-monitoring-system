
from pathlib import Path
from collections import Counter
import cv2

# SafEye project directory
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_DIR / "datasets" / "RWF-2000"

VIDEO_EXTENSIONS = {
    ".avi", ".mp4", ".mov", ".mkv",
    ".webm", ".mpeg", ".mpg"
}

EXPECTED_COUNTS = {
    "train/Train_Fight": 800,
    "train/Train_NonFight": 800,
    "val/Val_Fight": 200,
    "val/Val_NonFight": 200,
}


def get_category(video_path):
    """Identify the dataset split and class."""
    relative = video_path.relative_to(DATASET_DIR)
    folder_names = [part.lower() for part in relative.parts[:-1]]

    if "train" in folder_names:
        split = "train"
    elif "val" in folder_names:
        split = "val"
    else:
        return None

    if any("nonfight" in name for name in folder_names):
        label = "NonFight"
    elif any("fight" in name for name in folder_names):
        label = "Fight"
    else:
        return None

    return f"{split}/{split.capitalize()}_{label}"


def check_video(video_path):
    """Check whether OpenCV can read the first frame."""
    cap = cv2.VideoCapture(str(video_path))

    try:
        if not cap.isOpened():
            return False

        success, frame = cap.read()
        return success and frame is not None
    finally:
        cap.release()


def main():
    print("=" * 50)
    print("SAFEYE - RWF-2000 DATASET VALIDATION")
    print("=" * 50)

    if not DATASET_DIR.is_dir():
        print(f"ERROR: Dataset folder not found:\n{DATASET_DIR}")
        return

    print(f"\nDataset location:\n{DATASET_DIR}")

    videos = sorted(
        path for path in DATASET_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
    )

    counts = Counter()
    uncategorized = []

    for video in videos:
        category = get_category(video)

        if category is None:
            uncategorized.append(video)
        else:
            counts[category] += 1

    print("\n--- DATASET COUNTS ---")

    for category, expected in EXPECTED_COUNTS.items():
        actual = counts[category]
        status = "PASS" if actual == expected else "CHECK"

        print(
            f"{category}: {actual}/{expected} [{status}]"
        )

    total_expected = sum(EXPECTED_COUNTS.values())
    total_found = len(videos)

    print(f"\nTotal videos: {total_found}/{total_expected}")
    print(f"Uncategorized videos: {len(uncategorized)}")

    if uncategorized:
        print("\nFirst uncategorized files:")
        for video in uncategorized[:10]:
            print(video.relative_to(DATASET_DIR))

    print("\n--- VIDEO READABILITY TEST ---")

    # Test up to 5 videos from each category
    passed = 0
    failed = 0

    for category in EXPECTED_COUNTS:
        category_videos = [
            video for video in videos
            if get_category(video) == category
        ]

        print(f"\nTesting {category}:")

        for video in category_videos[:5]:
            if check_video(video):
                passed += 1
                print(f"PASS: {video.name}")
            else:
                failed += 1
                print(f"FAIL: {video.name}")

    print("\n" + "=" * 50)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 50)

    print(f"Total videos found: {total_found}")
    print(f"Expected videos: {total_expected}")
    print(f"Readable samples: {passed}")
    print(f"Failed samples: {failed}")

    counts_correct = (
        total_found == total_expected
        and len(uncategorized) == 0
        and all(
            counts[category] == expected
            for category, expected in EXPECTED_COUNTS.items()
        )
    )

    if counts_correct and failed == 0 and passed == 20:
        print("\nSUCCESS: Dataset counts and sampled videos passed!")
        print("Note: Not all 2,000 videos were decoded.")
    else:
        print("\nCHECK: Some validation results need attention.")


if __name__ == "__main__":
    main()
