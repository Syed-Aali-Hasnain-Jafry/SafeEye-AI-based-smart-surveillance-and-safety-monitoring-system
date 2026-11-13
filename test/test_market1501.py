
from pathlib import Path
from collections import defaultdict
import cv2
import re

BASE = Path(__file__).resolve().parent.parent
ROOT = BASE / "datasets" / "Market-1501-v15.09.15"

# Find either common extraction structure
candidates = [
    ROOT / "Market-1501-v15.09.15",
    ROOT,
]

dataset = next(
    (p for p in candidates if (p / "query").is_dir()),
    None
)

if dataset is None:
    raise FileNotFoundError(
        f"Market-1501 folders not found inside {ROOT}"
    )

folders = [
    "bounding_box_train",
    "bounding_box_test",
    "query",
]

pattern = re.compile(r"^(-?\d+)_c(\d+)")

print("\nMARKET-1501 DATASET CHECK")
print("Location:", dataset)

for folder in folders:
    directory = dataset / folder

    if not directory.is_dir():
        raise FileNotFoundError(
            f"Missing folder: {directory}"
        )

    images = sorted(directory.glob("*.jpg"))
    people = set()
    cameras = set()
    malformed = 0

    for path in images:
        match = pattern.match(path.name)

        if not match:
            malformed += 1
            continue

        person_id = int(match.group(1))
        camera_id = int(match.group(2))

        if person_id > 0:
            people.add(person_id)

        cameras.add(camera_id)

    print(f"\nFolder: {folder}")
    print("Images:", len(images))
    print("Labeled identities:", len(people))
    print("Camera IDs:", sorted(cameras))
    print("Invalid filenames:", malformed)

    if not images:
        raise RuntimeError(f"No images in {folder}")

    first = cv2.imread(str(images[0]))

    if first is None:
        raise RuntimeError(
            f"Cannot read image: {images[0]}"
        )

    print("First image shape:", first.shape)
    print("Image reading: SUCCESS")

print("\nMarket-1501 basic validation completed!")
