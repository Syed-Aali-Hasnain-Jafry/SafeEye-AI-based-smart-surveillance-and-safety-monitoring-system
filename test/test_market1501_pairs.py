
from pathlib import Path
from collections import defaultdict
import re
import cv2
import numpy as np

BASE = Path(__file__).resolve().parent.parent
ROOT = BASE / "datasets" / "Market-1501-v15.09.15"

if (ROOT / "Market-1501-v15.09.15").is_dir():
    ROOT = ROOT / "Market-1501-v15.09.15"

gallery = ROOT / "bounding_box_test"
pattern = re.compile(r"^(\d+)_c(\d+)")

people = defaultdict(dict)

for path in sorted(gallery.glob("*.jpg")):
    match = pattern.match(path.name)

    if match is None:
        continue

    person_id, camera_id = map(int, match.groups())

    if person_id <= 0:
        continue

    people[person_id].setdefault(camera_id, path)

# Find one labeled identity in two different cameras
pair = None

for person_id, cameras in people.items():
    if len(cameras) >= 2:
        paths = list(cameras.values())
        pair = (person_id, paths[0], paths[1])
        break

if pair is None:
    raise RuntimeError("No cross-camera image pair found")

person_id, path1, path2 = pair

image1 = cv2.imread(str(path1))
image2 = cv2.imread(str(path2))

if image1 is None or image2 is None:
    raise RuntimeError("Unable to read selected images")

# Display both images side by side
image1 = cv2.resize(image1, (128, 256))
image2 = cv2.resize(image2, (128, 256))

combined = np.hstack([image1, image2])

output = BASE / "runs" / "market1501"
output.mkdir(parents=True, exist_ok=True)

cv2.imwrite(str(output / "same_person.jpg"), combined)

print("Person ID:", person_id)
print("Camera 1 image:", path1.name)
print("Camera 2 image:", path2.name)
print("Cross-camera pair saved successfully!")
