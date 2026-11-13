
from pathlib import Path
from collections import Counter
import json
import cv2

BASE = Path(__file__).resolve().parent.parent
ROOT = BASE / "datasets" / "RSTPReid"

if not ROOT.is_dir():
    raise FileNotFoundError(
        f"Create the RSTPReid folder at: {ROOT}"
    )

print("\n===== RSTPReid DATASET CHECK =====")

# Find image files, even if folders are nested
images = sorted(
    p for p in ROOT.rglob("*")
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

# Find JSON annotations
json_files = sorted(ROOT.rglob("*.json"))

print("Dataset folder:", ROOT)
print("Images found:", len(images))
print("JSON files found:", len(json_files))

if not images:
    raise RuntimeError("No images found. Check extraction.")

# Check first 10 images
for path in images[:10]:
    img = cv2.imread(str(path))

    if img is None:
        raise RuntimeError(f"Unreadable image: {path}")

print("First 10 images: READABLE")

if not json_files:
    raise RuntimeError("No annotation JSON found.")

for json_path in json_files:
    print("\nAnnotation file:", json_path.relative_to(ROOT))

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    print("JSON root type:", type(data).__name__)

    if isinstance(data, list):
        print("Annotation records:", len(data))
        records = data

    elif isinstance(data, dict):
        print("JSON keys:", list(data.keys())[:15])
        records = next(
            (
                value for value in data.values()
                if isinstance(value, list)
                and value
                and isinstance(value[0], dict)
            ),
            []
        )
    else:
        records = []

    if records:
        sample = records[0]
        print("Sample record fields:", list(sample.keys()))

        if "split" in sample:
            counts = Counter(
                str(r.get("split", "unknown"))
                for r in records
            )
            print("Split counts:", dict(counts))

        if "captions" in sample:
            print("Sample captions:", sample["captions"][:2])

        if "file_path" in sample:
            print("Sample image path:", sample["file_path"])

print("\nRSTPReid basic file validation completed!")
