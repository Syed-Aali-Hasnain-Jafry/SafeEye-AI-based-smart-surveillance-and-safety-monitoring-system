
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
from sklearn.datasets import fetch_lfw_people

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "datasets" / "LFW"
OUTPUT_DIR = ROOT / "runs" / "lfw"

DATASET_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 45)
print("SAFEYE - LFW DATASET VALIDATION")
print("=" * 45)

print("\nDownloading/loading LFW dataset...")

lfw = fetch_lfw_people(
    data_home=str(DATASET_DIR),
    min_faces_per_person=0,
    resize=0.5,
    color=False,
)

images = lfw.images
labels = lfw.target

print("\n--- DATASET INFORMATION ---")
print("Total images:", len(images))
print("Unique identities:", len(lfw.target_names))
print("Image shape:", images[0].shape)

counts = Counter(labels.tolist())
multiple_images = sum(count >= 2 for count in counts.values())

print("Identities with 2+ images:", multiple_images)

print("\n--- IMAGE READABILITY TEST ---")

passed = 0

for index in range(min(5, len(images))):
    image = images[index]

    if image.size == 0 or not np.isfinite(image).all():
        print(f"FAIL: Image {index + 1}")
        continue

    # Save example face images for visual inspection
    scaled = np.clip(image, 0, 255).astype(np.uint8)

    output_file = OUTPUT_DIR / f"face_{index + 1}.png"
    Image.fromarray(scaled).save(output_file)

    passed += 1
    print(f"PASS: Image {index + 1}")

print("\n" + "=" * 45)
print("FINAL RESULTS")
print("=" * 45)

print("Total images:", len(images))
print("Readable samples:", passed, "/ 5")
print("Saved examples:", OUTPUT_DIR)

if len(images) == 13233 and passed == 5:
    print("\nSUCCESS: LFW initial validation passed!")
else:
    print("\nCHECK: Review dataset count or samples.")

print(
    "\nNote: This verifies dataset loading, "
    "not face-verification accuracy."
)
