
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image
from sklearn.datasets import fetch_lfw_people


# ----------------------------------------
# 1. Project paths
# ----------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "datasets" / "LFW"
OUTPUT_DIR = ROOT / "runs" / "lfw"

DATASET_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 45)
print("SAFEYE - LFW DATASET VALIDATION")
print("=" * 45)

print("\nDownloading/loading LFW dataset...")


# ----------------------------------------
# 2. Load dataset
# ----------------------------------------
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
multiple_images = sum(
    count >= 2 for count in counts.values()
)

print("Identities with 2+ images:", multiple_images)


# ----------------------------------------
# 3. Validate and save sample images
# ----------------------------------------
print("\n--- IMAGE READABILITY TEST ---")

passed = 0
sample_count = min(5, len(images))

for index in range(sample_count):
    image = images[index]

    if image.size == 0 or not np.isfinite(image).all():
        print(f"FAIL: Image {index + 1}")
        continue

    # Check original pixel values
    original_min = float(image.min())
    original_max = float(image.max())

    # LFW may provide images with values from 0 to 1.
    # Convert normalized values to the 0-255 range
    # before saving as an 8-bit PNG.
    if original_min >= 0 and original_max <= 1.0:
        scaled = np.clip(
            image * 255.0, 0, 255
        ).astype(np.uint8)
    else:
        scaled = np.clip(
            image, 0, 255
        ).astype(np.uint8)

    output_file = OUTPUT_DIR / f"face_{index + 1}.png"
    Image.fromarray(scaled).save(output_file)

    # Verify the saved image
    saved_image = np.asarray(Image.open(output_file))

    if saved_image.max() == 0:
        print(
            f"FAIL: Image {index + 1} "
            "- saved image is completely black"
        )
        continue

    passed += 1

    print(
        f"PASS: Image {index + 1} | "
        f"Original range: {original_min:.4f} "
        f"to {original_max:.4f} | "
        f"Saved range: {saved_image.min()} "
        f"to {saved_image.max()}"
    )


# ----------------------------------------
# 4. Final validation results
# ----------------------------------------
print("\n" + "=" * 45)
print("FINAL RESULTS")
print("=" * 45)

print("Total images:", len(images))
print("Readable and non-black saved samples:", passed, "/", sample_count)
print("Saved examples:", OUTPUT_DIR)

if len(images) == 13233 and passed == sample_count:
    print("\nSUCCESS: LFW initial validation passed!")
else:
    print("\nCHECK: Review dataset count or saved images.")

print(
    "\nNote: This verifies dataset loading and "
    "sample image saving, not face-verification accuracy."
)
