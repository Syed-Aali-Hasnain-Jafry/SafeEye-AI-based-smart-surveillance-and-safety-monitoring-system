from pathlib import Path
import cv2

BASE_DIR = Path(__file__).resolve().parent.parent

dataset = (
    BASE_DIR / "datasets" / "MOT17"
    / "train" / "MOT17-02-FRCNN"
)

images_folder = dataset / "img1"
gt_file = dataset / "gt" / "gt.txt"
config_file = dataset / "seqinfo.ini"

print("Checking MOT17 dataset...")

if not images_folder.is_dir():
    raise FileNotFoundError(
        f"Image folder missing: {images_folder}"
    )

images = sorted(images_folder.glob("*.jpg"))

print("Images found:", len(images))
print("Ground truth exists:", gt_file.is_file())
print("Sequence metadata exists:", config_file.is_file())

if not images:
    raise RuntimeError("No JPG images found!")

first_image = cv2.imread(str(images[0]))

if first_image is None:
    raise RuntimeError("Cannot read first image!")

print("First image size:", first_image.shape)
print("MOT17 image reading successful!")