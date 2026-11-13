
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "WIDER_FACE" / "WIDER_val" / "images"
OUTPUT = ROOT / "runs" / "wider_face"

OUTPUT.mkdir(parents=True, exist_ok=True)

if not DATASET.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET}")

images = sorted(DATASET.rglob("*.jpg"))

print("=" * 45)
print("SAFEYE - WIDER FACE VALIDATION")
print("=" * 45)
print("Total images found:", len(images))

cascade_path = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    raise RuntimeError("Could not load face detector")

readable = 0
total_faces = 0

for image_path in images[:5]:
    image = cv2.imread(str(image_path))

    if image is None:
        print("FAIL:", image_path.name)
        continue

    readable += 1

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(20, 20),
    )

    total_faces += len(faces)

    for x, y, w, h in faces:
        cv2.rectangle(
            image, (x, y),
            (x + w, y + h),
            (0, 255, 0), 2
        )

    output_path = OUTPUT / image_path.name
    cv2.imwrite(str(output_path), image)

    print(
        f"PASS: {image_path.name} "
        f"| Faces detected: {len(faces)}"
    )

print("\n" + "=" * 45)
print("FINAL RESULTS")
print("=" * 45)
print("Total dataset images:", len(images))
print("Readable sample images:", readable, "/ 5")
print("Face detections:", total_faces)
print("Results saved to:", OUTPUT)
print("Note: Detections are not accuracy measurements.")
