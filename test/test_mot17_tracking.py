from pathlib import Path
import cv2
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent.parent

image_dir = (
    BASE / "datasets" / "MOT17"
    / "train" / "MOT17-02-FRCNN" / "img1"
)

images = sorted(image_dir.glob("*.jpg"))[:100]

if not images:
    raise FileNotFoundError("MOT17 images not found")

model = YOLO(str(BASE / "yolo26n.pt"))

output_dir = BASE / "runs" / "mot17"
output_dir.mkdir(parents=True, exist_ok=True)

writer = None

try:
    for frame_no, image_path in enumerate(images, 1):

        frame = cv2.imread(str(image_path))

        if frame is None:
            print("Skipping unreadable image:", image_path)
            continue

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            verbose=False
        )

        result = results[0]
        annotated = result.plot()

        if writer is None:
            h, w = annotated.shape[:2]

            writer = cv2.VideoWriter(
                str(output_dir / "tracking.avi"),
                cv2.VideoWriter_fourcc(*"MJPG"),
                10,
                (w, h)
            )

            if not writer.isOpened():
                raise RuntimeError("Video writer failed")

        writer.write(annotated)

        ids = (
            result.boxes.id.int().tolist()
            if result.boxes.id is not None
            else []
        )

        if frame_no % 10 == 0:
            print(f"Frame {frame_no}: IDs {ids}")

finally:
    if writer is not None:
        writer.release()

print("MOT17 tracking test completed!")