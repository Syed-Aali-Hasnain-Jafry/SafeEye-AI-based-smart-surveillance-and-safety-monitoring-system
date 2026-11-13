from pathlib import Path
from ultralytics import YOLO

video_path = Path("datasets/test_video.mp4")

if not video_path.exists():
    raise FileNotFoundError(
        "Put a short test_video.mp4 inside datasets/"
    )

model = YOLO("yolo26n.pt")

results = model.track(
    source=str(video_path),
    tracker="bytetrack.yaml",
    classes=[0],
    conf=0.25,
    save=True,
    project="runs/safeye",
    name="tracking_test",
    stream=True
)

for frame_no, result in enumerate(results, 1):
    if result.boxes.id is not None:
        ids = result.boxes.id.int().tolist()
        print(f"Frame {frame_no}: Track IDs {ids}")
    else:
        print(f"Frame {frame_no}: No active tracks")