
from pathlib import Path
import cv2

BASE = Path(__file__).resolve().parent.parent
ROOT = BASE / "datasets" / "hmdb51_org"

videos = sorted(ROOT.rglob("*.avi"))

print("===== HMDB51 DATASET CHECK =====")
print("Videos found:", len(videos))

if not videos:
    raise FileNotFoundError(
        f"No AVI videos found inside {ROOT}"
    )

readable = 0
failed = 0

for path in videos:
    cap = cv2.VideoCapture(str(path))
    success, frame = cap.read()
    cap.release()

    if success:
        readable += 1
    else:
        failed += 1
        print("Unreadable:", path.name)

print("Readable videos:", readable)
print("Failed videos:", failed)

sample = videos[0]
cap = cv2.VideoCapture(str(sample))

print("Sample file:", sample.name)
print("Frame count:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
print("FPS:", cap.get(cv2.CAP_PROP_FPS))

cap.release()

if failed:
    raise RuntimeError("Some dataset videos could not be read")

print("HMDB51 video validation successful!")