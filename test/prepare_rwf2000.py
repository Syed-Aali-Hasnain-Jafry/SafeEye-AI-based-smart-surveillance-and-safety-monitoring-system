
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "RWF-2000"
OUTPUT = ROOT / "test" / "rwf2000_manifest.csv"

folders = {
    ("train", "Train_Fight"): 1,
    ("train", "Train_NonFight"): 0,
    ("val", "Val_Fight"): 1,
    ("val", "Val_NonFight"): 0,
}

rows = []

for (split, category), label in folders.items():
    folder = DATASET / split / category

    for video in sorted(folder.glob("*.avi")):
        if not video.is_file():
            continue

        rows.append({
            "path": str(video.relative_to(DATASET)),
            "split": split,
            "label": label,
        })

with OUTPUT.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file, fieldnames=["path", "split", "label"]
    )
    writer.writeheader()
    writer.writerows(rows)

print("Manifest saved:", OUTPUT)
print("Total usable videos:", len(rows))
print("Fighting videos:", sum(r["label"] == 1 for r in rows))
print("Non-fighting videos:", sum(r["label"] == 0 for r in rows))
