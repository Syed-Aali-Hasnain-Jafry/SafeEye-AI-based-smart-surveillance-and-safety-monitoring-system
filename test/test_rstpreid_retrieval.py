
from pathlib import Path
import json
import random
import torch
from PIL import Image, ImageDraw
from transformers import CLIPModel, CLIPProcessor

BASE = Path(__file__).resolve().parent.parent
ROOT = BASE / "datasets" / "RSTPReid"

with open(ROOT / "data_captions.json", encoding="utf-8") as f:
    records = json.load(f)

# Use only held-out test examples
records = [r for r in records if r["split"] == "test"]

# Find image files, regardless of the nested folder layout
image_index = {
    p.name: p
    for p in ROOT.rglob("*")
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
}

def get_image_path(record):
    relative = Path(record["img_path"])
    direct = ROOT / relative

    if direct.is_file():
        return direct

    return image_index.get(relative.name)

# Keep records that have readable image paths
records = [r for r in records if get_image_path(r) is not None]

# Choose a query with another image of the same identity
query = None
positive = None

for record in records:
    matches = [
        r for r in records
        if r["id"] == record["id"]
        and r["img_path"] != record["img_path"]
    ]
    if matches:
        query = record
        positive = matches[0]
        break

if query is None:
    raise RuntimeError("No positive identity pair found")

random.seed(42)

negatives = [
    r for r in records
    if r["id"] != query["id"]
]

negatives = random.sample(
    negatives, min(49, len(negatives))
)

candidates = [positive] + negatives
random.shuffle(candidates)

description = query["captions"][0]

print("\nQuery description:")
print(description)
print("\nCandidate images:", len(candidates))

model_id = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id)
model.eval()

scores = []

# Batch processing reduces memory use
with torch.inference_mode():
    for start in range(0, len(candidates), 8):
        batch = candidates[start:start + 8]

        images = []
        for record in batch:
            path = get_image_path(record)
            with Image.open(path) as im:
                images.append(im.convert("RGB"))

        inputs = processor(
            text=[description],
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        outputs = model(**inputs)
        scores.extend(
            outputs.logits_per_text[0].cpu().tolist()
        )

ranked = sorted(
    zip(candidates, scores),
    key=lambda x: x[1],
    reverse=True
)

print("\nTop 5 matches:")

for rank, (record, score) in enumerate(ranked[:5], 1):
    correct = record["id"] == query["id"]
    print(
        f"{rank}. {record['img_path']} | "
        f"CLIP score: {score:.3f} | "
        f"Correct identity: {correct}"
    )

# Save a visual comparison of the top five
output_dir = BASE / "runs" / "rstpreid"
output_dir.mkdir(parents=True, exist_ok=True)

canvas = Image.new("RGB", (150 * 5, 310), "white")
draw = ImageDraw.Draw(canvas)

for i, (record, score) in enumerate(ranked[:5]):
    path = get_image_path(record)

    with Image.open(path) as im:
        picture = im.convert("RGB").resize((140, 260))

    x = i * 150
    canvas.paste(picture, (x, 0))

    correct = record["id"] == query["id"]
    label = "MATCH" if correct else "OTHER"

    draw.text((x + 4, 268), label, fill="black")
    draw.text((x + 4, 286), f"Score: {score:.2f}", fill="black")

save_path = output_dir / "top5_text_matches.jpg"
canvas.save(save_path)

print("\nSaved:", save_path)
print("RSTPReid CLIP retrieval test completed!")
