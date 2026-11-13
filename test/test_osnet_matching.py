
from pathlib import Path
import re
import cv2
import numpy as np
import torch
from torch.nn.functional import normalize
from torchreid.utils import FeatureExtractor

BASE = Path(__file__).resolve().parent.parent
ROOT = BASE / "datasets" / "Market-1501-v15.09.15"
WEIGHTS = next(
    (BASE / "models").glob("osnet_x1_0_msmt17*.pth")
)

query_dir = ROOT / "query"
gallery_dir = ROOT / "bounding_box_test"

pattern = re.compile(r"^(\d+)_c(\d+)")

def get_info(path):
    match = pattern.match(path.name)
    if match is None:
        return None
    return tuple(map(int, match.groups()))

# Pick one query with valid cross-camera matches
query = None
gallery = None

gallery_info = [
    (p, get_info(p))
    for p in sorted(gallery_dir.glob("*.jpg"))
]

for candidate in sorted(query_dir.glob("*.jpg")):
    info = get_info(candidate)
    if info is None:
        continue

    pid, cam = info
    positives = [
        p for p, g in gallery_info
        if g is not None and g[0] == pid and g[1] != cam
    ]

    if positives:
        query = candidate
        # One correct match + distractors
        distractors = [
            p for p, g in gallery_info
            if g is not None
            and g[0] > 0
            and g[0] != pid
            and g[1] != cam
        ][:49]

        gallery = positives[:1] + distractors
        break

if query is None or not gallery:
    raise RuntimeError("No valid matching experiment found")

device = "cuda" if torch.cuda.is_available() else "cpu"

extractor = FeatureExtractor(
    model_name="osnet_x1_0",
    model_path=str(WEIGHTS),
    device=device,
    verbose=False
)

paths = [query] + gallery

with torch.no_grad():
    features = extractor([str(p) for p in paths])
    features = normalize(features, dim=1)

scores = (
    features[1:] @ features[0]
).cpu().numpy()

ranking = np.argsort(-scores)

print("Query:", query.name)
print("\nTop 5 matches:")

for rank, index in enumerate(ranking[:5], 1):
    candidate = gallery[index]
    is_correct = get_info(candidate)[0] == get_info(query)[0]

    print(
        f"{rank}. {candidate.name} "
        f"Similarity: {scores[index]:.4f} "
        f"Correct identity: {is_correct}"
    )

# Save a visual comparison
def prepare(path):
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Cannot read: {path}")
    return cv2.resize(img, (128, 256))

images = [prepare(query)]
images += [prepare(gallery[i]) for i in ranking[:5]]

output = BASE / "runs" / "osnet"
output.mkdir(parents=True, exist_ok=True)

cv2.imwrite(
    str(output / "top5_matches.jpg"),
    np.hstack(images)
)

print("\nSaved to runs/osnet/top5_matches.jpg")
