
import torch
from transformers import CLIPModel, CLIPProcessor

print("Loading CLIP...")

model_id = "openai/clip-vit-base-patch32"

processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id)
model.eval()

print("PyTorch:", torch.__version__)
print("CLIP model:", model_id)
print("CLIP successfully loaded!")
