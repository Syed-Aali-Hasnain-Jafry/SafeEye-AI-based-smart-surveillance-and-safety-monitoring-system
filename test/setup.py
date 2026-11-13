import sys
import cv2
import torch
import ultralytics

print("Python:", sys.version)
print("OpenCV:", cv2.__version__)
print("PyTorch:", torch.__version__)
print("Ultralytics:", ultralytics.__version__)

print("CUDA GPU:", torch.cuda.is_available())

if hasattr(torch.backends, "mps"):
    print("Apple GPU:", torch.backends.mps.is_available())

print("SafEye setup successful!")