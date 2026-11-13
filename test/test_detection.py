from ultralytics import YOLO

# Automatically downloads pretrained weights
model = YOLO("yolo26n.pt")

# Public example image
image_url = "https://ultralytics.com/images/bus.jpg"

# COCO class 0 = person
results = model.predict(
    source=image_url,
    classes=[0],
    conf=0.30,
    save=True,
    project="runs/safeye",
    name="detection_test"
)

for result in results:
    print("People detected:", len(result.boxes))

print("Detection test completed!")