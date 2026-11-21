
from datetime import datetime, timezone
from pathlib import Path

from ultralytics import YOLO


class PersonTracker:
    """
    Detect and track people in consecutive video frames.

    Each instance maintains tracking state for one
    camera and one processing session.
    """

    def __init__(
        self,
        model_path,
        camera_id="camera_01",
        confidence=0.25,
    ):
        model_path = Path(model_path)

        if not model_path.is_file():
            raise FileNotFoundError(
                f"YOLO model not found: {model_path}"
            )

        self.model = YOLO(str(model_path))
        self.camera_id = camera_id
        self.confidence = confidence
        self.frame_number = 0

    def process_frame(self, frame):
        """
        Process one frame and return structured
        person-tracking information.
        """

        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty.")

        self.frame_number += 1

        results = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=self.confidence,
            verbose=False,
        )

        result = results[0]
        detections = []

        boxes = result.boxes

        if boxes is not None and boxes.id is not None:
            track_ids = boxes.id.int().cpu().tolist()
            coordinates = boxes.xyxy.cpu().tolist()
            confidences = boxes.conf.cpu().tolist()

            for track_id, bbox, score in zip(
                track_ids,
                coordinates,
                confidences,
            ):
                detections.append(
                    {
                        "track_id": track_id,
                        "bbox": [
                            round(value, 2)
                            for value in bbox
                        ],
                        "confidence": round(score, 4),
                    }
                )

        tracking_data = {
            "camera_id": self.camera_id,
            "frame_number": self.frame_number,
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "tracks": detections,
        }

        return tracking_data
