from ultralytics import YOLO
import supervision as sv
import torch

from config import (
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    IMAGE_SIZE,
    TRACK_CLASSES,
    USE_GPU,
)


class SceneDetector:

    def __init__(self):

        self.device = (
            "cuda"
            if torch.cuda.is_available() and USE_GPU
            else "cpu"
        )

        self.model = YOLO(MODEL_PATH)
        self.model.to(self.device)

    def detect(self, frame):

        result = self.model.predict(
            frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )[0]

        detections = sv.Detections.from_ultralytics(result)

        mask = []

        for cls in detections.class_id:
            mask.append(int(cls) in TRACK_CLASSES)

        detections = detections[mask]

        return detections