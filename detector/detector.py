from ultralytics import YOLO
import supervision as sv
import torch

from config import (
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    IMAGE_SIZE,
    PERSON_CLASS,
    USE_GPU
)


class PersonDetector:

    def __init__(self):

        self.device = (
            "cuda"
            if torch.cuda.is_available() and USE_GPU
            else "cpu"
        )

        print(f"Running on {self.device}")

        self.model = YOLO(MODEL_PATH)
        self.model.to(self.device)

    def detect(self, frame):

        results = self.model.predict(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            imgsz=IMAGE_SIZE,
            verbose=False
        )[0]

        detections = sv.Detections.from_ultralytics(results)

        detections = detections[detections.class_id == PERSON_CLASS]

        return detections 