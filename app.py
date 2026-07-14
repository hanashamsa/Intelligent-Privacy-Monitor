import cv2
import supervision as sv

from detector.detector import PersonDetector
from tracker.tracker import PersonTracker
from utils.fps import FPSCounter

from config import (
    CAMERA_INDEX,
    WINDOW_NAME
)


def main():

    detector = PersonDetector()

    tracker = PersonTracker()

    fps_counter = FPSCounter()

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera.")

    box_annotator = sv.BoxAnnotator()

    label_annotator = sv.LabelAnnotator()

    while True:

        success, frame = cap.read()

        if not success:
            break

        detections = detector.detect(frame)

        detections = tracker.update(detections)

        labels = []

        if detections.tracker_id is not None:

            for confidence, tracker_id in zip(
                detections.confidence,
                detections.tracker_id
            ):

                labels.append(
                    f"Person #{tracker_id} ({confidence:.2f})"
                )

        frame = box_annotator.annotate(
            scene=frame,
            detections=detections
        )

        frame = label_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )

        fps = fps_counter.update()

        cv2.putText(
            frame,
            f"FPS : {fps}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()