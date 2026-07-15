import time
import cv2


class AlertOverlay:

    def __init__(self):

        self.visible = False
        self.start_time = 0
        self.duration = 5

        self.message = ""
        self.score = 0.0

    def trigger(self, message, score):

        self.visible = True
        self.start_time = time.time()
        self.message = message
        self.score = score

    def draw(self, frame):

        if not self.visible:
            return frame

        if time.time() - self.start_time > self.duration:
            self.visible = False
            return frame

        h, w = frame.shape[:2]

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (40, 40),
            (w - 40, 180),
            (0, 0, 180),
            -1,
        )

        alpha = 0.65

        frame = cv2.addWeighted(
            overlay,
            alpha,
            frame,
            1 - alpha,
            0,
        )

        cv2.putText(
            frame,
            "PRIVACY ALERT",
            (70, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3,
        )

        cv2.putText(
            frame,
            self.message,
            (70, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Threat Score : {self.score:.2f}",
            (70, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )

        return frame