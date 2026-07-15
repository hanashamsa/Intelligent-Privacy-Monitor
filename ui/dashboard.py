import cv2
import numpy as np


class Dashboard:

    def __init__(self, width=320):

        self.width = width

    def render(
        self,
        frame,
        fps,
        people,
        recorder_state,
        incident_count,
    ):

        h = frame.shape[0]

        panel = np.full(
            (h, self.width, 3),
            35,
            dtype=np.uint8,
        )

        owner_present = any(
            p.label == "Owner"
            for p in people
        )

        unknown_count = sum(
            p.label == "Unknown"
            for p in people
        )

        highest = max(
            people,
            key=lambda p: p.threat_score,
            default=None,
        )

        score = 0.0
        level = "SAFE"

        if highest is not None:
            score = highest.threat_score
            level = highest.threat_level

        lines = [

            "PRIVACYGUARD AI",

            "",

            f"FPS               : {fps:.1f}",

            "",

            f"Owner             : {'YES' if owner_present else 'NO'}",

            f"Unknown People    : {unknown_count}",

            "",

            f"Threat Level      : {level}",

            f"Threat Score      : {score:.2f}",

            "",

            f"Recorder          : {recorder_state}",

            f"Incidents         : {incident_count}",

        ]

        y = 40

        for line in lines:

            cv2.putText(

                panel,

                line,

                (15, y),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (255, 255, 255),

                2,

            )

            y += 30

        return np.hstack([frame, panel])