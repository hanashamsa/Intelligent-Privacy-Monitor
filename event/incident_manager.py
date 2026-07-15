from __future__ import annotations

import os
import time
import cv2


class IncidentManager:

    def __init__(self, video_buffer):

        self.video_buffer = video_buffer

        self.cooldown = 15

        self.last_incident = {}

        self.incident_count = 0

        os.makedirs(
            "outputs/screenshots",
            exist_ok=True
        )

        os.makedirs(
            "outputs/videos",
            exist_ok=True
        )

    def process(self, person, frame):

        if person.threat_level != "HIGH":

            return False

        now = time.time()

        previous = self.last_incident.get(
            person.tracker_id,
            0
        )

        if now - previous < self.cooldown:

            return False

        filename = (
            f"outputs/screenshots/"
            f"{int(now)}_"
            f"{person.tracker_id}.jpg"
        )

        cv2.imwrite(
            filename,
            frame
        )

        video_name = (
            f"outputs/videos/"
            f"{int(now)}_"
            f"{person.tracker_id}.mp4"
        )

        self.video_buffer.save(video_name)

        self.last_incident[
            person.tracker_id
        ] = now

        self.incident_count += 1

        print(
            f"[INCIDENT] Screenshot saved -> {filename}"
        )

        return True
