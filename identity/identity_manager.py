import time


class IdentityManager:

    def __init__(self):

        self.identities = {}

        self.recognition_interval = 15

    def should_recognize(self, tracker_id):

        if tracker_id not in self.identities:
            return True

        frame_count = self.identities[tracker_id]["frame_count"]

        return frame_count >= self.recognition_interval

    def update(self, tracker_id, label, confidence):

        self.identities[tracker_id] = {

            "label": label,

            "confidence": confidence,

            "frame_count": 0,

            "last_seen": time.time()

        }

    def tick(self):

        remove = []

        now = time.time()

        for tracker_id in self.identities:

            self.identities[tracker_id]["frame_count"] += 1

            self.identities[tracker_id]["last_seen"] = now

        for tracker_id in list(self.identities):

            if now - self.identities[tracker_id]["last_seen"] > 10:

                remove.append(tracker_id)

        for tracker_id in remove:

            del self.identities[tracker_id]

    def get_label(self, tracker_id):

        if tracker_id not in self.identities:

            return "Recognizing...", 0

        item = self.identities[tracker_id]

        return item["label"], item["confidence"]