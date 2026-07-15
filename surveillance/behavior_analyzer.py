import time


class BehaviorAnalyzer:

    def __init__(self):

        self.history = {}

        self.threshold = 2.0

    def analyze(self, people):

        now = time.time()

        active_ids = set()

        for person in people:

            active_ids.add(person.tracker_id)

            if person.label != "Unknown":
                continue

            if not person.in_danger_zone:
                self.history.pop(person.tracker_id, None)
                person.danger_duration = 0.0
                person.is_suspicious = False
                continue

            if person.tracker_id not in self.history:

                self.history[person.tracker_id] = now

            duration = now - self.history[person.tracker_id]

            person.danger_duration = duration

            if (
                duration >= self.threshold
                and person.head_pose == "Looking Center"
            ):

                person.is_suspicious = True

            else:

                person.is_suspicious = False

        inactive = set(self.history.keys()) - active_ids

        for tracker_id in inactive:

            del self.history[tracker_id]