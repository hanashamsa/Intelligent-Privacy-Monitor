from dataclasses import dataclass


@dataclass(slots=True)
class ThreatResult:

    score: float

    level: str


class ThreatEngine:

    def __init__(self):

        self.weights = {

            "unknown": 0.25,

            "danger_zone": 0.20,

            "screen": 0.25,

            "duration": 0.15,

            "head": 0.10,

            "distance": 0.05,

        }

    def evaluate(self, person):

        score = 0.0

        if person.label == "Unknown":

            score += self.weights["unknown"]

        if person.in_danger_zone:

            score += self.weights["danger_zone"]

        if person.looking_at_screen:

            score += self.weights["screen"]

        if person.danger_duration >= 2:

            score += self.weights["duration"]

        if person.head_pose == "Looking Center":

            score += self.weights["head"]

        if person.distance_to_owner < 150:

            score += self.weights["distance"]

        if score < 0.30:

            level = "SAFE"

        elif score < 0.60:

            level = "LOW"

        elif score < 0.80:

            level = "MEDIUM"

        else:

            level = "HIGH"

        return ThreatResult(score, level)