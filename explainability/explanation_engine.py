from dataclasses import dataclass


@dataclass(slots=True)
class ThreatExplanation:

    score: float

    level: str

    reasons: list[str]


class ExplanationEngine:

    def generate(self, person):

        reasons = []

        if person.label == "Unknown":
            reasons.append(
                "Unknown person detected (+0.25)"
            )

        if person.in_danger_zone:
            reasons.append(
                "Inside owner's danger zone (+0.20)"
            )

        if person.looking_at_screen:
            reasons.append(
                "Looking at laptop screen (+0.25)"
            )

        if person.danger_duration >= 2:

            reasons.append(
                f"Observed for {person.danger_duration:.1f}s (+0.15)"
            )

        if person.head_pose == "Looking Center":

            reasons.append(
                "Head oriented toward owner (+0.10)"
            )

        if person.distance_to_owner < 150:

            reasons.append(
                "Very close to owner (+0.05)"
            )

        return ThreatExplanation(

            score=person.threat_score,

            level=person.threat_level,

            reasons=reasons

        )