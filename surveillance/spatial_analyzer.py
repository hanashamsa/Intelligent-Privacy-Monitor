import math


class SpatialAnalyzer:

    def __init__(self):

        self.owner = None

    def _center(self, bbox):

        x1, y1, x2, y2 = bbox

        return (
            (x1 + x2) // 2,
            (y1 + y2) // 2
        )

    def analyze(self, people):

        owner = None

        for person in people:

            if person.label == "Owner":

                owner = person

                break

        if owner is None:

            return

        ox, oy = self._center(owner.bbox)

        owner.center = (ox, oy)

        for person in people:

            px, py = self._center(person.bbox)

            person.center = (px, py)

            dx = px - ox

            dy = py - oy

            distance = math.sqrt(dx ** 2 + dy ** 2)

            person.distance_to_owner = distance

            if person.label == "Owner":

                person.relative_position = "Owner"

                continue

            if abs(dx) < 120:

                if dy < 0:

                    person.relative_position = "Behind"

                else:

                    person.relative_position = "Front"

            elif dx < 0:

                person.relative_position = "Left"

            else:

                person.relative_position = "Right"

            person.in_danger_zone = distance < 250