import math


class ScreenIntersection:

    def evaluate(self, person, screen):

        if screen is None:

            person.gaze_target = "None"

            person.looking_at_screen = False

            return

        px, py = person.center

        sx, sy = screen.center

        dx = sx - px

        dy = sy - py

        distance = math.sqrt(dx * dx + dy * dy)

        person.screen_distance = distance

        if (

            abs(person.gaze_yaw) < 12 and

            abs(person.gaze_pitch) < 10 and

            distance < 350

        ):

            person.looking_at_screen = True

            person.gaze_target = screen.screen_type

        else:

            person.looking_at_screen = False

            person.gaze_target = "Elsewhere"