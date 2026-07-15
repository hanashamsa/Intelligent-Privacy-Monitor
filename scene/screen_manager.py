from scene.screen import Screen


class ScreenManager:

    def __init__(self):

        self.screen = None

    def update(self, scene_objects):

        self.screen = None

        best_score = 0

        for obj in scene_objects:

            if obj.label not in ("Laptop", "Monitor"):
                continue

            area = (
                (obj.bbox[2] - obj.bbox[0]) *
                (obj.bbox[3] - obj.bbox[1])
            )

            score = area * obj.confidence

            if score > best_score:

                best_score = score

                self.screen = Screen(

                    bbox=obj.bbox,

                    center=obj.center,

                    confidence=obj.confidence,

                    screen_type=obj.label

                )

        return self.screen