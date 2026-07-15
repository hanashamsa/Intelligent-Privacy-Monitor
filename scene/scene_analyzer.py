from scene.scene_object import SceneObject


class SceneAnalyzer:

    CLASS_NAMES = {

        0: "Person",

        62: "Monitor",

        63: "Laptop",

        67: "Phone",

    }

    def extract_objects(self, detections):

        objects = []

        if len(detections) == 0:
            return objects

        for i in range(len(detections.xyxy)):

            x1, y1, x2, y2 = map(
                int,
                detections.xyxy[i]
            )

            cls = int(detections.class_id[i])

            conf = float(detections.confidence[i])

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            objects.append(

                SceneObject(

                    object_id=i,

                    label=self.CLASS_NAMES.get(
                        cls,
                        "Unknown"
                    ),

                    bbox=(x1, y1, x2, y2),

                    confidence=conf,

                    center=(cx, cy),

                )

            )

        return objects