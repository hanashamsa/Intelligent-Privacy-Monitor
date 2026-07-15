from identity.person import Person


class IdentityManager:

    def __init__(self, recognizer, recognition_interval=15):

        self.recognizer = recognizer

        self.people = {}

        self.recognition_interval = recognition_interval

    def process(self, tracker_id, crop, bbox, confidence):

        if tracker_id not in self.people:

            person = Person(
                tracker_id=tracker_id,
                bbox=bbox,
                confidence=confidence
            )

            self.people[tracker_id] = person

        person = self.people[tracker_id]

        person.bbox = bbox
        person.confidence = confidence

        person.frames_since_recognition += 1

        if (
            not person.recognized
            or person.frames_since_recognition >= self.recognition_interval
        ):

            label, similarity, face_bbox = self.recognizer.recognize(crop)

            if label != "No Face":

                person.label = label

                person.similarity = similarity

                person.face_bbox = face_bbox

                person.recognized = True

                person.frames_since_recognition = 0

        return person

    def cleanup(self, active_ids):

        remove = []

        for tracker_id in self.people:

            if tracker_id not in active_ids:

                remove.append(tracker_id)

        for tracker_id in remove:

            del self.people[tracker_id]