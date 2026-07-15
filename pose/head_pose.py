import cv2
import mediapipe as mp
import math


class HeadPoseEstimator:

    def __init__(self):

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def estimate(self, face):

        if face is None:
            return "Unknown"

        rgb = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )

        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return "Unknown"

        landmarks = results.multi_face_landmarks[0].landmark

        nose = landmarks[1]

        left = landmarks[234]

        right = landmarks[454]

        dx = right.x - left.x

        offset = nose.x - ((left.x + right.x) / 2)

        ratio = offset / dx

        if ratio < -0.08:
            return "Looking Left"

        if ratio > 0.08:
            return "Looking Right"

        return "Looking Center"