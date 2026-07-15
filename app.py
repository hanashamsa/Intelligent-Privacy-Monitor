import cv2
import supervision as sv
from numpy.typing import NDArray

from config import (
    CAMERA_INDEX,
    WINDOW_NAME,
    FACE_RECOGNITION_INTERVAL,
    PERSON_CLASS,
)
from surveillance.spatial_analyzer import SpatialAnalyzer
from detector.detector import SceneDetector
from tracker.tracker import PersonTracker
from face.recognizer import FaceRecognizer
from identity.identity_manager import IdentityManager
from utils.fps import FPSCounter
from surveillance.behavior_analyzer import BehaviorAnalyzer
from surveillance.threat_engine import ThreatEngine
from event.incident_manager import IncidentManager
from recording.video_buffer import VideoBuffer
from recording.incident_recorder import IncidentRecorder
from pose.head_pose import HeadPoseEstimator
from gaze.gaze_estimator import GazeEstimator
from identity.person import Person
from scene.scene_analyzer import SceneAnalyzer
from scene.screen_manager import ScreenManager
from scene.screen_intersection import ScreenIntersection
from alert.overlay import AlertOverlay
from alert.sound import AlertSound
from ui.dashboard import Dashboard


class PrivacyGuard:

    def __init__(self):

        self.detector = SceneDetector()

        self.tracker = PersonTracker()

        self.recognizer = FaceRecognizer()

        self.behavior = BehaviorAnalyzer()

        self.threat_engine = ThreatEngine()

        self.alert_overlay = AlertOverlay()

        self.alert_sound = AlertSound()

        self.dashboard = Dashboard()

        self.video_buffer = VideoBuffer(
            fps=30,
            seconds=10,
            frame_size=(640, 480),
        )

        self.incident_manager = IncidentManager(self.video_buffer)

        self.recorder = IncidentRecorder()

        self.spatial = SpatialAnalyzer()

        self.scene = SceneAnalyzer()

        self.screen_manager = ScreenManager()

        self.screen_intersection = ScreenIntersection()

        self.identity_manager = IdentityManager(
            self.recognizer,
            recognition_interval=FACE_RECOGNITION_INTERVAL
        )

        self.pose_estimator = HeadPoseEstimator()

        self.gaze_estimator = GazeEstimator()

        self.fps_counter = FPSCounter()

        self.box_annotator = sv.BoxAnnotator()

        

        self.cap = cv2.VideoCapture(CAMERA_INDEX)

        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera.")

    def run(self):

        while True:

            success, frame = self.cap.read()

            if not success:
                break

            self.video_buffer.add_frame(frame)

            detections = self.detector.detect(frame)
            scene_objects = self.scene.extract_objects(detections)
            screen = self.screen_manager.update(scene_objects)
            detections = self.tracker.update(detections)

            active_ids = []
            people = []

        # -------------------------------------------------
        # PASS 1 : Detection + Tracking + Recognition
        # -------------------------------------------------

            if (
                detections.tracker_id is not None
                and len(detections.tracker_id) > 0
            ):

                for i in range(len(detections.xyxy)):

                    if int(detections.class_id[i]) != PERSON_CLASS:
                        continue

                    tracker_id = int(detections.tracker_id[i])
                    active_ids.append(tracker_id)

                    confidence = float(detections.confidence[i])

                    x1, y1, x2, y2 = detections.xyxy[i].astype(int)

                    h, w = frame.shape[:2]

                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop = frame[y1:y2, x1:x2]

                    person = self.identity_manager.process(
                        tracker_id=tracker_id,
                        crop=crop,
                        bbox=(x1, y1, x2, y2),
                        confidence=confidence,
                    )

                    people.append(person)

            # Head pose and gaze must run before behavior analysis: behavior
            # consumes the most recent per-person face observations.
            for person in people:
                self._update_face_observations(frame, person)

            self.spatial.analyze(people)

            self.behavior.analyze(people)

            for person in people:
                self.screen_intersection.evaluate(person, screen)

            high = False
            for person in people:
                threat = self.threat_engine.evaluate(person)
                person.threat_score = threat.score
                person.threat_level = threat.level

                if person.threat_level == "HIGH":
                    high = True
                    self.alert_overlay.trigger(
                        "Unknown person looking at your screen",
                        person.threat_score,
                    )

            if high:
                self.alert_sound.play()
            else:
                self.alert_sound.reset()

            state = self.recorder.update(high)

            for person in people:
                self.incident_manager.process(person, frame)

        # -------------------------------------------------
        # PASS 3 : Draw Everything
        # -------------------------------------------------

            owner = None

            for p in people:

                if p.label == "Owner":
                    owner = p
                    break


            if owner is not None:

                ox, oy = owner.center

                cv2.circle(
                    frame,
                    (ox, oy),
                    250,
                    (255, 0, 255),
                    2,
                )

            for obj in scene_objects:

                if obj.label == "Person":
                    continue

                color = (255, 0, 255)

                if obj.label == "Laptop":
                    color = (255, 255, 0)

                elif obj.label == "Phone":
                    color = (0, 255, 255)

                elif obj.label == "Monitor":
                    color = (255, 0, 0)

                x1, y1, x2, y2 = obj.bbox

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2,
                )

                cv2.putText(
                    frame,
                    obj.label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            if screen is not None:

                x1, y1, x2, y2 = screen.bbox

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 255),
                    3,
                )

                cv2.putText(
                    frame,
                    f"SCREEN ({screen.screen_type})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 255),
                    2,
                )

            for person in people:

                x1, y1, x2, y2 = person.bbox

                if person.threat_level == "SAFE":
                    color = (0, 255, 0)

                elif person.threat_level == "LOW":
                    color = (0, 255, 255)

                elif person.threat_level == "MEDIUM":
                    color = (0, 165, 255)

                else:
                    color = (0, 0, 255)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2,
                )

                face_bbox = self._absolute_face_bbox(frame, person)
                if face_bbox is not None:
                    fx1, fy1, fx2, fy2 = face_bbox
                    cv2.rectangle(
                        frame,
                        (fx1, fy1),
                        (fx2, fy2),
                        (255, 255, 0),
                        2,
                    )

                self._draw_gaze_overlay(frame, person, color)

            self.identity_manager.cleanup(active_ids)

            fps = self.fps_counter.update()

            frame = self.alert_overlay.draw(frame)

            frame = self.dashboard.render(
                frame=frame,
                fps=fps,
                people=people,
                recorder_state=self.recorder.state.name,
                incident_count=self.incident_manager.incident_count,
            )

            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()

        cv2.destroyAllWindows()

    def _update_face_observations(
        self, frame: NDArray, person: Person
    ) -> None:
        """Populate pose and gaze for a person when a valid face crop exists."""
        face_bbox = self._absolute_face_bbox(frame, person)
        if face_bbox is None:
            person.head_pose = "Unknown"
            person.gaze_yaw = 0.0
            person.gaze_pitch = 0.0
            person.gaze_direction = "Unknown"
            person.looking_at_screen = False
            return

        fx1, fy1, fx2, fy2 = face_bbox
        face_crop = frame[fy1:fy2, fx1:fx2]
        if face_crop.size == 0:
            person.head_pose = "Unknown"
            person.gaze_direction = "Unknown"
            person.looking_at_screen = False
            return

        person.head_pose = self.pose_estimator.estimate(face_crop)
        gaze = self.gaze_estimator.estimate(face_crop)
        person.gaze_yaw = gaze.yaw
        person.gaze_pitch = gaze.pitch
        person.gaze_direction = gaze.direction
        person.looking_at_screen = gaze.looking_at_screen

    @staticmethod
    def _absolute_face_bbox(
        frame: NDArray, person: Person
    ) -> tuple[int, int, int, int] | None:
        """Convert a person-relative face box to a clipped frame-relative box."""
        if person.face_bbox is None:
            return None

        x1, y1, x2, y2 = person.bbox
        fx1, fy1, fx2, fy2 = person.face_bbox
        height, width = frame.shape[:2]
        fx1 = max(x1, min(x1 + int(fx1), min(x2, width)))
        fy1 = max(y1, min(y1 + int(fy1), min(y2, height)))
        fx2 = max(x1, min(x1 + int(fx2), min(x2, width)))
        fy2 = max(y1, min(y1 + int(fy2), min(y2, height)))
        if fx2 <= fx1 or fy2 <= fy1:
            return None
        return fx1, fy1, fx2, fy2

    @staticmethod
    def _draw_gaze_overlay(frame: NDArray, person: Person, color: tuple[int, int, int]) -> None:
        """Draw the identity, pose, and L2CS-Net gaze estimate beside a person."""
        x1, y1, _, _ = person.bbox
        eyes = person.gaze_direction.removeprefix("Looking ")
        text = (
            f"{person.label} | "
            f"{person.gaze_target} | "
            f"{person.threat_level} | "
            f"{person.threat_score:.2f}"
        )
        lines = (
            text,
            f"Head : {person.head_pose}",
            f"Eyes : {eyes}",
            f"Yaw : {person.gaze_yaw:.1f}\N{DEGREE SIGN}",
            f"Pitch : {person.gaze_pitch:.1f}\N{DEGREE SIGN}",
        )
        line_height = 22
        baseline = max(line_height, y1 - line_height * len(lines))
        for index, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (x1, baseline + index * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )


if __name__ == "__main__":

    app = PrivacyGuard()

    app.run()
