from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Person:

    tracker_id: int

    bbox: Tuple[int, int, int, int]

    confidence: float

    label: str = "Recognizing..."

    similarity: float = 0.0

    recognized: bool = False

    frames_since_recognition: int = 0

    face_bbox: Optional[Tuple[int, int, int, int]] = None

    head_pose: str = "Unknown"

    center: Tuple[int, int] = (0, 0)

    distance_to_owner: float = 0.0

    relative_position: str = "Unknown"

    in_danger_zone: bool = False

    # BehaviorAnalyzer populates this for unknown people; a default is required
    # because ThreatEngine evaluates every tracked person, including the owner.
    danger_duration: float = 0.0

    gaze_yaw: float = 0.0

    gaze_pitch: float = 0.0

    gaze_direction: str = "Unknown"

    looking_at_screen: bool = False

    threat_score: float = 0.0

    threat_level: str = "SAFE"


    gaze_target: str = "Unknown"

    screen_distance: float = 0.0
