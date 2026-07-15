from dataclasses import dataclass
from typing import Tuple


@dataclass(slots=True)
class SceneObject:

    object_id: int

    label: str

    bbox: Tuple[int, int, int, int]

    confidence: float

    center: Tuple[int, int] = (0, 0)