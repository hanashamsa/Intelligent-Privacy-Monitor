from dataclasses import dataclass
from typing import Tuple


@dataclass(slots=True)
class Screen:

    bbox: Tuple[int, int, int, int]

    center: Tuple[int, int]

    confidence: float

    screen_type: str