"""Shared records. Boxes use [x1, y1, x2, y2] in processed-frame pixels."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Detection:
    label: str
    bbox: tuple[float, float, float, float]
    confidence: float
    subtype: str = ""


@dataclass
class Track:
    track_id: int
    label: str
    bbox: tuple[float, float, float, float]
    confidence: float
    first_seen: float
    last_seen: float
    hits: int = 1
    confirmed: bool = False
    observed: bool = True
    subtype: str = ""


@dataclass
class Event:
    timestamp: float
    event: str
    confidence: float
    confirmed_at: float
    track_id: int | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
