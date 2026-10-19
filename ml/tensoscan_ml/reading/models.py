"""Value types of the parser (`reading-rules.md` section 4.1 and 4.4)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


@dataclass(frozen=True)
class Box:
    """Rectangle normalised to the ROI (`[0, 1]`, origin at the top left)."""

    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2.0

    @classmethod
    def from_sequence(cls, values: Iterable[float]) -> Box:
        left, top, right, bottom = (float(v) for v in values)
        return cls(left, top, right, bottom)


@dataclass(frozen=True)
class SymbolDetection:
    """One detected glyph: `symbol` is a class label of the model contract (`0`..`9`, `colon`, `dash`, `E`, `r`)."""

    symbol: str
    box: Box
    confidence: float

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> SymbolDetection:
        return cls(str(raw["symbol"]), Box.from_sequence(raw["box"]), float(raw["confidence"]))

    @property
    def digit(self) -> int | None:
        return int(self.symbol) if len(self.symbol) == 1 and self.symbol.isdigit() else None


class FrameStatus(StrEnum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    NO_READING = "NO_READING"
    MEASURING = "MEASURING"
    DEVICE_ERROR = "DEVICE_ERROR"


@dataclass(frozen=True)
class FieldValue:
    value: int
    confidence: float
    box: Box


@dataclass(frozen=True)
class FrameReading:
    """What one frame says about the three fields; a field is `None` when the frame did not show it."""

    status: FrameStatus
    systolic: FieldValue | None = None
    diastolic: FieldValue | None = None
    pulse: FieldValue | None = None
    #: Relative height of the tallest number on screen (a hint to move closer), or `None` if there were none.
    primary_height: float | None = None

    @property
    def values(self) -> tuple[int | None, int | None, int | None]:
        return (
            self.systolic.value if self.systolic else None,
            self.diastolic.value if self.diastolic else None,
            self.pulse.value if self.pulse else None,
        )
