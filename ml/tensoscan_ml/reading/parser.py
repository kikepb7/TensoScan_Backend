"""Display layout parser: from glyph detections to SYS/DIA/PUL (`reading-rules.md` section 4.3).

Deterministic and free of "magic corrections": it never swaps SYS and DIA nor completes digits. The structure follows
the Kotlin reference (`DisplayLayoutParser` / `NumberGrouper`) step by step so both implementations agree.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from tensoscan_ml.reading.models import Box, FieldValue, FrameReading, FrameStatus, SymbolDetection

# Constants of section 4.2. Changing one means a new version of the shared contract.
MIN_SYMBOL_CONFIDENCE: Final = 0.40
MIN_SYMBOL_HEIGHT: Final = 0.04
GROUP_MIN_VERTICAL_OVERLAP: Final = 0.60
GROUP_MAX_HEIGHT_RATIO: Final = 1.35
GROUP_MAX_GAP_FACTOR: Final = 0.35
GROUP_MAX_OVERLAP_FACTOR: Final = 0.45
ONE_CELL_FACTOR: Final = 0.55
MIN_PUNCTUATION_HEIGHT: Final = 0.01
DEFAULT_ROI_ASPECT: Final = 0.8
ROW_MIN_VERTICAL_OVERLAP: Final = 0.50
PRIMARY_MIN_HEIGHT_RATIO: Final = 0.75
SECONDARY_MIN_HEIGHT_RATIO: Final = 0.35

SPLIT_GAP_RATIO: Final = 2.0

MIN_NUMBER_DIGITS: Final = 2
MAX_NUMBER_DIGITS: Final = 3
MIN_DASHES: Final = 2


@dataclass(frozen=True)
class _Number:
    """A two or three digit value: its box is the envelope of its glyphs and its confidence their mean."""

    detections: tuple[SymbolDetection, ...]

    @property
    def value(self) -> int:
        total = 0
        for detection in self.detections:
            digit = detection.digit
            if digit is None:  # pragma: no cover - only digit groups are classified as numbers
                raise ValueError("a number holds digits only")
            total = total * 10 + digit
        return total

    @property
    def box(self) -> Box:
        return Box(
            left=min(d.box.left for d in self.detections),
            top=min(d.box.top for d in self.detections),
            right=max(d.box.right for d in self.detections),
            bottom=max(d.box.bottom for d in self.detections),
        )

    @property
    def confidence(self) -> float:
        return sum(d.confidence for d in self.detections) / len(self.detections)

    def to_field(self) -> FieldValue:
        return FieldValue(self.value, self.confidence, self.box)


def vertical_overlap(a: Box, b: Box) -> float:
    """`vov(a, b)`: overlap of the vertical extents relative to the shorter box."""
    overlap = max(0.0, min(a.bottom, b.bottom) - max(a.top, b.top))
    return overlap / min(a.height, b.height)


def _max_gap(following: SymbolDetection, tallest: float, roi_aspect: float) -> float:
    """Widest gap that still joins `following` to the glyph before it (step 2).

    A "1" only lights the right-hand segments of its cell, so the empty cell width adds to the gap on its left. Boxes
    are normalised to the ROI, so the allowance divides by the aspect of the ROI (width / height in pixels).
    """
    allowance = 0.0
    if following.symbol == "1":
        allowance = max(0.0, ONE_CELL_FACTOR * following.box.height / roi_aspect - following.box.width)
    return GROUP_MAX_GAP_FACTOR * tallest + allowance


def _gap_if_joinable(last: SymbolDetection, following: SymbolDetection, roi_aspect: float) -> float | None:
    """The horizontal gap when `following` continues the number ending in `last`, else `None`."""
    a, b = last.box, following.box
    tallest = max(a.height, b.height)
    shortest = min(a.height, b.height)
    gap = b.left - a.right
    narrowest = min(a.width, b.width)
    max_gap = _max_gap(following, tallest, roi_aspect)
    joinable = (
        vertical_overlap(a, b) >= GROUP_MIN_VERTICAL_OVERLAP
        and tallest / shortest <= GROUP_MAX_HEIGHT_RATIO
        and gap <= max_gap
        and gap >= -GROUP_MAX_OVERLAP_FACTOR * narrowest
    )
    return gap if joinable else None


def _passes_size_filter(detection: SymbolDetection) -> bool:
    """Step 1: noise is small. Dashes and colons are small by nature, so they only need to be more than a speck."""
    minimum = MIN_PUNCTUATION_HEIGHT if detection.symbol in ("dash", "colon") else MIN_SYMBOL_HEIGHT
    return detection.confidence >= MIN_SYMBOL_CONFIDENCE and detection.box.height >= minimum


def _group(detections: Sequence[SymbolDetection], roi_aspect: float) -> list[list[SymbolDetection]]:
    """Steps 1 and 2: filter noise, then chain neighbouring glyphs into runs."""
    candidates = sorted(
        (d for d in detections if _passes_size_filter(d)),
        key=lambda d: (d.box.left, d.box.top),
    )
    groups: list[list[SymbolDetection]] = []
    for detection in candidates:
        best: list[SymbolDetection] | None = None
        best_gap = 0.0
        for group in groups:
            gap = _gap_if_joinable(group[-1], detection, roi_aspect)
            if gap is not None and (best is None or gap < best_gap):
                best, best_gap = group, gap
        if best is None:
            groups.append([detection])
        else:
            best.append(detection)
    return groups


def _split_merged_runs(groups: list[list[SymbolDetection]]) -> list[list[SymbolDetection]]:
    """Step 2b: a run of 4 to 6 digits that is really two numbers side by side is split at its widest gap.

    The allowance for a "1" (step 2) can chain two neighbouring numbers when the second one starts with a "1". The run
    is split only when the widest gap is at least `SPLIT_GAP_RATIO` times every other gap and both parts have 2 or 3
    digits; evenly spaced runs (a year, a clock without colon) stay whole and are ignored in step 3.
    """
    result: list[list[SymbolDetection]] = []
    for group in groups:
        digits = len(group)
        if not (MAX_NUMBER_DIGITS < digits <= 2 * MAX_NUMBER_DIGITS) or any(d.digit is None for d in group):
            result.append(group)
            continue
        gaps = [group[i + 1].box.left - group[i].box.right for i in range(digits - 1)]
        widest = max(range(len(gaps)), key=lambda i: gaps[i])
        others = [g for i, g in enumerate(gaps) if i != widest]
        first, second = widest + 1, digits - widest - 1
        clear = all(gaps[widest] >= SPLIT_GAP_RATIO * max(g, 0.0) for g in others) and gaps[widest] > 0
        if (
            clear
            and MIN_NUMBER_DIGITS <= first <= MAX_NUMBER_DIGITS
            and MIN_NUMBER_DIGITS <= second <= MAX_NUMBER_DIGITS
        ):
            result.extend([group[: widest + 1], group[widest + 1 :]])
        else:
            result.append(group)
    return result


def parse_detections(detections: Sequence[SymbolDetection], roi_aspect: float = DEFAULT_ROI_ASPECT) -> FrameReading:
    """Parses the glyphs of one frame (boxes normalised to the ROI) into a frame reading.

    `roi_aspect` is the width / height of the ROI crop in pixels, which the normalised boxes hide.
    """
    if roi_aspect <= 0:
        raise ValueError("roi_aspect must be positive")
    numbers: list[_Number] = []
    has_error = False
    # Dashes sit one digit cell apart and are far flatter than a digit, so they do not chain into runs reliably; two or
    # more of them are the "measuring" display whatever their layout.
    has_measuring = sum(1 for d in detections if d.symbol == "dash" and _passes_size_filter(d)) >= MIN_DASHES
    for group in _split_merged_runs(_group(detections, roi_aspect)):
        symbols = [d.symbol for d in group]
        if "E" in symbols:
            has_error = True
        elif "colon" in symbols:
            continue  # the clock
        elif all(d.digit is not None for d in group) and MIN_NUMBER_DIGITS <= len(group) <= MAX_NUMBER_DIGITS:
            numbers.append(_Number(tuple(group)))
        # anything else (lone memory index, 4+ digits, mixes) is ignored

    if has_error:
        return FrameReading(FrameStatus.DEVICE_ERROR)
    if not numbers:
        return FrameReading(FrameStatus.MEASURING if has_measuring else FrameStatus.NO_READING)
    return _assign(numbers)


def _reading_order(numbers: Sequence[_Number], tallest: float) -> list[_Number]:
    """Candidate numbers ordered row by row (top to bottom) and left to right inside each row (step 5)."""
    secondary = sorted(
        (n for n in numbers if n.box.height >= SECONDARY_MIN_HEIGHT_RATIO * tallest),
        key=lambda n: n.box.center_y,
    )
    rows: list[list[_Number]] = []
    for number in secondary:
        for row in rows:
            if vertical_overlap(row[0].box, number.box) >= ROW_MIN_VERTICAL_OVERLAP:
                row.append(number)
                break
        else:
            rows.append([number])
    ordered_rows = sorted(rows, key=lambda row: min(n.box.top for n in row))
    return [number for row in ordered_rows for number in sorted(row, key=lambda n: n.box.left)]


def _assign(numbers: Sequence[_Number]) -> FrameReading:
    """Step 6: SYS and DIA are the first two primary numbers, PUL the next one in reading order."""
    tallest = max(n.box.height for n in numbers)
    order = _reading_order(numbers, tallest)
    primary = [n for n in order if n.box.height >= PRIMARY_MIN_HEIGHT_RATIO * tallest]
    if len(primary) < 2:
        return FrameReading(FrameStatus.NO_READING)

    systolic, diastolic = primary[0], primary[1]
    following = order.index(diastolic) + 1
    pulse = order[following] if following < len(order) else None
    return FrameReading(
        status=FrameStatus.COMPLETE if pulse is not None else FrameStatus.PARTIAL,
        systolic=systolic.to_field(),
        diastolic=diastolic.to_field(),
        pulse=pulse.to_field() if pulse is not None else None,
        primary_height=tallest,
    )
