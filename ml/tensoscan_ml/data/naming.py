"""File-name convention of the real captures and the `labels.csv` alternative (REQ-B003-09, design section 4).

```
<brand>-<model>__<sys>-<dia>-<pul>__<condition>__<nn>.<ext>      omron-m3__128-82-67__dim__01.mp4
<brand>-<model>__error__<condition>__<nn>.<ext>                   omron-m3__error__good__01.mp4
```
Every problem is reported with a code and a message: a name that does not follow the convention is never guessed.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from tensoscan_ml.reading import validate_reading

VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"})
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
CONDITIONS = ("good", "dim", "glare")
STATUS_READING = "COMPLETE"
STATUS_ERROR = "DEVICE_ERROR"

_DEVICE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_NAME = re.compile(
    r"^(?P<device>[^_]+)__(?P<reading>\d{2,3}-\d{2,3}-\d{2,3}|error)__(?P<condition>[a-z]+)__(?P<take>\d{1,3})$"
)
LABELS_COLUMNS = ("file", "device", "sys", "dia", "pul", "condition")


class NamingError(ValueError):
    """A file name or a `labels.csv` row that does not follow the convention."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class CaptureLabel:
    """What the capture shows: three values, or a device error."""

    device: str
    status: str
    sys: int | None
    dia: int | None
    pul: int | None
    condition: str
    take: str | None = None

    @property
    def reading_key(self) -> str:
        """Group key of the reading for splitting (`128-82-67`, or `error`)."""
        return "error" if self.status == STATUS_ERROR else f"{self.sys}-{self.dia}-{self.pul}"


@dataclass(frozen=True)
class Roi:
    """A manual region of interest, in pixels of the (upright) image: `x, y, width, height`."""

    x: int
    y: int
    width: int
    height: int


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def is_media(path: Path) -> bool:
    return is_video(path) or is_image(path)


def check_device(device: str) -> str:
    cleaned = device.strip().lower()
    if not _DEVICE.match(cleaned):
        raise NamingError(
            "BAD_DEVICE", f"device '{device}' must look like brand-model (lowercase letters, digits, '-')"
        )
    return cleaned


def check_condition(condition: str) -> str:
    cleaned = condition.strip().lower()
    if cleaned not in CONDITIONS:
        raise NamingError("BAD_CONDITION", f"condition '{condition}' must be one of {', '.join(CONDITIONS)}")
    return cleaned


def build_label(
    device: str, sys: int | None, dia: int | None, pul: int | None, condition: str, take: str | None = None
) -> CaptureLabel:
    """Validates a reading and returns the label. `sys=None` means the display shows an error code."""
    checked_device, checked_condition = check_device(device), check_condition(condition)
    if sys is None and dia is None and pul is None:
        return CaptureLabel(checked_device, STATUS_ERROR, None, None, None, checked_condition, take)
    result = validate_reading(sys, dia, pul)
    if result.errors:
        detail = ", ".join(f"{issue.field}:{issue.code}" for issue in result.errors)
        raise NamingError("IMPLAUSIBLE_READING", f"reading {sys}/{dia}/{pul} is not valid ({detail})")
    return CaptureLabel(checked_device, STATUS_READING, sys, dia, pul, checked_condition, take)


def parse_name(filename: str | Path) -> CaptureLabel:
    """Parses `<brand>-<model>__<sys>-<dia>-<pul>__<condition>__<nn>.<ext>` (or `error` instead of the values)."""
    path = Path(filename)
    if not is_media(path):
        raise NamingError("BAD_EXTENSION", f"'{path.name}' is not a supported video or image file")
    match = _NAME.match(path.stem)
    if match is None:
        raise NamingError(
            "BAD_NAME",
            f"'{path.name}' does not follow <brand>-<model>__<sys>-<dia>-<pul>__<condition>__<nn>.<ext>",
        )
    reading = match["reading"]
    if reading == "error":
        return build_label(match["device"], None, None, None, match["condition"], match["take"])
    sys, dia, pul = (int(part) for part in reading.split("-"))
    return build_label(match["device"], sys, dia, pul, match["condition"], match["take"])


@dataclass(frozen=True)
class LabelRow:
    """A validated row of `labels.csv`."""

    file: str
    label: CaptureLabel
    roi: Roi | None


@dataclass(frozen=True)
class RowProblem:
    line: int
    file: str
    code: str
    message: str


def _int(raw: str, field: str) -> int:
    try:
        return int(raw.strip())
    except ValueError as error:
        raise NamingError("BAD_NUMBER", f"{field} '{raw}' is not an integer") from error


def parse_labels_csv(path: str | Path) -> tuple[list[LabelRow], list[RowProblem]]:
    """Reads `file,device,sys,dia,pul,condition[,roi_x,roi_y,roi_w,roi_h]`.

    An error reading has `error` in the `sys` column (and empty `dia` and `pul`). A missing or empty `condition`
    is an error: nothing is defaulted silently. Returns the valid rows and one problem per rejected row.
    """
    rows: list[LabelRow] = []
    problems: list[RowProblem] = []
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = [name.strip().lower() for name in (reader.fieldnames or [])]
        missing = [column for column in LABELS_COLUMNS if column not in header]
        if missing:
            raise ValueError(f"{path}: missing columns {', '.join(missing)} (expected {','.join(LABELS_COLUMNS)})")
        reader.fieldnames = header
        seen: set[str] = set()
        for record in reader:
            line = reader.line_num
            file = (record.get("file") or "").strip()
            try:
                if not file:
                    raise NamingError("MISSING_FILE", "the file column is empty")
                if file in seen:
                    raise NamingError("DUPLICATE_ROW", f"'{file}' appears more than once")
                seen.add(file)
                rows.append(LabelRow(file, _row_label(record), _row_roi(record)))
            except NamingError as error:
                problems.append(RowProblem(line, file, error.code, error.message))
    return rows, problems


def _row_label(record: dict[str, str | None]) -> CaptureLabel:
    device = (record.get("device") or "").strip()
    condition = (record.get("condition") or "").strip()
    sys_raw = (record.get("sys") or "").strip()
    if sys_raw.lower() == "error":
        return build_label(device, None, None, None, condition)
    values = [(record.get(name) or "").strip() for name in ("sys", "dia", "pul")]
    if not all(values):
        raise NamingError("MISSING_VALUE", "sys, dia and pul are all required (or 'error' in sys)")
    sys, dia, pul = (_int(raw, name) for raw, name in zip(values, ("sys", "dia", "pul"), strict=True))
    return build_label(device, sys, dia, pul, condition)


def _row_roi(record: dict[str, str | None]) -> Roi | None:
    parts = [(record.get(name) or "").strip() for name in ("roi_x", "roi_y", "roi_w", "roi_h")]
    if not any(parts):
        return None
    if not all(parts):
        raise NamingError("BAD_ROI", "roi_x, roi_y, roi_w and roi_h must all be filled in or all empty")
    x, y, w, h = (_int(raw, name) for raw, name in zip(parts, ("roi_x", "roi_y", "roi_w", "roi_h"), strict=True))
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise NamingError("BAD_ROI", "the ROI must have x, y >= 0 and positive width and height")
    return Roi(x, y, w, h)


def distinct_readings(labels: Iterable[CaptureLabel]) -> set[str]:
    return {label.reading_key for label in labels if label.status == STATUS_READING}
