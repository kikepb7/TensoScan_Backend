"""Helpers for data written by the previous version, where every value was a string and nothing was validated."""

import math
import re

_DIGITS = re.compile(r"\d{1,4}")
LEGACY_MODEL_VERSION = "legacy-keras"


def parse_legacy_integer(value: object) -> int | None:
    """An int from a legacy value (``"128"``, `` " 67 "``, ``128``), or None when it is not a plain number."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and _DIGITS.fullmatch(value.strip()):
        return int(value.strip())
    return None


def legacy_confidence(value: object) -> float:
    """The legacy confidence clamped to ``[0, 1]``; ``1.0`` when it is missing or not a number."""
    if isinstance(value, int | float) and not isinstance(value, bool) and not math.isnan(value):
        return min(1.0, max(0.0, float(value)))
    return 1.0
