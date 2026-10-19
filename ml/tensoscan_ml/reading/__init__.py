"""Official Python implementation of `specs/contracts/reading-rules.md` (REQ-B003-04).

The same rules run in the app (Kotlin), in the backend validation and here, where the parser measures the accuracy of
the model per reading. Constants and algorithm are the contract's; changing one is a new contract version.
"""

from tensoscan_ml.reading.models import Box, FieldValue, FrameReading, FrameStatus, SymbolDetection
from tensoscan_ml.reading.parser import parse_detections
from tensoscan_ml.reading.validator import ValidationIssue, ValidationResult, validate_reading

__all__ = [
    "Box",
    "FieldValue",
    "FrameReading",
    "FrameStatus",
    "SymbolDetection",
    "ValidationIssue",
    "ValidationResult",
    "parse_detections",
    "validate_reading",
]
