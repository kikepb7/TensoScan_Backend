"""Detection decoding: numerically the same as the app's `DetectionDecoder` (contract section 6).

Threshold, class-agnostic NMS, limit, and undoing the letterbox so the boxes are normalised to the ROI.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from tensoscan_ml.contract import BOX_VALUES, CLASSES, MAX_DETECTIONS, NMS_IOU_THRESHOLD, SCORE_THRESHOLD
from tensoscan_ml.detector.preprocess import LetterboxTransform
from tensoscan_ml.reading import Box, SymbolDetection


def _iou(a: Box, b: Box) -> float:
    width = min(a.right, b.right) - max(a.left, b.left)
    height = min(a.bottom, b.bottom) - max(a.top, b.top)
    if width <= 0 or height <= 0:
        return 0.0
    intersection = width * height
    union = a.width * a.height + b.width * b.height - intersection
    return 0.0 if union <= 0 else intersection / union


def decode_output(
    output: NDArray[np.float32],
    transform: LetterboxTransform,
    *,
    classes: Sequence[str] = CLASSES,
    score_threshold: float = SCORE_THRESHOLD,
    nms_iou_threshold: float = NMS_IOU_THRESHOLD,
    max_detections: int = MAX_DETECTIONS,
) -> list[SymbolDetection]:
    """Decodes a `[N, 4 + C]` (or `[1, N, 4 + C]`) model output into detections in coordinates normalised to the ROI."""
    rows = output.reshape(-1, BOX_VALUES + len(classes))
    scores = rows[:, BOX_VALUES:]
    best_class = scores.argmax(axis=1)
    best_score = scores.max(axis=1)
    keep = np.nonzero(best_score >= score_threshold)[0]

    size = float(transform.input_size)
    candidates: list[SymbolDetection] = []
    for index in keep:
        cx, cy, w, h = (float(v) * size for v in rows[index, :BOX_VALUES])
        left = min(max((cx - w / 2 - transform.pad_left) / transform.scaled_width, 0.0), 1.0)
        right = min(max((cx + w / 2 - transform.pad_left) / transform.scaled_width, 0.0), 1.0)
        top = min(max((cy - h / 2 - transform.pad_top) / transform.scaled_height, 0.0), 1.0)
        bottom = min(max((cy + h / 2 - transform.pad_top) / transform.scaled_height, 0.0), 1.0)
        if right > left and bottom > top:  # a box that lies entirely in the border is ignored
            candidates.append(
                SymbolDetection(
                    classes[int(best_class[index])], Box(left, top, right, bottom), float(best_score[index])
                )
            )

    candidates.sort(key=lambda d: -d.confidence)
    kept: list[SymbolDetection] = []
    for candidate in candidates:  # class-agnostic: glyphs of a display never overlap
        if all(_iou(other.box, candidate.box) <= nms_iou_threshold for other in kept):
            kept.append(candidate)
    return kept[:max_detections]
