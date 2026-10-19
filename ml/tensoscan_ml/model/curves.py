"""Training curves as a PNG, drawn with OpenCV (no plotting dependency)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_COLORS = ((200, 60, 20), (30, 150, 30), (30, 60, 220))


def _panel(
    title: str, series: dict[str, list[float | None]], width: int = 420, height: int = 300
) -> np.ndarray[Any, Any]:
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    values = [v for line in series.values() for v in line if v is not None]
    cv2.putText(canvas, title, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    left, top, right, bottom = 45, 30, width - 12, height - 28
    cv2.rectangle(canvas, (left, top), (right, bottom), (180, 180, 180), 1)
    if not values:
        return canvas
    low, high = min(values), max(values)
    if high - low < 1e-9:
        high = low + 1.0
    count = max(len(line) for line in series.values())
    for color, (name, line) in zip(_COLORS, series.items(), strict=False):
        points = [
            (
                round(left + (right - left) * i / max(count - 1, 1)),
                round(bottom - (bottom - top) * (v - low) / (high - low)),
            )
            for i, v in enumerate(line)
            if v is not None
        ]
        if len(points) > 1:
            cv2.polylines(canvas, [np.array(points, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
        elif points:
            cv2.circle(canvas, points[0], 3, color, -1)
        y = 40 + 16 * list(series).index(name)
        cv2.putText(canvas, name, (right - 110, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{high:.3g}", (2, top + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{low:.3g}", (2, bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1, cv2.LINE_AA)
    return canvas


def plot_curves(rows: Sequence[dict[str, Any]], path: Path) -> None:
    """Loss, validation mAP@0.5 and learning rate per epoch across all stages."""

    def column(key: str) -> list[float | None]:
        return [None if row.get(key) in (None, "") else float(row[key]) for row in rows]

    panels = [
        _panel("loss", {"train": column("train_loss"), "val": column("val_loss")}),
        _panel("val mAP@0.5", {"synthetic": column("val_map50_synthetic"), "real": column("val_map50_real")}),
        _panel("learning rate", {"lr": column("lr")}),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.concatenate(panels, axis=1))
