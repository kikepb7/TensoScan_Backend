"""Contact sheet of synthetic samples with their labels drawn, for human review (T-B003-05)."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from tensoscan_ml.config import Config
from tensoscan_ml.synth.generate import Sample, generate_sample
from tensoscan_ml.synth.raster import as_uint8

_PALETTE = ((230, 30, 30), (30, 150, 30), (30, 60, 230), (200, 120, 0), (150, 0, 150))


def annotate(sample: Sample) -> NDArray[np.uint8]:
    """The sample image with each label box and class name drawn on it."""
    image = np.ascontiguousarray(sample.image.copy())
    height, width = image.shape[:2]
    for label in sample.labels:
        color = _PALETTE[label.class_index % len(_PALETTE)]
        x0, y0 = round((label.cx - label.w / 2) * width), round((label.cy - label.h / 2) * height)
        x1, y1 = round((label.cx + label.w / 2) * width), round((label.cy + label.h / 2) * height)
        cv2.rectangle(image, (x0, y0), (x1, y1), color, 1)
        cv2.putText(image, label.symbol, (x0, max(10, y0 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return image


def render_gallery(
    config: Config, seed: int, count: int = 50, columns: int = 10, tile_width: int = 96
) -> NDArray[np.uint8]:
    """A `columns`-wide grid with `count` annotated samples (`gallery` split, so it never overlaps a dataset)."""
    tiles: list[NDArray[np.uint8]] = []
    for index in range(count):
        annotated = annotate(generate_sample(config, seed, "gallery", index))
        scale = tile_width / annotated.shape[1]
        tiles.append(as_uint8(cv2.resize(annotated, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)))
    tile_height = tiles[0].shape[0]
    rows = -(-count // columns)
    sheet = np.full((rows * tile_height, columns * tile_width, 3), 255, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        sheet[row * tile_height : (row + 1) * tile_height, column * tile_width : (column + 1) * tile_width] = tile
    return sheet
