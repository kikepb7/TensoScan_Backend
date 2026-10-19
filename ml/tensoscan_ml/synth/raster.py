"""Anti-aliased polygon rasterisation with sub-pixel precision."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

_SHIFT = 4  # fractional bits of the polygon coordinates

Image = NDArray[np.uint8]
Polygon = NDArray[np.float64]


def _fixed_point(polygon: Polygon, scale: float) -> NDArray[np.int32]:
    # Polygons use *edge* coordinates (pixel i covers [i, i + 1]); OpenCV draws with pixel-centre coordinates.
    return np.round((polygon * scale - 0.5) * (1 << _SHIFT)).astype(np.int32)


def fill_polygons(
    target: NDArray[np.uint8],
    polygons: Sequence[Polygon],
    value: int | tuple[int, ...],
    scale: float = 1.0,
    *,
    antialias: bool = True,
) -> None:
    """Fills each polygon (coordinates multiplied by `scale`) in place.

    With anti-aliasing OpenCV also draws the outline, so the shape grows by about half a pixel; masks used to check
    the labels are drawn without it.
    """
    line_type = cv2.LINE_AA if antialias else cv2.LINE_8
    for polygon in polygons:
        cv2.fillPoly(target, [_fixed_point(polygon, scale)], value, lineType=line_type, shift=_SHIFT)


def polygon_mask(
    shape: tuple[int, int], polygons: Sequence[Polygon], scale: float = 1.0, *, antialias: bool = False
) -> Image:
    """Single-channel mask (0..255) of the union of `polygons` (hard edges by default)."""
    mask: Image = np.zeros(shape, dtype=np.uint8)
    fill_polygons(mask, polygons, 255, scale, antialias=antialias)
    return mask


def blend(target: NDArray[np.float32], mask: Image, color: tuple[int, int, int], alpha: float) -> None:
    """`target = target * (1 - a*m) + color * a*m`, where `m` is the mask in 0..1, in place (RGB float image)."""
    weight = (mask.astype(np.float32) / 255.0 * alpha)[..., None]
    target *= 1.0 - weight
    target += weight * np.array(color, dtype=np.float32)


def as_float32(array: object) -> NDArray[np.float32]:
    """Typed view of an OpenCV result as `float32` (OpenCV's stubs only promise "some number type")."""
    return np.asarray(array, dtype=np.float32)


def as_uint8(array: object) -> NDArray[np.uint8]:
    return np.asarray(array, dtype=np.uint8)
