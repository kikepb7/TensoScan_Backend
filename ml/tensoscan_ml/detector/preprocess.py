"""Letterbox preprocessing: numerically the same as the app's `LetterboxPreprocessor` (contract section 3).

The frame is scaled keeping its aspect ratio until the longest side fits the input, centred on a grey (114/255) square
and normalised to `0..1`, RGB, NHWC. Bilinear sampling maps pixel centres to pixel centres, like the Kotlin code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tensoscan_ml.contract import INPUT_SIZE, PAD_VALUE

RgbImage = NDArray[np.uint8]


@dataclass(frozen=True)
class LetterboxTransform:
    """How a frame was fitted into the square input; needed to map detections back to the frame."""

    input_size: int
    scaled_width: int
    scaled_height: int
    pad_left: int
    pad_top: int


def _round_half_up(value: float) -> int:
    return math.floor(value + 0.5)


def letterbox_transform(width: int, height: int, input_size: int = INPUT_SIZE) -> LetterboxTransform:
    scale = min(input_size / width, input_size / height)
    scaled_width = min(max(1, _round_half_up(width * scale)), input_size)
    scaled_height = min(max(1, _round_half_up(height * scale)), input_size)
    return LetterboxTransform(
        input_size, scaled_width, scaled_height, (input_size - scaled_width) // 2, (input_size - scaled_height) // 2
    )


def letterbox(image_rgb: RgbImage, input_size: int = INPUT_SIZE) -> tuple[NDArray[np.float32], LetterboxTransform]:
    """Returns the `[1, S, S, 3]` float32 model input and the transform that produced it."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("expected an RGB image of shape (height, width, 3)")
    height, width = image_rgb.shape[:2]
    transform = letterbox_transform(width, height, input_size)

    source_x = np.clip((np.arange(transform.scaled_width) + 0.5) / transform.scaled_width * width - 0.5, 0, width - 1)
    source_y = np.clip(
        (np.arange(transform.scaled_height) + 0.5) / transform.scaled_height * height - 0.5, 0, height - 1
    )
    x0 = source_x.astype(np.int64)
    y0 = source_y.astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    fx = (source_x - x0).astype(np.float32)[None, :, None]
    fy = (source_y - y0).astype(np.float32)[:, None, None]
    pixels = image_rgb.astype(np.float32)
    top = pixels[y0][:, x0] * (1 - fx) + pixels[y0][:, x1] * fx
    bottom = pixels[y1][:, x0] * (1 - fx) + pixels[y1][:, x1] * fx
    scaled = (top * (1 - fy) + bottom * fy) / 255.0

    output = np.full((1, input_size, input_size, 3), PAD_VALUE, dtype=np.float32)
    output[
        0,
        transform.pad_top : transform.pad_top + transform.scaled_height,
        transform.pad_left : transform.pad_left + transform.scaled_width,
    ] = scaled
    return output, transform
