"""Synthetic videos and photos built inside the tests (no binary fixtures in git)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

BgrImage = NDArray[np.uint8]


def scene(index: int, size: tuple[int, int] = (960, 720)) -> BgrImage:
    """A sharp, high-contrast image whose layout depends on `index` (so different indexes are visibly different)."""
    width, height = size
    rng = np.random.default_rng(index)
    image = np.full((height, width, 3), 90, dtype=np.uint8)
    for _ in range(4):  # big blocks: low-frequency content, what the perceptual hash looks at
        x0, y0 = int(rng.integers(0, width - 200)), int(rng.integers(0, height - 150))
        w, h = int(rng.integers(120, 360)), int(rng.integers(90, 260))
        cv2.rectangle(
            image,
            (x0, y0),
            (min(width - 1, x0 + w), min(height - 1, y0 + h)),
            tuple(int(v) for v in rng.integers(20, 255, 3)),
            -1,
        )
    for step in range(0, width, 16):  # fine lines: sharp edges for the Laplacian
        cv2.line(image, (step, 0), (step, height - 1), (255, 255, 255), 1)
    cv2.putText(
        image, f"{128 + index}", (width // 3, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 0), 6, cv2.LINE_AA
    )
    return image


def blurry(image: BgrImage) -> BgrImage:
    return np.asarray(cv2.GaussianBlur(image, (0, 0), 14), dtype=np.uint8)


def write_video(path: Path, frames: list[BgrImage], fps: int = 10) -> Path:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))  # type: ignore[attr-defined]
    assert writer.isOpened()
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path


def write_image(path: Path, image: BgrImage) -> Path:
    cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path
