"""Perceptual hash (DCT based, 64 bits) used to de-duplicate frames; implemented here to avoid heavy dependencies."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

HASH_BITS = 64


def phash(image_bgr: NDArray[np.uint8]) -> int:
    """64-bit pHash: 32x32 grey image -> DCT -> the 8x8 low frequencies -> bits above their median."""
    grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    small = cv2.resize(grey, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = np.asarray(cv2.dct(small))[:8, :8].flatten()
    median = float(np.median(low[1:]))  # the DC term (mean brightness) would dominate the median
    bits = (low > median).astype(np.uint8)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming(first: int, second: int) -> int:
    return (first ^ second).bit_count()
