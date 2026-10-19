"""Photometric augmentations that imitate the app's ROI (design section 3.3, REQ-B003-07).

Order: shadow, glare, gamma, white balance, blur, resolution round trip, sensor noise. JPEG compression is applied when
the image is written. Every step has a probability and a range in `configs/synth/default.yaml`; all randomness comes
from the generator passed in, so a sample is reproducible from its seed.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from numpy.typing import NDArray

from tensoscan_ml.config import Config
from tensoscan_ml.synth.params import Rng, chance, uniform
from tensoscan_ml.synth.raster import as_float32

FloatImage = NDArray[np.float32]


def shadow(image: FloatImage, rng: Rng) -> tuple[FloatImage, dict[str, float]]:
    """A soft half-plane shadow with a random direction."""
    height, width = image.shape[:2]
    angle = rng.uniform(0, 2 * math.pi)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    distance = (xx - width / 2) * math.cos(angle) + (yy - height / 2) * math.sin(angle)
    offset = rng.uniform(-0.3, 0.3) * max(width, height)
    softness = rng.uniform(0.04, 0.25) * max(width, height)
    depth = rng.uniform(0.25, 0.65)
    ramp = 1.0 / (1.0 + np.exp(-(distance - offset) / softness))
    return image * (1.0 - depth * ramp)[..., None], {"shadow_depth": round(float(depth), 3)}


def glare(image: FloatImage, rng: Rng, alpha_range: tuple[float, float]) -> tuple[FloatImage, dict[str, float]]:
    """Specular reflection: an elongated Gaussian blob, often placed over the display."""
    height, width = image.shape[:2]
    cx, cy = rng.uniform(0.15, 0.85) * width, rng.uniform(0.15, 0.85) * height
    ax, ay = rng.uniform(0.08, 0.5) * width, rng.uniform(0.03, 0.3) * height
    angle = rng.uniform(0, math.pi)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    u = dx * math.cos(angle) + dy * math.sin(angle)
    v = -dx * math.sin(angle) + dy * math.cos(angle)
    profile = np.exp(-(((u / ax) ** 2 + (v / ay) ** 2) ** 1.2)).astype(np.float32)
    alpha = rng.uniform(*alpha_range)
    tint = np.array([1.0, rng.uniform(0.93, 1.0), rng.uniform(0.85, 1.0)], dtype=np.float32)
    weight = (alpha * profile)[..., None]
    return image * (1.0 - weight) + weight * tint, {"glare_alpha": round(float(alpha), 3)}


def motion_blur(image: FloatImage, rng: Rng, kernel_range: tuple[int, int]) -> tuple[FloatImage, dict[str, float]]:
    size = int(rng.integers(kernel_range[0], kernel_range[1] + 1)) | 1
    angle = rng.uniform(0, 180)
    kernel = np.zeros((size, size), dtype=np.float32)
    cv2.line(kernel, (0, size // 2), (size - 1, size // 2), 1.0, 1)
    rotation = cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle, 1.0)
    kernel = as_float32(cv2.warpAffine(kernel, rotation, (size, size)))
    kernel /= max(float(kernel.sum()), 1e-6)
    return as_float32(cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)), {"motion_blur": float(size)}


def sensor_noise(image: FloatImage, rng: Rng, sigma: float, shot: float) -> FloatImage:
    """Poisson (shot) noise plus Gaussian read noise."""
    result = image
    if shot > 0:
        photons = 255.0 / max(shot, 1e-3) * 2.0
        result = rng.poisson(np.clip(result, 0.0, 1.0) * photons).astype(np.float32) / photons
    return result + rng.normal(0.0, sigma / 255.0, size=result.shape).astype(np.float32)


def apply_augmentations(image: FloatImage, rng: Rng, config: Config) -> tuple[FloatImage, dict[str, float]]:
    """Applies the configured chain. Returns the image (0..1) and the parameters that were used."""
    cfg = config["augment"]
    used: dict[str, float] = {}
    out = image
    if chance(rng, float(cfg["shadow_prob"])):
        out, info = shadow(out, rng)
        used.update(info)
    if chance(rng, float(cfg["glare_prob"])):
        out, info = glare(out, rng, (float(cfg["glare_alpha"][0]), float(cfg["glare_alpha"][1])))
        used.update(info)
    gamma = uniform(rng, cfg["gamma"])
    out = np.power(np.clip(out, 0.0, 1.0), gamma)
    used["gamma"] = round(gamma, 3)
    gains = 1.0 + rng.uniform(-1.0, 1.0, size=3).astype(np.float32) * float(cfg["white_balance"])
    out = out * gains
    if chance(rng, float(cfg["motion_blur_prob"])):
        out, info = motion_blur(out, rng, (int(cfg["motion_blur_kernel"][0]), int(cfg["motion_blur_kernel"][1])))
        used.update(info)
    elif chance(rng, float(cfg["gaussian_blur_prob"])):
        sigma = uniform(rng, cfg["gaussian_blur_sigma"])
        out = as_float32(cv2.GaussianBlur(out, (0, 0), sigma))
        used["gaussian_blur"] = round(sigma, 3)
    if chance(rng, float(cfg["rescale_prob"])):
        factor = uniform(rng, cfg["rescale_range"])
        height, width = out.shape[:2]
        small = cv2.resize(
            out, (max(8, round(width * factor)), max(8, round(height * factor))), interpolation=cv2.INTER_AREA
        )
        out = as_float32(cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR))
        used["rescale"] = round(factor, 3)
    if chance(rng, float(cfg["noise_prob"])):
        sigma = uniform(rng, cfg["noise_sigma"])
        out = sensor_noise(out, rng, sigma, float(cfg["shot_noise"]))
        used["noise_sigma"] = round(sigma, 3)
    return np.clip(out, 0.0, 1.0).astype(np.float32), used
