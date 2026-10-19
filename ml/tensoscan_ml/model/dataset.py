"""Training data: YOLO label files for synthetic datasets and for real frames listed in a manifest (REQ-B003-15)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import ConcatDataset, Dataset, WeightedRandomSampler

from tensoscan_ml.contract import INPUT_SIZE, PAD_GREY
from tensoscan_ml.data.manifest import read_manifest
from tensoscan_ml.detector.preprocess import letterbox

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class AugmentConfig:
    """Light on-the-fly augmentation on top of the generator's own (no flips: a mirrored digit is another glyph)."""

    scale: float = 0.15
    translate: float = 0.08
    rotate_deg: float = 4.0
    brightness: float = 0.25
    blur_prob: float = 0.15
    noise_prob: float = 0.3

    @classmethod
    def off(cls) -> AugmentConfig:
        return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def read_labels(path: Path | None) -> NDArray[np.float32]:
    """`[G, 5]` array of `class cx cy w h` (normalised to the image)."""
    if path is None or not path.is_file():
        return np.zeros((0, 5), dtype=np.float32)
    rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return np.array([[float(v) for v in row[:5]] for row in rows], dtype=np.float32).reshape(-1, 5)


def items_from_yolo_dir(split_dir: str | Path) -> list[tuple[Path, Path | None]]:
    """`<split>/images/*` with `<split>/labels/<stem>.txt` (what `tensoscan-ml synth` writes)."""
    root = Path(split_dir)
    images = sorted(p for p in (root / "images").iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    return [(image, root / "labels" / f"{image.stem}.txt") for image in images]


def items_from_manifest(manifest: str | Path) -> list[tuple[Path, Path | None]]:
    """Entries of a manifest that have boxes (`labels` set); the rest is only used for reading accuracy."""
    path = Path(manifest)
    return [
        (path.parent / entry.path, path.parent / entry.labels)
        for entry in read_manifest(path)
        if entry.labels and (path.parent / entry.labels).is_file()
    ]


class DigitDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Letterboxed images (the app's preprocessing) and their boxes in input pixels."""

    def __init__(
        self,
        items: list[tuple[Path, Path | None]],
        input_size: int = INPUT_SIZE,
        augment: AugmentConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.items = items
        self.input_size = input_size
        self.augment = augment or AugmentConfig.off()
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.items)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, label_path = self.items[index]
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"cannot read {image_path}")
        image = np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
        labels = read_labels(label_path)
        rng = np.random.default_rng([self.seed, self.epoch, index])
        image, labels = self._augment(image, labels, rng)
        model_input, transform = letterbox(image, self.input_size)
        targets = labels.copy()
        targets[:, 1] = labels[:, 1] * transform.scaled_width + transform.pad_left
        targets[:, 2] = labels[:, 2] * transform.scaled_height + transform.pad_top
        targets[:, 3] = labels[:, 3] * transform.scaled_width
        targets[:, 4] = labels[:, 4] * transform.scaled_height
        tensor = torch.from_numpy(model_input[0]).permute(2, 0, 1).contiguous()
        return tensor, torch.from_numpy(targets)

    def _augment(
        self, image: NDArray[np.uint8], labels: NDArray[np.float32], rng: np.random.Generator
    ) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
        cfg = self.augment
        height, width = image.shape[:2]
        if cfg.scale or cfg.translate or cfg.rotate_deg:
            scale = 1.0 + rng.uniform(-cfg.scale, cfg.scale)
            angle = rng.uniform(-cfg.rotate_deg, cfg.rotate_deg)
            matrix = np.asarray(cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale), dtype=np.float64)
            matrix[0, 2] += rng.uniform(-cfg.translate, cfg.translate) * width
            matrix[1, 2] += rng.uniform(-cfg.translate, cfg.translate) * height
            image = np.asarray(
                cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=(PAD_GREY,) * 3),
                dtype=np.uint8,
            )
            labels = _warp_boxes(labels, matrix, width, height)
        if cfg.brightness:
            gain = 1.0 + rng.uniform(-cfg.brightness, cfg.brightness)
            bias = rng.uniform(-20, 20)
            image = np.clip(image.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
        if cfg.blur_prob and rng.random() < cfg.blur_prob:
            image = np.asarray(cv2.GaussianBlur(image, (0, 0), rng.uniform(0.5, 1.5)), dtype=np.uint8)
        if cfg.noise_prob and rng.random() < cfg.noise_prob:
            noise = rng.normal(0, rng.uniform(1, 6), image.shape)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return image, labels


def _warp_boxes(
    labels: NDArray[np.float32], matrix: NDArray[np.float64], width: int, height: int
) -> NDArray[np.float32]:
    """Boxes after an affine warp: the envelope of the warped corners, clipped; boxes mostly outside are dropped."""
    kept: list[list[float]] = []
    for cls, cx, cy, w, h in labels:
        corners = np.array(
            [
                [(cx - w / 2) * width, (cy - h / 2) * height],
                [(cx + w / 2) * width, (cy - h / 2) * height],
                [(cx + w / 2) * width, (cy + h / 2) * height],
                [(cx - w / 2) * width, (cy + h / 2) * height],
            ]
        )
        warped = corners @ matrix[:, :2].T + matrix[:, 2]
        x0, y0 = warped.min(axis=0)
        x1, y1 = warped.max(axis=0)
        area = (x1 - x0) * (y1 - y0)
        cx0, cy0, cx1, cy1 = max(0.0, x0), max(0.0, y0), min(float(width), x1), min(float(height), y1)
        if cx1 <= cx0 or cy1 <= cy0 or (cx1 - cx0) * (cy1 - cy0) < 0.6 * area:
            continue
        kept.append([cls, (cx0 + cx1) / 2 / width, (cy0 + cy1) / 2 / height, (cx1 - cx0) / width, (cy1 - cy0) / height])
    return np.array(kept, dtype=np.float32).reshape(-1, 5)


def collate(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, list[torch.Tensor]]:
    return torch.stack([item[0] for item in batch]), [item[1] for item in batch]


def mixed_sampler(
    datasets: list[DigitDataset], weights: list[float], epoch_size: int, seed: int, *, replacement: bool = True
) -> tuple[ConcatDataset[tuple[torch.Tensor, torch.Tensor]], WeightedRandomSampler]:
    """Concatenates the sources and draws `epoch_size` samples with each source getting its share (`weights`)."""
    per_item: list[float] = []
    for dataset, weight in zip(datasets, weights, strict=True):
        per_item.extend([weight / max(len(dataset), 1)] * len(dataset))
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(per_item, num_samples=epoch_size, replacement=replacement, generator=generator)
    return ConcatDataset(datasets), sampler


def cosine_with_warmup(step: int, total: int, warmup: int, floor: float) -> float:
    """Learning-rate multiplier: linear warm-up, then cosine decay to `floor`."""
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
