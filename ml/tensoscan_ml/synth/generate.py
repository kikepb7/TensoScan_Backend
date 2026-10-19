"""One synthetic sample: layout -> scene -> augmentations -> exact YOLO labels + metadata (REQ-B003-08)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from tensoscan_ml.config import Config
from tensoscan_ml.contract import CLASS_INDEX
from tensoscan_ml.reading import Box, SymbolDetection, parse_detections
from tensoscan_ml.synth.augment import apply_augmentations
from tensoscan_ml.synth.fonts import sample_palette
from tensoscan_ml.synth.layouts import DisplayLayout, generate_layout
from tensoscan_ml.synth.scene import GlyphLabel, compose_scene

#: Glyphs whose label decides the reading (the rest are distractors such as clock, date or memory index).
KEY_ROLES = frozenset({"sys", "dia", "pul", "error", "measuring"})
MIN_VISIBLE = 0.6  # a glyph less visible than this is not labelled
MIN_BOX_PX = 3.0
#: The parser drops boxes under 0.04 of the ROI height; a sample is "readable" with some margin over that.
READABLE_MIN_HEIGHT = 0.05

SPLIT_IDS = {"train": 0, "val": 1, "test": 2, "gallery": 3}


@dataclass(frozen=True)
class YoloLabel:
    symbol: str
    role: str
    cx: float
    cy: float
    w: float
    h: float

    @property
    def class_index(self) -> int:
        return CLASS_INDEX[self.symbol]

    def line(self) -> str:
        return f"{self.class_index} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"

    def detection(self, confidence: float = 1.0) -> SymbolDetection:
        return SymbolDetection(
            self.symbol,
            Box(self.cx - self.w / 2, self.cy - self.h / 2, self.cx + self.w / 2, self.cy + self.h / 2),
            confidence,
        )


@dataclass
class Sample:
    name: str
    image: NDArray[np.uint8]  # RGB
    labels: list[YoloLabel]
    meta: dict[str, Any]
    jpeg_quality: int
    masks: list[NDArray[np.uint8]] = field(default_factory=list)
    layout: DisplayLayout | None = None


def sample_rng(seed: int, split: str, index: int) -> np.random.Generator:
    """The generator of one sample: independent of the others, so batches can be produced in parallel."""
    return np.random.default_rng(np.random.SeedSequence(entropy=seed, spawn_key=(SPLIT_IDS[split], index)))


def to_yolo(label: GlyphLabel, size: tuple[int, int], margin: float) -> YoloLabel | None:
    """Pixel box -> YOLO normalised box, with the label margin (per side, relative to the box size) added."""
    width, height = size
    x0, y0, x1, y1 = label.box
    pad_x, pad_y = margin * (x1 - x0), margin * (y1 - y0)
    x0, y0 = max(0.0, x0 - pad_x), max(0.0, y0 - pad_y)
    x1, y1 = min(float(width), x1 + pad_x), min(float(height), y1 + pad_y)
    if label.visible < MIN_VISIBLE or x1 - x0 < MIN_BOX_PX * 0.5 or y1 - y0 < MIN_BOX_PX:
        return None
    return YoloLabel(
        label.symbol, label.role, (x0 + x1) / 2 / width, (y0 + y1) / 2 / height, (x1 - x0) / width, (y1 - y0) / height
    )


def generate_sample(
    config: Config,
    seed: int,
    split: str,
    index: int,
    *,
    with_masks: bool = False,
    template: str | None = None,
    labels_only: bool = False,
) -> Sample:
    """Generates sample `index` of `split`. Pure function of `(config, seed, split, index)`.

    `labels_only` skips the photographic rendering (the image is black): labels and metadata are identical, and much
    cheaper to obtain, which is all the parser-consistency tests need."""
    rng = sample_rng(seed, split, index)
    layout = generate_layout(rng, config, template)
    palette = sample_palette(rng)
    scene = compose_scene(layout, palette, rng, config, with_masks=with_masks, labels_only=labels_only)
    image, augmentation = (scene.image, {}) if labels_only else apply_augmentations(scene.image, rng, config)
    quality = int(rng.integers(config["image"]["jpeg_quality"][0], config["image"]["jpeg_quality"][1] + 1))

    out_w, out_h = int(config["image"]["width"]), int(config["image"]["height"])
    margin = float(config["glyph"]["label_margin"])
    kept: list[tuple[GlyphLabel, YoloLabel]] = []
    for glyph_label in scene.labels:
        yolo = to_yolo(glyph_label, (out_w, out_h), margin)
        if yolo is not None:
            kept.append((glyph_label, yolo))
    labels = [yolo for _, yolo in kept]
    masks = (
        [scene.masks[i] for i, label in enumerate(scene.labels) if any(label is k[0] for k in kept)]
        if with_masks
        else []
    )

    name = f"{index:06d}"
    meta = _metadata(
        name, split, seed, index, layout, palette.kind, scene.params, augmentation, quality, scene.labels, labels
    )
    rgb = np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8)
    return Sample(name, rgb, labels, meta, quality, masks, layout)


def _metadata(
    name: str,
    split: str,
    seed: int,
    index: int,
    layout: DisplayLayout,
    palette: str,
    scene_params: dict[str, object],
    augmentation: dict[str, float],
    quality: int,
    drawn: list[GlyphLabel],
    labels: list[YoloLabel],
) -> dict[str, Any]:
    key_drawn = sum(1 for g in drawn if g.role in KEY_ROLES)
    key_labels = [label for label in labels if label.role in KEY_ROLES]
    min_height = min((label.h for label in key_labels), default=0.0)
    readable = bool(key_labels) and len(key_labels) == key_drawn and min_height >= READABLE_MIN_HEIGHT
    expected = layout.expected
    parsed = parse_detections([label.detection() for label in labels])
    matches: bool | None = None
    if layout.covered and expected is not None:
        matches = parsed.status.value == expected.status and parsed.values == (
            expected.systolic,
            expected.diastolic,
            expected.pulse,
        )
    style = layout.font.style
    return {
        "name": name,
        "split": split,
        "seed": seed,
        "index": index,
        "template": layout.template,
        "base_template": layout.extras.get("base_template"),
        "covered": layout.covered,
        "expected": None
        if expected is None
        else {"status": expected.status, "sys": expected.systolic, "dia": expected.diastolic, "pul": expected.pulse},
        "value_kind": layout.extras.get("value_kind"),
        "distractors": {key: bool(layout.extras.get(key)) for key in ("clock", "date", "memory")},
        "palette": palette,
        "font": {
            "thickness": round(style.thickness, 4),
            "gap": round(style.gap, 4),
            "slant_deg": round(style.slant_deg, 3),
            "bevel": round(style.bevel, 3),
            "rounded": style.rounded,
            "width_ratio": round(style.width_ratio, 4),
            "spacing": round(layout.font.spacing, 4),
            "ghost_alpha": round(style.ghost_alpha, 4),
        },
        "parser_compatible": layout.font.parser_compatible,
        "scene": scene_params,
        "augmentation": augmentation,
        "jpeg_quality": quality,
        "num_labels": len(labels),
        "dropped_glyphs": len(drawn) - len(labels),
        "key_min_height": round(min_height, 4),
        "readable": readable,
        "parser_status": parsed.status.value,
        "parser_values": list(parsed.values),
        "parser_matches": matches,
        "extras": {
            k: v for k, v in layout.extras.items() if k in ("error_kind", "rows", "negative_kind", "coverage_kind")
        },
    }


def encode_image(sample: Sample, fmt: str) -> bytes:
    """Bytes of the image file. JPEG encoding is deterministic for a given OpenCV build."""
    bgr = cv2.cvtColor(sample.image, cv2.COLOR_RGB2BGR)
    if fmt == "png":
        ok, buffer = cv2.imencode(".png", bgr)
    else:
        ok, buffer = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, sample.jpeg_quality])
    if not ok:  # pragma: no cover
        raise RuntimeError("could not encode the image")
    return bytes(buffer.tobytes())


def label_text(sample: Sample) -> str:
    return "".join(label.line() + "\n" for label in sample.labels)


__all__ = [
    "KEY_ROLES",
    "Sample",
    "YoloLabel",
    "encode_image",
    "generate_sample",
    "label_text",
    "sample_rng",
    "to_yolo",
]
