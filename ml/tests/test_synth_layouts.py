"""T-B003-04: layout templates, YOLO labels and metadata; generator and official parser agree (REQ-B003-06/08)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tensoscan_ml.config import Config
from tensoscan_ml.contract import CLASSES
from tensoscan_ml.reading import parse_detections
from tensoscan_ml.synth.fonts import DigitFont
from tensoscan_ml.synth.generate import generate_sample
from tensoscan_ml.synth.glyph import GlyphStyle, build_glyph
from tensoscan_ml.synth.layouts import COVERED_TEMPLATES, generate_layout
from tensoscan_ml.synth.params import TEMPLATE_NAMES, load_synth_config, validate_synth_config

FLAT_SCENE = {"scene": {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}}


@pytest.fixture(scope="module")
def config() -> Config:
    return load_synth_config()


@pytest.fixture(scope="module")
def flat_config() -> Config:
    return load_synth_config(overrides=FLAT_SCENE)


def test_default_layout_shares_add_up_to_one_and_cover_every_template(config: Config) -> None:
    shares = {name: config["layouts"][name] for name in TEMPLATE_NAMES}

    assert sum(shares.values()) == pytest.approx(1.0)
    assert shares["stacked_small_pulse"] == pytest.approx(0.35)


def test_invalid_layout_shares_are_rejected(config: Config) -> None:
    broken = {**config, "layouts": {**config["layouts"], "error": 0.5}}

    with pytest.raises(ValueError, match="add up to 1"):
        validate_synth_config(broken)


def test_layout_mix_follows_the_configured_proportions(config: Config) -> None:
    rng = np.random.default_rng(0)
    counts = Counter(generate_layout(rng, config).template for _ in range(1500))

    for name in TEMPLATE_NAMES:
        assert counts[name] / 1500 == pytest.approx(config["layouts"][name], abs=0.04)


def test_value_ranges_mix_normal_extreme_and_random(config: Config) -> None:
    from tensoscan_ml.synth.layouts import sample_values

    rng = np.random.default_rng(1)
    kinds = Counter(sample_values(rng, config)[1] for _ in range(2000))

    assert kinds["normal"] / 2000 == pytest.approx(0.5, abs=0.05)
    assert kinds["extreme"] / 2000 == pytest.approx(0.3, abs=0.05)
    assert kinds["random"] / 2000 == pytest.approx(0.2, abs=0.05)


def test_every_template_renders_and_labels_are_valid_yolo(config: Config) -> None:
    for template in TEMPLATE_NAMES:
        sample = generate_sample(config, 5, "train", 3, template=template)

        assert sample.image.shape == (config["image"]["height"], config["image"]["width"], 3)
        for label in sample.labels:
            assert label.symbol in CLASSES
            assert label.cx - label.w / 2 >= 0.0 and label.cx + label.w / 2 <= 1.0 + 1e-9
            assert label.cy - label.h / 2 >= 0.0 and label.cy + label.h / 2 <= 1.0 + 1e-9
            assert len(label.line().split()) == 5


def test_negative_templates_have_no_labels_and_the_expected_status(config: Config) -> None:
    for index in range(6):
        sample = generate_sample(config, 9, "train", index, template="negative")

        assert sample.labels == []
        assert sample.meta["expected"]["status"] == "NO_READING"


def test_metadata_records_layout_values_and_render_parameters(config: Config) -> None:
    sample = generate_sample(config, 11, "val", 7, template="stacked_small_pulse")
    meta: dict[str, Any] = sample.meta

    assert meta["template"] == "stacked_small_pulse"
    assert meta["seed"] == 11
    assert meta["split"] == "val"
    assert set(meta["expected"]) == {"status", "sys", "dia", "pul"}
    assert {"yaw_deg", "pitch_deg", "roll_deg", "display_fraction"} <= set(meta["scene"])
    assert {"thickness", "slant_deg", "ghost_alpha"} <= set(meta["font"])
    assert meta["palette"]


def _covered_samples(config: Config, per_template: int) -> list[Any]:
    samples = []
    for template in COVERED_TEMPLATES:
        for index in range(per_template):
            samples.append(generate_sample(config, 21, "val", index, template=template, labels_only=True))
    return samples


def test_official_parser_reproduces_the_rendered_values_from_the_exact_labels(flat_config: Config) -> None:
    """The generator and the parser are consistent (T-B003-04): flat scenes, parser-compatible fonts, readable size."""
    checked = 0
    for sample in _covered_samples(flat_config, 80):
        meta = sample.meta
        if not (meta["covered"] and meta["readable"] and meta["parser_compatible"]):
            continue
        expected = meta["expected"]
        actual = parse_detections([label.detection() for label in sample.labels])

        assert actual.status.value == expected["status"], meta["template"]
        assert actual.values == (expected["sys"], expected["dia"], expected["pul"]), meta["template"]
        checked += 1

    assert checked >= 120


def test_parser_agreement_stays_high_under_perspective_and_rotation(config: Config) -> None:
    matches = total = 0
    for sample in _covered_samples(config, 80):
        meta = sample.meta
        complete = meta["expected"] and meta["expected"]["status"] == "COMPLETE"
        if complete and meta["readable"] and meta["parser_compatible"]:
            total += 1
            matches += int(bool(meta["parser_matches"]))

    assert total >= 80
    assert matches / total >= 0.80


def _readings_with_ones() -> list[tuple[int, int, int]]:
    return [(121, 81, 71), (131, 71, 91), (111, 61, 51)]


def _parse_flat_numbers(font: DigitFont, values: tuple[int, int, int]) -> tuple[int | None, ...]:
    """Parses three stacked right-aligned numbers drawn with `font`, labelled with the exact glyph boxes."""
    from tensoscan_ml.reading import Box, SymbolDetection
    from tensoscan_ml.synth.glyph import translate

    height_px, width_px = 480.0, 384.0
    detections = []
    glyph_h = 0.12 * height_px
    for row, value in enumerate(values):
        x = 0.30 * width_px
        for symbol in str(value):
            glyph = build_glyph(symbol, glyph_h, font.style)
            x0, y0, x1, y1 = translate(glyph, x, (0.1 + 0.2 * row) * height_px).envelope()
            detections.append(
                SymbolDetection(symbol, Box(x0 / width_px, y0 / height_px, x1 / width_px, y1 / height_px), 0.9)
            )
            x += glyph.width + font.spacing * glyph_h
    return parse_detections(detections).values


def test_a_realistic_font_keeps_a_one_with_its_left_neighbour() -> None:
    """Contract 1.1.0 (finding F1): the empty cell in front of a "1" no longer splits "81" or "121" with an ordinary
    font (stroke 0.15, cell 0.50, spacing 0.12 of the glyph height)."""
    ordinary = DigitFont(GlyphStyle(thickness=0.15, width_ratio=0.5, bevel=1.0), spacing=0.12)

    for values in _readings_with_ones():
        assert _parse_flat_numbers(ordinary, values) == values


def test_a_parser_compatible_font_keeps_numbers_with_ones_together() -> None:
    compatible = DigitFont(GlyphStyle(thickness=0.2, width_ratio=0.4, bevel=1.0, gap=0.015), spacing=0.03)

    for values in _readings_with_ones():
        assert _parse_flat_numbers(compatible, values) == values


def test_dataset_writer_produces_images_labels_metadata_and_manifest(config: Config, tmp_path: Path) -> None:
    from tensoscan_ml.data.manifest import read_manifest
    from tensoscan_ml.synth.writer import write_dataset

    info = write_dataset(tmp_path / "ds", config, {"train": 4, "val": 2}, seed=3)

    root = tmp_path / "ds"
    assert (root / "classes.txt").read_text(encoding="utf-8").split() == list(CLASSES)
    assert len(list((root / "train" / "images").glob("*.jpg"))) == 4
    assert len(list((root / "val" / "labels").glob("*.txt"))) == 2
    entries = read_manifest(root / "train" / "manifest.jsonl")
    assert [e.path for e in entries] == [f"images/{i:06d}.jpg" for i in range(4)]
    assert all(len(e.sha256) == 64 and e.labels for e in entries)
    assert info["counts"] == {"train": 4, "val": 2}
    assert len(info["dataset_hash"]) == 64
