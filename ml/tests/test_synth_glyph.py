"""T-B003-03: seven-segment geometry. Segments per class and label box versus rendered mask (CA-B003-02)."""

from __future__ import annotations

import numpy as np
import pytest

from tensoscan_ml.contract import CLASSES
from tensoscan_ml.synth.glyph import (
    DIGIT_SEGMENTS,
    DOT_NAMES,
    GlyphStyle,
    build_glyph,
    translate,
    with_lit,
)
from tensoscan_ml.synth.raster import polygon_mask

STYLES = {
    "hexagonal": GlyphStyle(thickness=0.16, bevel=1.0),
    "rectangular": GlyphStyle(thickness=0.2, bevel=0.0, width_ratio=0.5),
    "rounded": GlyphStyle(thickness=0.14, rounded=True),
    "slanted": GlyphStyle(thickness=0.12, bevel=0.5, slant_deg=12.0, gap=0.04),
    "thin": GlyphStyle(thickness=0.10, gap=0.01, width_ratio=0.62),
}
LABEL_MARGIN = 0.02  # per side, relative to the box size (the generator's default)


@pytest.mark.parametrize(
    ("symbol", "segments"),
    [
        ("0", "abcdef"),
        ("1", "bc"),
        ("2", "abdeg"),
        ("3", "abcdg"),
        ("4", "bcfg"),
        ("5", "acdfg"),
        ("6", "acdefg"),
        ("7", "abc"),
        ("8", "abcdefg"),
        ("9", "abcdfg"),
        ("dash", "g"),
        ("E", "adefg"),
        ("r", "eg"),
    ],
)
def test_lit_segments_of_each_class_follow_the_standard_table(symbol: str, segments: str) -> None:
    glyph = build_glyph(symbol, 100.0, GlyphStyle())

    assert glyph.lit == frozenset(segments)
    assert set(glyph.polygons) == set("abcdefg")


def test_colon_is_two_dots_and_every_contract_class_can_be_built() -> None:
    colon = build_glyph("colon", 100.0, GlyphStyle())

    assert colon.lit == frozenset(DOT_NAMES)
    for symbol in CLASSES:
        assert build_glyph(symbol, 80.0, GlyphStyle()).lit


def test_digit_table_has_the_ten_digits() -> None:
    assert sorted(DIGIT_SEGMENTS) == [str(d) for d in range(10)]


def test_font_variants_change_the_lit_segments_of_6_7_and_9() -> None:
    style = GlyphStyle(six_tail=False, nine_tail=False, seven_f=True)

    assert build_glyph("6", 100.0, style).lit == frozenset("cdefg")
    assert build_glyph("9", 100.0, style).lit == frozenset("abcfg")
    assert build_glyph("7", 100.0, style).lit == frozenset("abcf")


def test_unknown_symbol_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown symbol"):
        build_glyph("x", 100.0, GlyphStyle())


def test_slant_leans_the_top_to_the_right_and_keeps_the_bottom_in_place() -> None:
    straight = build_glyph("8", 100.0, GlyphStyle(slant_deg=0.0))
    leaning = build_glyph("8", 100.0, GlyphStyle(slant_deg=12.0))

    top_shift = leaning.polygons["a"][:, 0].max() - straight.polygons["a"][:, 0].max()
    bottom_shift = leaning.polygons["d"][:, 0].min() - straight.polygons["d"][:, 0].min()

    assert top_shift == pytest.approx(100.0 * np.tan(np.radians(12.0)), rel=0.1)
    assert abs(bottom_shift) < 2.0
    assert leaning.width > straight.width


def _box_iou_with_mask(glyph_polygons: list[np.ndarray], box: tuple[float, float, float, float], size: int) -> float:
    """IoU between a label box and the bounding box of the rasterised glyph mask."""
    mask = polygon_mask((size, size), glyph_polygons) > 127
    ys, xs = np.nonzero(mask)
    mask_box = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
    inter_w = max(0.0, min(box[2], mask_box[2]) - max(box[0], mask_box[0]))
    inter_h = max(0.0, min(box[3], mask_box[3]) - max(box[1], mask_box[1]))
    inter = inter_w * inter_h
    area = (box[2] - box[0]) * (box[3] - box[1]) + (mask_box[2] - mask_box[0]) * (mask_box[3] - mask_box[1]) - inter
    return inter / area


@pytest.mark.parametrize("style_name", list(STYLES))
@pytest.mark.parametrize("symbol", CLASSES)
def test_label_box_hugs_the_rendered_glyph_with_iou_above_090(symbol: str, style_name: str) -> None:
    style = STYLES[style_name]
    glyph = translate(build_glyph(symbol, 400.0, style), 20.0, 20.0)
    x0, y0, x1, y1 = glyph.envelope()
    pad_x, pad_y = LABEL_MARGIN * (x1 - x0), LABEL_MARGIN * (y1 - y0)
    label = (x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y)

    actual = _box_iou_with_mask(glyph.lit_polygons, label, 460)

    assert actual >= 0.90


def test_envelope_of_a_glyph_is_exactly_the_rendered_extent() -> None:
    glyph = translate(build_glyph("8", 160.0, GlyphStyle(thickness=0.16)), 10.0, 10.0)
    x0, y0, x1, y1 = glyph.envelope()

    actual = _box_iou_with_mask(glyph.lit_polygons, (x0, y0, x1, y1), 220)

    assert actual >= 0.98


def test_switched_off_glyph_has_no_lit_polygons_but_keeps_its_ghost_segments() -> None:
    glyph = with_lit(build_glyph("8", 100.0, GlyphStyle()), frozenset())

    assert glyph.lit_polygons == []
    assert len(glyph.unlit_polygons) == 7


def test_one_is_as_tall_as_eight_in_a_mitred_font() -> None:
    style = GlyphStyle(thickness=0.16, bevel=1.0)
    eight = build_glyph("8", 100.0, style).envelope()
    one = build_glyph("1", 100.0, style).envelope()

    assert (one[3] - one[1]) / (eight[3] - eight[1]) > 0.95
