"""Seven-segment glyph geometry (design section 3.1).

```
 -a-
|f  |b      Segments a-g are polygons (bevelled hexagons, rectangles or rounded "stadiums").
 -g-        Parameters: height H, width ratio, thickness t, gap between segments and slant.
|e  |c      colon = 2 dots, dash = g, E = a d e f g, r = e g.
 -d-
```

The geometry lives in a *cell* whose top-left corner is the origin, in pixels. Nothing here draws: the polygons are used
both to rasterise the glyph and to compute its exact label box, so image and label cannot disagree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

Polygon = NDArray[np.float64]

SEGMENT_NAMES: tuple[str, ...] = ("a", "b", "c", "d", "e", "f", "g")

#: Lit segments per digit (standard table). `6`, `9` and `7` have optional tails/strokes, see `GlyphStyle`.
DIGIT_SEGMENTS: dict[str, str] = {
    "0": "abcdef",
    "1": "bc",
    "2": "abdeg",
    "3": "abcdg",
    "4": "bcfg",
    "5": "acdfg",
    "6": "acdefg",
    "7": "abc",
    "8": "abcdefg",
    "9": "abcdfg",
}
#: Letters and signs of the model contract (class 11-13); the colon is made of two dots.
SYMBOL_SEGMENTS: dict[str, str] = {**DIGIT_SEGMENTS, "dash": "g", "E": "adefg", "r": "eg"}
COLON = "colon"
DOT_NAMES: tuple[str, ...] = ("dot_top", "dot_bottom")

_ARC_STEPS = 6


@dataclass(frozen=True)
class GlyphStyle:
    """Shape parameters of one display font (REQ-B003-05)."""

    thickness: float = 0.15  # t / H, 0.10-0.22
    gap: float = 0.025  # gap between segments / H, 0.01-0.04
    slant_deg: float = 0.0  # 0-12 degrees, leaning right
    bevel: float = 1.0  # 0 = rectangular ends, 1 = hexagonal ends
    rounded: bool = False  # stadium-shaped segments (overrides bevel)
    width_ratio: float = 0.56  # W / H of a digit cell
    ghost_alpha: float = 0.12  # opacity of the switched-off segments, 0-0.3
    six_tail: bool = True  # 6 drawn with the top segment
    nine_tail: bool = True  # 9 drawn with the bottom segment
    seven_f: bool = False  # 7 drawn with the upper-left stroke

    def segments_of(self, symbol: str) -> str:
        """Lit segments of `symbol` for this font."""
        if symbol == "6" and not self.six_tail:
            return "cdefg"
        if symbol == "9" and not self.nine_tail:
            return "abcfg"
        if symbol == "7" and self.seven_f:
            return "abcf"
        return SYMBOL_SEGMENTS[symbol]


@dataclass(frozen=True)
class GlyphGeometry:
    """Polygons of one glyph, in cell pixels (origin at the top-left corner of the cell)."""

    symbol: str
    height: float
    width: float  # advance width of the cell (includes the slant overhang)
    polygons: dict[str, Polygon] = field(repr=False)
    lit: frozenset[str] = field(default_factory=frozenset)

    @property
    def lit_polygons(self) -> list[Polygon]:
        return [self.polygons[name] for name in self.polygons if name in self.lit]

    @property
    def unlit_polygons(self) -> list[Polygon]:
        return [self.polygons[name] for name in self.polygons if name not in self.lit]

    def envelope(self) -> tuple[float, float, float, float]:
        """Tight box `(x0, y0, x1, y1)` around the lit segments: this is the label before any margin."""
        points = np.concatenate(self.lit_polygons)
        return (
            float(points[:, 0].min()),
            float(points[:, 1].min()),
            float(points[:, 0].max()),
            float(points[:, 1].max()),
        )


def _hexagon_bar(start: float, end: float, centre: float, half: float, bevel: float, *, vertical: bool) -> Polygon:
    """Bar between its tips `start` and `end`, centred on `centre`: a hexagon, or a rectangle at bevel 0."""
    flat = half * (1.0 - bevel)
    inset = half * bevel
    points = np.array(
        [
            (start, centre - flat),
            (start + inset, centre - half),
            (end - inset, centre - half),
            (end, centre - flat),
            (end, centre + flat),
            (end - inset, centre + half),
            (start + inset, centre + half),
            (start, centre + flat),
        ],
        dtype=np.float64,
    )
    return points[:, ::-1].copy() if vertical else points


def _mitred_segments(width: float, height: float, thickness: float, gap: float, bevel: float) -> dict[str, Polygon]:
    """Standard seven-segment outline: the silhouette is the full cell, the corners are cut at 45 degrees.

    `bevel` 1 gives the mitred joints of LCD fonts (the vertical bars reach the cell edge, so "1" is as tall as "8");
    `bevel` 0 gives plain rectangles whose vertical bars start below the horizontal ones.
    """
    half = thickness / 2.0
    gp = gap * 0.7  # distance of each bar to the cell edge along the mitre
    mid = height / 2.0
    w, h, t = width, height, thickness
    edge_top = gp + (1.0 - bevel) * t  # y of the outer corner of the upper vertical bars
    flat = half * (1.0 - bevel)
    inset = half * bevel

    def horizontal(y_outer: float, y_inner: float) -> Polygon:
        shift = bevel * t
        return np.array(
            [(gp, y_outer), (w - gp, y_outer), (w - gp - shift, y_inner), (gp + shift, y_inner)], dtype=np.float64
        )

    def upper_right() -> Polygon:
        return np.array(
            [
                (w, edge_top),
                (w - t, t + gp),
                (w - t, mid - gap - inset),
                (w - half - flat, mid - gap),
                (w - half + flat, mid - gap),
                (w, mid - gap - inset),
            ],
            dtype=np.float64,
        )

    b = upper_right()
    f = b * np.array([-1.0, 1.0]) + np.array([w, 0.0])
    c = b * np.array([1.0, -1.0]) + np.array([0.0, h])
    e = f * np.array([1.0, -1.0]) + np.array([0.0, h])
    return {
        "a": horizontal(0.0, t),
        "b": b,
        "c": c,
        "d": horizontal(h, h - t),
        "e": e,
        "f": f,
        "g": _hexagon_bar(t + gap, w - t - gap, mid, half, bevel, vertical=False),
    }


def _stadium(start: float, end: float, centre: float, half: float, *, vertical: bool) -> Polygon:
    """Rounded segment (a rectangle with semicircular ends) along one axis."""
    rx = min(half, (end - start) / 2.0)
    points: list[tuple[float, float]] = []
    for step in range(_ARC_STEPS + 1):  # left cap: top -> tip -> bottom
        angle = math.pi / 2 + math.pi * step / _ARC_STEPS
        points.append((start + rx + rx * math.cos(angle), centre - half * math.sin(angle)))
    for step in range(_ARC_STEPS + 1):  # right cap: bottom -> tip -> top
        angle = -math.pi / 2 + math.pi * step / _ARC_STEPS
        points.append((end - rx + rx * math.cos(angle), centre - half * math.sin(angle)))
    array = np.array(points, dtype=np.float64)
    return array[:, ::-1].copy() if vertical else array


def _apply_slant(polygon: Polygon, height: float, slant_deg: float) -> Polygon:
    """Shear to the right around the bottom edge, as italic LCD fonts do."""
    if slant_deg == 0.0:
        return polygon
    shear = math.tan(math.radians(slant_deg))
    sheared = polygon.copy()
    sheared[:, 0] += (height - sheared[:, 1]) * shear
    return sheared


def build_glyph(symbol: str, height: float, style: GlyphStyle) -> GlyphGeometry:
    """Polygons of `symbol` in a cell of the given pixel `height`."""
    if symbol != COLON and symbol not in SYMBOL_SEGMENTS:
        raise ValueError(f"Unknown symbol '{symbol}'")
    shear = math.tan(math.radians(style.slant_deg))
    thickness = style.thickness * height
    half = thickness / 2.0
    gap = style.gap * height

    if symbol == COLON:
        dot = max(thickness, 0.12 * height)
        width = dot * 1.6
        centre_x = width / 2.0
        polygons: dict[str, Polygon] = {}
        for name, fraction in zip(DOT_NAMES, (0.17, 0.83), strict=True):
            cy = height * fraction
            square = np.array(
                [
                    (centre_x - dot / 2, cy - dot / 2),
                    (centre_x + dot / 2, cy - dot / 2),
                    (centre_x + dot / 2, cy + dot / 2),
                    (centre_x - dot / 2, cy + dot / 2),
                ],
                dtype=np.float64,
            )
            polygons[name] = _apply_slant(square, height, style.slant_deg)
        overhang = height * shear
        return GlyphGeometry(symbol, height, width + overhang, polygons, frozenset(DOT_NAMES))

    cell_width = style.width_ratio * height
    if style.rounded:
        # Stadium bars that never touch: the horizontal bars sit between the vertical ones.
        x_left, x_right = half, cell_width - half
        y_mid = height / 2.0
        h_start, h_end = thickness + gap, cell_width - thickness - gap
        raw: dict[str, Polygon] = {
            "a": _stadium(h_start, h_end, half, half, vertical=False),
            "g": _stadium(h_start, h_end, y_mid, half, vertical=False),
            "d": _stadium(h_start, h_end, height - half, half, vertical=False),
            "f": _stadium(0.0, y_mid - gap, x_left, half, vertical=True),
            "b": _stadium(0.0, y_mid - gap, x_right, half, vertical=True),
            "e": _stadium(y_mid + gap, height, x_left, half, vertical=True),
            "c": _stadium(y_mid + gap, height, x_right, half, vertical=True),
        }
    else:
        raw = _mitred_segments(cell_width, height, thickness, gap, style.bevel)
    polygons = {name: _apply_slant(raw[name], height, style.slant_deg) for name in SEGMENT_NAMES}
    return GlyphGeometry(
        symbol=symbol,
        height=height,
        width=cell_width + height * shear,
        polygons=polygons,
        lit=frozenset(style.segments_of(symbol)),
    )


def translate(geometry: GlyphGeometry, dx: float, dy: float) -> GlyphGeometry:
    """The same glyph moved by `(dx, dy)`."""
    offset = np.array([dx, dy], dtype=np.float64)
    return GlyphGeometry(
        geometry.symbol,
        geometry.height,
        geometry.width,
        {name: polygon + offset for name, polygon in geometry.polygons.items()},
        geometry.lit,
    )


def with_lit(geometry: GlyphGeometry, lit: frozenset[str]) -> GlyphGeometry:
    """The same glyph with another set of lit segments (used for switched-off cells)."""
    return GlyphGeometry(geometry.symbol, geometry.height, geometry.width, geometry.polygons, lit)
