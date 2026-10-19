"""Screen layout templates (design section 3.2, REQ-B003-06).

A layout is a flat description of one display in *display coordinates* (pixels of a nominal display plane): where each
seven-segment glyph sits (the labelled objects), plus unlabelled distractors (captions such as SYS/DIA/PUL/mmHg, a
heart icon, the memory "M"...). Nothing is rasterised here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tensoscan_ml.config import Config
from tensoscan_ml.synth.fonts import DigitFont, sample_font
from tensoscan_ml.synth.glyph import GlyphGeometry, build_glyph, translate, with_lit
from tensoscan_ml.synth.params import TEMPLATE_NAMES, Rng, chance, choose_weighted, uniform

NOMINAL_WIDTH = 600.0
COVERED_TEMPLATES = (
    "stacked_small_pulse",
    "three_rows_equal",
    "sys_dia_row_pulse_below",
    "clock_date",
    "error",
    "measuring",
    "negative",
)
_MAX_FIT_ATTEMPTS = 16
_EDGE = 0.97


@dataclass(frozen=True)
class PlacedGlyph:
    """A glyph on the display. `role` says what it is for (value, clock, date, memory, error, measuring...)."""

    symbol: str
    geometry: GlyphGeometry
    role: str
    labelled: bool = True  # False for switched-off cells that only show ghost segments


@dataclass(frozen=True)
class TextItem:
    """An unlabelled caption in a non-seven-segment font."""

    text: str
    x: float
    y: float  # top
    height: float


@dataclass(frozen=True)
class IconItem:
    kind: str  # heart | battery | memory | slash
    x: float
    y: float
    size: float


@dataclass(frozen=True)
class ExpectedReading:
    """What the official parser must produce from the exact labels of a covered layout."""

    status: str
    systolic: int | None = None
    diastolic: int | None = None
    pulse: int | None = None


@dataclass
class DisplayLayout:
    template: str
    width: float
    height: float
    font: DigitFont
    glyphs: list[PlacedGlyph] = field(default_factory=list)
    texts: list[TextItem] = field(default_factory=list)
    icons: list[IconItem] = field(default_factory=list)
    expected: ExpectedReading | None = None
    covered: bool = True
    display_on: bool = True
    has_display: bool = True
    extras: dict[str, Any] = field(default_factory=dict)


def sample_values(rng: Rng, config: Config) -> tuple[tuple[int, int, int], str]:
    """SYS, DIA and PUL shown by a display: 50 % normal, 30 % extreme but valid, 20 % random (design 3.2)."""
    kind = choose_weighted(rng, config["values"], ("normal", "extreme", "random"))
    if kind == "normal":
        return (int(rng.integers(100, 140)), int(rng.integers(60, 90)), int(rng.integers(55, 96))), kind
    if kind == "extreme":
        systolic = int(rng.integers(60, 261))
        diastolic = int(rng.integers(30, min(160, systolic - 1) + 1))
        return (systolic, diastolic, int(rng.integers(30, 221))), kind
    return (int(rng.integers(10, 300)), int(rng.integers(10, 300)), int(rng.integers(10, 300))), kind


class _Builder:
    """Places seven-segment glyphs on a display plane."""

    def __init__(self, width: float, height: float, font: DigitFont) -> None:
        self.width = width
        self.height = height
        self.font = font
        self.glyphs: list[PlacedGlyph] = []
        self.texts: list[TextItem] = []
        self.icons: list[IconItem] = []

    def advance(self, glyph_height: float) -> float:
        return build_glyph("8", glyph_height, self.font.style).width + self.font.spacing * glyph_height

    def field_width(self, cells: int, glyph_height: float) -> float:
        return cells * self.advance(glyph_height) - self.font.spacing * glyph_height

    def sequence(self, symbols: Sequence[str], left: float, top: float, glyph_height: float, role: str) -> float:
        """Places symbols one after another and returns the right edge of the last one."""
        x = left
        for symbol in symbols:
            geometry = build_glyph(symbol, glyph_height, self.font.style)
            self.glyphs.append(PlacedGlyph(symbol, translate(geometry, x, top), role))
            x += geometry.width + self.font.spacing * glyph_height
        return x - self.font.spacing * glyph_height

    def field(self, text: str, cells: int, right: float, top: float, glyph_height: float, role: str) -> float:
        """A right-aligned fixed field of `cells` cells; unused cells show switched-off segments. Returns its left."""
        advance = self.advance(glyph_height)
        left = right - self.field_width(cells, glyph_height)
        blanks = cells - len(text)
        for index in range(blanks):
            geometry = build_glyph("8", glyph_height, self.font.style)
            blank = with_lit(translate(geometry, left + index * advance, top), frozenset())
            self.glyphs.append(PlacedGlyph("8", blank, role, labelled=False))
        self.sequence(text, left + blanks * advance, top, glyph_height, role)
        return left

    def caption(self, text: str, x: float, y: float, height: float) -> None:
        self.texts.append(TextItem(text, x, y, height))

    def extent(self) -> tuple[float, float]:
        """Right and bottom edge of everything placed so far."""
        right = bottom = 0.0
        for placed in self.glyphs:
            for polygon in placed.geometry.polygons.values():
                right = max(right, float(polygon[:, 0].max()))
                bottom = max(bottom, float(polygon[:, 1].max()))
        return right, bottom

    def fits(self) -> bool:
        right, bottom = self.extent()
        return right <= _EDGE * self.width and bottom <= _EDGE * self.height


# --- distractors -----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Strip:
    """Which small items appear in the top strip. Decided before the fit loop so retries do not change them."""

    clock: bool
    date: bool
    memory: bool
    seed: int

    @classmethod
    def draw(cls, rng: Rng, layouts: Config, *, force_clock_date: bool) -> _Strip:
        return cls(
            clock=force_clock_date or chance(rng, float(layouts["clock_prob"])),
            date=force_clock_date or chance(rng, float(layouts["date_prob"])),
            memory=chance(rng, float(layouts["memory_prob"])),
            seed=int(rng.integers(0, 2**31 - 1)),
        )

    def apply(self, builder: _Builder, *, limit_height: float) -> float:
        """Places the items and returns the y of the bottom edge of the strip (0 when empty)."""
        if not (self.clock or self.date or self.memory):
            return 0.0
        return _add_clock_and_date(
            builder,
            np.random.default_rng(self.seed),
            clock=self.clock,
            date=self.date,
            memory=self.memory,
            limit_height=limit_height,
        )


def _add_clock_and_date(
    builder: _Builder, rng: Rng, *, clock: bool, date: bool, memory: bool, limit_height: float
) -> float:
    """Adds the small items of the top strip. Returns the y of the bottom edge of the strip."""
    height = builder.height
    bottom = 0.0
    if clock:
        ch = min(uniform(rng, (0.05, 0.07)) * height, limit_height)
        digits = [f"{int(rng.integers(0, 24)):02d}", f"{int(rng.integers(0, 60)):02d}"]
        symbols = [*digits[0], "colon", *digits[1]]
        width = sum(build_glyph(s, ch, builder.font.style).width + builder.font.spacing * ch for s in symbols)
        top = 0.02 * height
        builder.sequence(symbols, builder.width * 0.95 - width, top, ch, "clock")
        bottom = max(bottom, top + ch)
    if date:
        dh = min(uniform(rng, (0.04, 0.06)) * height, limit_height * 0.9)
        month, day = int(rng.integers(1, 13)), int(rng.integers(1, 29))
        symbols = [*f"{month:02d}", "dash", *f"{day:02d}"]
        top = 0.025 * height
        builder.sequence(symbols, builder.width * uniform(rng, (0.24, 0.30)), top, dh, "date")
        bottom = max(bottom, top + dh)
    if memory:
        mh = min(uniform(rng, (0.06, 0.09)) * height, limit_height * 1.2)
        top = 0.025 * height
        left = builder.width * 0.04
        builder.icons.append(IconItem("memory", left, top, mh))
        builder.sequence([str(int(rng.integers(0, 10)))], left + mh * 1.1, top, mh, "memory")
        bottom = max(bottom, top + mh)
    return bottom


def _add_captions(builder: _Builder, rows: list[tuple[str, float, float]], field_left: float, rng: Rng) -> None:
    """SYS / DIA / PUL captions to the left of the number fields, when there is room for them."""
    for text, y, row_height in rows:
        cap_height = min(0.075 * builder.height, row_height * 0.35)
        cap_width = cap_height * 0.75 * len(text)
        x = builder.width * 0.04
        if x + cap_width < field_left - 0.02 * builder.width:
            builder.caption(text, x, y + row_height * 0.1, cap_height)
    if chance(rng, 0.5):
        builder.caption("mmHg", builder.width * 0.04, builder.height * 0.46, builder.height * 0.05)


# --- templates -------------------------------------------------------------------------------------------------


def _stacked_small_pulse(
    rng: Rng, config: Config, font: DigitFont, values: tuple[int, int, int], force_clock_date: bool = False
) -> DisplayLayout:
    layouts = config["layouts"]
    width = NOMINAL_WIDTH
    height = width / uniform(rng, (1.15, 1.7))
    hs = uniform(rng, (0.24, 0.30)) * height
    dia_ratio = uniform(rng, (0.9, 1.0))
    pulse_ratio = uniform(rng, (0.42, 0.62))
    right_frac = uniform(rng, (0.60, 0.78))
    shift = uniform(rng, (0.0, 0.12))
    top_frac = uniform(rng, (0.10, 0.16))
    gap1, gap2 = uniform(rng, (0.04, 0.08)), uniform(rng, (0.02, 0.06))
    captions = chance(rng, float(layouts["captions_prob"]))
    strip = _Strip.draw(rng, layouts, force_clock_date=force_clock_date)
    for attempt in range(_MAX_FIT_ATTEMPTS):
        scale = 0.93**attempt
        builder = _Builder(width, height, font)
        h_sys, h_dia, h_pul = hs * scale, hs * scale * dia_ratio, hs * scale * pulse_ratio
        bottom = strip.apply(builder, limit_height=0.3 * h_sys)
        top = max(top_frac * height * scale, bottom + 0.02 * height)
        right = right_frac * width
        left = builder.field(str(values[0]), 3, right, top, h_sys, "sys")
        top2 = top + h_sys + gap1 * height * scale
        builder.field(str(values[1]), 3, right, top2, h_dia, "dia")
        top3 = top2 + h_dia + gap2 * height * scale
        builder.field(str(values[2]), 3, min(right + shift * width, _EDGE * width), top3, h_pul, "pul")
        if builder.fits():
            if captions:
                _add_captions(builder, [("SYS", top, h_sys), ("DIA", top2, h_dia)], left, rng)
                builder.icons.append(IconItem("heart", max(left - 0.02 * width, 0.05 * width) - 0.0, top3, h_pul * 0.8))
            return _finish("stacked_small_pulse", builder, font, values)
    raise RuntimeError("stacked layout does not fit the display")  # pragma: no cover


def _three_rows_equal(
    rng: Rng, config: Config, font: DigitFont, values: tuple[int, int, int], force_clock_date: bool = False
) -> DisplayLayout:
    layouts = config["layouts"]
    width = NOMINAL_WIDTH
    height = width / uniform(rng, (1.15, 1.7))
    hs = uniform(rng, (0.20, 0.26)) * height
    right_frac = uniform(rng, (0.55, 0.82))
    top_frac = uniform(rng, (0.08, 0.14))
    gap = uniform(rng, (0.03, 0.06))
    captions = chance(rng, float(layouts["captions_prob"]))
    strip = _Strip.draw(rng, layouts, force_clock_date=force_clock_date)
    for attempt in range(_MAX_FIT_ATTEMPTS):
        scale = 0.93**attempt
        builder = _Builder(width, height, font)
        h = hs * scale
        bottom = strip.apply(builder, limit_height=0.3 * h)
        top = max(top_frac * height * scale, bottom + 0.02 * height)
        right = right_frac * width
        tops = [top + i * (h + gap * height * scale) for i in range(3)]
        left = min(
            builder.field(str(value), 3, right, y, h, role)
            for value, y, role in zip(values, tops, ("sys", "dia", "pul"), strict=True)
        )
        if builder.fits():
            if captions:
                _add_captions(builder, [("SYS", tops[0], h), ("DIA", tops[1], h), ("PUL", tops[2], h)], left, rng)
            return _finish("three_rows_equal", builder, font, values)
    raise RuntimeError("three-row layout does not fit the display")  # pragma: no cover


def _sys_dia_row(
    rng: Rng, config: Config, font: DigitFont, values: tuple[int, int, int], force_clock_date: bool = False
) -> DisplayLayout:
    layouts = config["layouts"]
    width = NOMINAL_WIDTH
    height = width / uniform(rng, (1.15, 1.7))
    hs = uniform(rng, (0.24, 0.32)) * height
    pulse_ratio = uniform(rng, (0.45, 0.9))
    top_frac = uniform(rng, (0.10, 0.16))
    row_gap = uniform(rng, (0.03, 0.07))
    separator = chance(rng, 0.4)
    captions = chance(rng, float(layouts["captions_prob"]))
    strip = _Strip.draw(rng, layouts, force_clock_date=force_clock_date)
    for attempt in range(_MAX_FIT_ATTEMPTS):
        scale = 0.93**attempt
        builder = _Builder(width, height, font)
        h_sys = hs * scale
        h_pul = h_sys * pulse_ratio
        bottom = strip.apply(builder, limit_height=0.3 * h_sys)
        top = max(top_frac * height * scale, bottom + 0.02 * height)
        field_w = builder.field_width(3, h_sys)
        total = 2 * field_w + 0.08 * width
        start = (width - total) / 2
        builder.field(str(values[0]), 3, start + field_w, top, h_sys, "sys")
        builder.field(str(values[1]), 3, start + total, top, h_sys, "dia")
        top2 = top + h_sys + row_gap * height * scale
        pulse_right = width * uniform(rng, (0.55, 0.78)) if attempt == 0 else width * 0.62
        builder.field(str(values[2]), 3, pulse_right, top2, h_pul, "pul")
        if builder.fits():
            if separator:
                builder.icons.append(IconItem("slash", start + field_w + 0.01 * width, top, h_sys))
            if captions:
                cap = min(0.06 * height, h_sys * 0.3)
                builder.caption("SYS", start, top - cap * 1.2 if top > cap * 1.4 else top + h_sys + 1, cap)
                builder.caption(
                    "PUL", max(0.03 * width, pulse_right - builder.field_width(3, h_pul) - cap * 3), top2, cap
                )
            return _finish("sys_dia_row_pulse_below", builder, font, values)
    raise RuntimeError("side-by-side layout does not fit the display")  # pragma: no cover


def _error(rng: Rng, config: Config, font: DigitFont) -> DisplayLayout:
    del config
    width = NOMINAL_WIDTH
    height = width / uniform(rng, (1.15, 1.7))
    h = uniform(rng, (0.30, 0.42)) * height
    kind = "err" if chance(rng, 0.4) else "e_digit"
    symbols = ["E", "r", "r"] if kind == "err" else ["E", str(int(rng.integers(1, 6)))]
    for attempt in range(_MAX_FIT_ATTEMPTS):
        builder = _Builder(width, height, font)
        size = h * 0.93**attempt
        total = sum(build_glyph(s, size, font.style).width + font.spacing * size for s in symbols)
        builder.sequence(
            symbols, (width - total) / 2, (height - size) / 2 + uniform(rng, (-0.05, 0.05)) * height, size, "error"
        )
        if builder.fits():
            layout = _finish("error", builder, font, None)
            layout.expected = ExpectedReading("DEVICE_ERROR")
            layout.extras["error_kind"] = kind
            return layout
    raise RuntimeError("error layout does not fit the display")  # pragma: no cover


def _measuring(rng: Rng, config: Config, font: DigitFont) -> DisplayLayout:
    del config
    width = NOMINAL_WIDTH
    height = width / uniform(rng, (1.15, 1.7))
    rows = int(rng.integers(1, 4))
    h = uniform(rng, (0.36, 0.46)) / (1 if rows == 1 else rows * 0.75) * height
    h = min(h, 0.42 * height)
    for attempt in range(_MAX_FIT_ATTEMPTS):
        builder = _Builder(width, height, font)
        size = h * 0.93**attempt
        gap = 0.05 * height
        block = rows * size + (rows - 1) * gap
        top = (height - block) / 2
        right = width * uniform(rng, (0.55, 0.75)) if attempt == 0 else width * 0.62
        for row in range(rows):
            builder.sequence(
                ["dash"] * 3, right - builder.field_width(3, size), top + row * (size + gap), size, "measuring"
            )
        if builder.fits():
            layout = _finish("measuring", builder, font, None)
            layout.expected = ExpectedReading("MEASURING")
            layout.extras["rows"] = rows
            return layout
    raise RuntimeError("measuring layout does not fit the display")  # pragma: no cover


def _negative(rng: Rng, config: Config, font: DigitFont) -> DisplayLayout:
    del config
    width = NOMINAL_WIDTH
    height = width / uniform(rng, (1.15, 1.7))
    builder = _Builder(width, height, font)
    kind = "display_off" if chance(rng, 0.6) else "no_display"
    if kind == "display_off":
        h = uniform(rng, (0.22, 0.3)) * height
        top = uniform(rng, (0.1, 0.16)) * height
        for row in range(2):
            builder.field("", 3, width * 0.7, top + row * h * 1.15, h, "off")
    layout = _finish("negative", builder, font, None)
    layout.expected = ExpectedReading("NO_READING")
    layout.display_on = False
    layout.has_display = kind == "display_off"
    layout.extras["negative_kind"] = kind
    return layout


def _out_of_coverage(rng: Rng, config: Config, font: DigitFont) -> DisplayLayout:
    del config
    width = NOMINAL_WIDTH
    height = width / uniform(rng, (1.15, 1.7))
    kind = "lone_number" if chance(rng, 0.5) else "four_digits"
    text = str(int(rng.integers(10, 300))) if kind == "lone_number" else f"{int(rng.integers(1000, 10000))}"
    h = uniform(rng, (0.34, 0.5)) * height
    for attempt in range(_MAX_FIT_ATTEMPTS):
        builder = _Builder(width, height, font)
        size = h * 0.93**attempt
        total = builder.field_width(len(text), size)
        builder.sequence(text, (width - total) / 2, (height - size) / 2, size, "other")
        if builder.fits():
            layout = _finish("out_of_coverage", builder, font, None)
            layout.covered = False
            layout.expected = None
            layout.extras["coverage_kind"] = kind
            return layout
    raise RuntimeError("out-of-coverage layout does not fit the display")  # pragma: no cover


# --- assembly --------------------------------------------------------------------------------------------------


def _finish(template: str, builder: _Builder, font: DigitFont, values: tuple[int, int, int] | None) -> DisplayLayout:
    layout = DisplayLayout(
        template=template,
        width=builder.width,
        height=builder.height,
        font=font,
        glyphs=builder.glyphs,
        texts=builder.texts,
        icons=builder.icons,
        extras={
            "clock": any(g.role == "clock" for g in builder.glyphs),
            "date": any(g.role == "date" for g in builder.glyphs),
            "memory": any(g.role == "memory" for g in builder.glyphs),
        },
    )
    if values is not None:
        layout.expected = ExpectedReading("COMPLETE", *values)
        layout.extras["values"] = list(values)
    return layout


_COVERED_BUILDERS: dict[str, Callable[..., DisplayLayout]] = {
    "stacked_small_pulse": _stacked_small_pulse,
    "three_rows_equal": _three_rows_equal,
    "sys_dia_row_pulse_below": _sys_dia_row,
}


def choose_template(rng: Rng, config: Config) -> str:
    return choose_weighted(rng, config["layouts"], TEMPLATE_NAMES)


def generate_layout(rng: Rng, config: Config, template: str | None = None) -> DisplayLayout:
    """Draws a font, values and a layout of `template` (random according to the config when omitted)."""
    template = template or choose_template(rng, config)
    font = sample_font(rng, config)
    values, value_kind = sample_values(rng, config)
    if template == "clock_date":
        base = ("stacked_small_pulse", "three_rows_equal", "sys_dia_row_pulse_below")[int(rng.integers(0, 3))]
        layout = _COVERED_BUILDERS[base](rng, config, font, values, force_clock_date=True)
        layout.template = "clock_date"
        layout.extras["base_template"] = base
    elif template in _COVERED_BUILDERS:
        layout = _COVERED_BUILDERS[template](rng, config, font, values)
    elif template == "error":
        layout = _error(rng, config, font)
    elif template == "measuring":
        layout = _measuring(rng, config, font)
    elif template == "negative":
        layout = _negative(rng, config, font)
    elif template == "out_of_coverage":
        layout = _out_of_coverage(rng, config, font)
    else:
        raise ValueError(f"Unknown layout template '{template}'")
    layout.extras["value_kind"] = value_kind
    return layout
