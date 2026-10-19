"""Display styles: glyph fonts (thickness, slant, bevel, ghost segments) and LCD/LED palettes (REQ-B003-05/06)."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from tensoscan_ml.config import Config
from tensoscan_ml.synth.glyph import GlyphStyle
from tensoscan_ml.synth.params import Rng, chance, uniform

Color = tuple[int, int, int]

#: The official parser joins two glyphs only when the horizontal gap is at most 0.35 of the glyph height, both measured
#: in coordinates normalised to the ROI. In pixels that is 0.35 x (ROI width / ROI height) of the glyph height, which
#: is 0.28 for the app's 4:5 ROI. Glyphs whose lit segments do not reach the cell edge ("1", "3", "7") leave a larger
#: gap to their left neighbour. A font is *parser-compatible* when its widest such gap stays under the limit with a
#: safety margin for perspective (finding F1 of the README).
PARSER_GAP_SAFETY = 0.92
MIN_HEIGHT_RATIO = 0.78  # the parser joins glyphs whose heights differ by at most 1.35
PARSER_MAX_GAP_FACTOR = 0.35


def parser_gap_limit(aspect: float) -> float:
    """Largest gap/height ratio (in pixels) the parser accepts for an image of the given width/height."""
    return PARSER_MAX_GAP_FACTOR * aspect * PARSER_GAP_SAFETY


@dataclass(frozen=True)
class DigitFont:
    """A glyph style plus the horizontal spacing of the digits of a number."""

    style: GlyphStyle
    spacing: float  # space between two digit cells / glyph height
    parser_compatible: bool = False  # set by `sample_font` for the image aspect of the dataset

    @property
    def min_height_ratio(self) -> float:
        """Height of the shortest glyph ("1", "4") relative to the cell: bevelled fonts reach the cell edge."""
        style = self.style
        if style.rounded:
            return 1.0
        return 1.0 - 2.0 * (0.7 * style.gap + (1.0 - style.bevel) * style.thickness)

    @property
    def max_gap_ratio(self) -> float:
        """Widest gap between the lit box of a digit and its left neighbour, per unit of glyph height."""
        style = self.style
        before_one = style.width_ratio + self.spacing - style.thickness
        slant = math.tan(math.radians(style.slant_deg)) * (1.0 - style.thickness)
        before_three_or_seven = self.spacing + style.thickness + style.gap + slant
        return max(before_one, before_three_or_seven)

    def is_parser_compatible(self, limit: float) -> bool:
        """The parser needs similar heights (ratio <= 1.35) and gaps under `limit` times the glyph height."""
        shortest = self.min_height_ratio
        return shortest >= MIN_HEIGHT_RATIO and self.max_gap_ratio / shortest <= limit


@dataclass(frozen=True)
class DisplayPalette:
    """Colours of one display technology (REQ-B003-06): grey or green LCD, backlit LCD and LED."""

    kind: str
    panel: Color
    lit: Color
    bezel: Color
    emissive: bool  # LED / backlit: lit segments glow and ghost segments are dim copies of the lit colour
    ghost_color: Color


def sample_font(rng: Rng, config: Config) -> DigitFont:
    """Draws a font; `parser_compatible_prob` of them are built so that the parser can join a "1" to its neighbour."""
    image = config["image"]
    limit = parser_gap_limit(image["width"] / image["height"])
    if chance(rng, float(config["glyph"]["parser_compatible_prob"])):
        font = _compatible_font(rng, config, limit)
        return replace(font, parser_compatible=True)
    font = _draw_font(rng, config)
    for _ in range(50):
        if not font.is_parser_compatible(limit):
            break
        font = _draw_font(rng, config)
    return replace(font, parser_compatible=font.is_parser_compatible(limit))


def _compatible_font(rng: Rng, config: Config, limit: float) -> DigitFont:
    """Chunky strokes and tight spacing, found by rejection sampling inside a window known to contain such fonts."""
    low, high = config["glyph"]["width_ratio"]
    base = _draw_font(rng, config)
    for _ in range(300):
        style = replace(
            base.style,
            thickness=uniform(rng, (0.16, 0.22)),
            gap=uniform(rng, config["glyph"]["gap"]),
            slant_deg=uniform(rng, (0.0, 5.0)) if chance(rng, 0.4) else 0.0,
            width_ratio=uniform(rng, (low, min(high, 0.50))),
            bevel=uniform(rng, (0.5, 1.0)),
        )
        font = DigitFont(style, spacing=uniform(rng, (0.03, 0.08)))
        if font.is_parser_compatible(limit):
            return font
    safe = replace(
        base.style, thickness=0.2, gap=0.015, slant_deg=0.0, width_ratio=low, bevel=1.0, rounded=False
    )  # pragma: no cover
    return DigitFont(safe, spacing=0.03)  # pragma: no cover


def _draw_font(rng: Rng, config: Config) -> DigitFont:
    glyph = config["glyph"]
    style = GlyphStyle(
        thickness=uniform(rng, glyph["thickness"]),
        gap=uniform(rng, glyph["gap"]),
        slant_deg=uniform(rng, glyph["slant_deg"]) if chance(rng, 0.5) else 0.0,
        bevel=uniform(rng, glyph["bevel"]),
        rounded=chance(rng, float(glyph["rounded_prob"])),
        width_ratio=uniform(rng, glyph["width_ratio"]),
        ghost_alpha=uniform(rng, glyph["ghost_alpha"]),
        six_tail=chance(rng, float(glyph["six_tail_prob"])),
        nine_tail=chance(rng, float(glyph["nine_tail_prob"])),
        seven_f=chance(rng, float(glyph["seven_f_prob"])),
    )
    return DigitFont(style, spacing=uniform(rng, glyph["digit_spacing"]))


def _jitter(rng: Rng, base: Color, amount: int) -> Color:
    r, g, b = (min(255, max(0, c + int(rng.integers(-amount, amount + 1)))) for c in base)
    return (r, g, b)


def sample_palette(rng: Rng) -> DisplayPalette:
    """Draws a display technology: grey/green LCD with dark segments, backlit LCD (blue/white) or LED."""
    kind = ("lcd_gray", "lcd_green", "lcd_backlit", "backlit_inverted", "led_red", "led_green", "led_orange")[
        int(rng.choice(7, p=[0.28, 0.20, 0.12, 0.10, 0.12, 0.08, 0.10]))
    ]
    bezel = _jitter(rng, (30, 32, 36), 18)
    if kind == "lcd_gray":
        panel = _jitter(rng, (184, 190, 178), 14)
        lit = _jitter(rng, (32, 34, 36), 12)
        return DisplayPalette(kind, panel, lit, bezel, emissive=False, ghost_color=lit)
    if kind == "lcd_green":
        panel = _jitter(rng, (168, 200, 140), 14)
        lit = _jitter(rng, (34, 48, 30), 10)
        return DisplayPalette(kind, panel, lit, bezel, emissive=False, ghost_color=lit)
    if kind == "lcd_backlit":
        panel = _jitter(rng, (150, 205, 235), 14)
        lit = _jitter(rng, (24, 40, 64), 10)
        return DisplayPalette(kind, panel, lit, bezel, emissive=False, ghost_color=lit)
    if kind == "backlit_inverted":
        panel = _jitter(rng, (16, 52, 140), 14)
        lit = _jitter(rng, (236, 244, 255), 10)
        return DisplayPalette(kind, panel, lit, bezel, emissive=True, ghost_color=lit)
    panel = _jitter(rng, (8, 8, 10), 5)
    lit = {"led_red": (255, 30, 24), "led_green": (60, 255, 90), "led_orange": (255, 150, 20)}[kind]
    return DisplayPalette(kind, panel, _jitter(rng, lit, 12), bezel, emissive=True, ghost_color=lit)
