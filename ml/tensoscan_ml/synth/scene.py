"""Display, casing, background and camera (design section 3.3, REQ-B003-06/07).

`render_display` rasterises one layout in a given palette. `compose_scene` puts the display in its casing, projects the
plane with a random camera (yaw/pitch up to 25 degrees, roll up to 8) over a procedural background, and **transforms
the glyph polygons with the same homography**, so the labels hug the glyphs that were actually drawn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray

from tensoscan_ml.config import Config
from tensoscan_ml.synth.fonts import DisplayPalette
from tensoscan_ml.synth.layouts import DisplayLayout, IconItem, TextItem
from tensoscan_ml.synth.params import Rng, uniform
from tensoscan_ml.synth.raster import as_float32, as_uint8, blend, fill_polygons, polygon_mask

FloatImage = NDArray[np.float32]
_CASINGS = ((236, 236, 230), (204, 205, 210), (30, 31, 34), (44, 64, 112), (226, 216, 192), (150, 160, 170))
_FONTS = (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX)


@dataclass(frozen=True)
class GlyphLabel:
    """Visible box of one glyph in the final image (pixels) and the pixels it covers."""

    symbol: str
    role: str
    box: tuple[float, float, float, float]  # tight x0, y0, x1, y1 around the lit segments, clipped to the image
    visible: float  # fraction of the unclipped box that lies inside the image


@dataclass
class Scene:
    image: FloatImage  # RGB, 0..1, final resolution
    labels: list[GlyphLabel]
    params: dict[str, object] = field(default_factory=dict)
    masks: list[NDArray[np.uint8]] = field(default_factory=list)  # per labelled glyph, only when requested


# --- display ---------------------------------------------------------------------------------------------------


def render_display(layout: DisplayLayout, palette: DisplayPalette, scale: float, rng: Rng) -> FloatImage:
    """The flat display panel with all its glyphs, captions and icons, as an RGB float image (0..255)."""
    width, height = round(layout.width * scale), round(layout.height * scale)
    panel = np.array(palette.panel, dtype=np.float32)
    if not layout.display_on:
        panel = panel * (0.45 if not palette.emissive else 0.5)
    image = np.ones((height, width, 3), dtype=np.float32) * panel
    gradient = np.linspace(1.0 + 0.05, 1.0 - 0.05, height, dtype=np.float32)[:, None, None]
    image *= gradient
    if not palette.emissive:
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        edge = np.minimum(np.minimum(xx, width - xx), np.minimum(yy, height - yy)) / max(width, height)
        image *= (0.90 + 0.10 * np.clip(edge * 12.0, 0.0, 1.0))[..., None]

    ghost_alpha = layout.font.style.ghost_alpha
    unlit = np.zeros((height, width), dtype=np.uint8)
    lit = np.zeros((height, width), dtype=np.uint8)
    for placed in layout.glyphs:
        fill_polygons(unlit, placed.geometry.unlit_polygons, 255, scale)
        fill_polygons(lit, placed.geometry.lit_polygons, 255, scale)
    blend(image, unlit, palette.ghost_color, ghost_alpha)
    blend(image, lit, palette.lit, 1.0)
    if palette.emissive and lit.any():
        radius = max(1.0, 0.012 * height)
        glow = as_uint8(cv2.GaussianBlur(lit, (0, 0), radius * 2.0))
        blend(image, glow, palette.lit, 0.55)
        blend(image, lit, palette.lit, 1.0)

    ink = tuple(int(c * (0.75 if palette.emissive else 1.0)) for c in palette.lit)
    byte_image = np.clip(image, 0, 255).astype(np.uint8)
    for text in layout.texts:
        _draw_text(byte_image, text, scale, ink, rng)
    for icon in layout.icons:
        _draw_icon(byte_image, icon, scale, ink)
    return as_float32(byte_image)


def _draw_text(image: NDArray[np.uint8], item: TextItem, scale: float, color: tuple[int, ...], rng: Rng) -> None:
    font = _FONTS[int(rng.integers(0, len(_FONTS)))]
    thickness = max(1, round(item.height * scale / 14))
    (_, cap_height), _ = cv2.getTextSize("H", font, 1.0, thickness)
    font_scale = item.height * scale / max(cap_height, 1)
    origin = (round(item.x * scale), round((item.y + item.height) * scale))
    cv2.putText(image, item.text, origin, font, font_scale, color, thickness, cv2.LINE_AA)


def _draw_icon(image: NDArray[np.uint8], icon: IconItem, scale: float, color: tuple[int, ...]) -> None:
    x, y, size = icon.x * scale, icon.y * scale, icon.size * scale
    if icon.kind == "heart":
        t = np.linspace(0, 2 * math.pi, 40)
        hx = 16 * np.sin(t) ** 3
        hy = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
        pts = np.stack([x + size * 0.5 + hx * size / 34, y + size * 0.5 + hy * size / 34], axis=1)
        fill_polygons(image, [pts], color)
    elif icon.kind == "battery":
        cv2.rectangle(
            image, (round(x), round(y)), (round(x + size * 1.6), round(y + size * 0.8)), color, max(1, round(size / 14))
        )
        cv2.rectangle(
            image,
            (round(x + size * 1.6), round(y + size * 0.25)),
            (round(x + size * 1.75), round(y + size * 0.55)),
            color,
            -1,
        )
    elif icon.kind == "memory":
        cv2.putText(
            image,
            "M",
            (round(x), round(y + size)),
            cv2.FONT_HERSHEY_SIMPLEX,
            size / 22.0,
            color,
            max(1, round(size / 12)),
            cv2.LINE_AA,
        )
    elif icon.kind == "slash":
        cv2.line(
            image,
            (round(x), round(y + size)),
            (round(x + size * 0.25), round(y)),
            color,
            max(2, round(size / 18)),
            cv2.LINE_AA,
        )


# --- casing and background -------------------------------------------------------------------------------------


def _rounded_rect(x0: float, y0: float, x1: float, y1: float, radius: float) -> NDArray[np.float64]:
    points: list[tuple[float, float]] = []
    for cx, cy, start in (
        (x1 - radius, y0 + radius, -90),
        (x1 - radius, y1 - radius, 0),
        (x0 + radius, y1 - radius, 90),
        (x0 + radius, y0 + radius, 180),
    ):
        for step in range(7):
            angle = math.radians(start + 15 * step)
            points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return np.array(points, dtype=np.float64)


def _smooth_noise(rng: Rng, height: int, width: int, cells: int) -> NDArray[np.float32]:
    small = rng.random((max(2, height // cells), max(2, width // cells))).astype(np.float32)
    return as_float32(cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC))


def background(rng: Rng, width: int, height: int) -> FloatImage:
    """Procedural table, cloth, wood or dark surface (0..255, RGB)."""
    kind = ("table", "cloth", "wood", "dark", "tile")[int(rng.integers(0, 5))]
    base = np.array([rng.integers(70, 235), rng.integers(70, 235), rng.integers(70, 235)], dtype=np.float32)
    if kind == "wood":
        base = np.array([rng.integers(110, 190), rng.integers(70, 120), rng.integers(30, 80)], dtype=np.float32)
    if kind == "dark":
        base = base * 0.25
    low = _smooth_noise(rng, height, width, 64)
    fine = _smooth_noise(rng, height, width, 3)
    texture = 0.85 + 0.30 * low + 0.06 * (fine - 0.5)
    if kind == "wood":
        yy = np.arange(height, dtype=np.float32)[:, None]
        texture = texture * (0.9 + 0.1 * np.sin(yy * rng.uniform(0.05, 0.2) + low * 14.0))
    if kind == "cloth":
        texture = texture * (0.95 + 0.1 * _smooth_noise(rng, height, width, 2))
    base = 0.5 * base + 0.5 * float(base.mean())  # keep surfaces close to real tables and cloth: not too saturated
    image = texture[..., None] * base
    if kind == "tile":
        step = int(rng.integers(60, 160))
        image[::step, :] *= 0.75
        image[:, ::step] *= 0.75
    return np.clip(image, 0, 255).astype(np.float32)


def _casing_plane(
    layout: DisplayLayout, palette: DisplayPalette, display: FloatImage | None, ps: float, margin: float, rng: Rng
) -> tuple[FloatImage, NDArray[np.float32]]:
    """Casing + bezel + display on a flat plane. Returns premultiplied RGB (0..255) and alpha (0..1)."""
    width, height = layout.width, layout.height
    plane_w, plane_h = round((width + 2 * margin) * ps), round((height + 2 * margin) * ps)
    color = np.array(_CASINGS[int(rng.integers(0, len(_CASINGS)))], dtype=np.float32)
    rgb = np.ones((plane_h, plane_w, 3), dtype=np.float32) * color
    rgb *= np.linspace(1.04, 0.92, plane_h, dtype=np.float32)[:, None, None]
    outline = _rounded_rect(0.0, 0.0, plane_w - 1.0, plane_h - 1.0, min(plane_w, plane_h) * uniform(rng, (0.04, 0.12)))
    alpha = polygon_mask((plane_h, plane_w), [outline], antialias=True).astype(np.float32) / 255.0

    def to_px(value: float) -> int:
        return round((value + margin) * ps)

    byte = np.clip(rgb, 0, 255).astype(np.uint8)
    contrast = (30, 30, 30) if color.mean() > 110 else (200, 200, 200)
    for _ in range(int(rng.integers(0, 4))):  # buttons and legends: distractors
        bx = to_px(uniform(rng, (0.0, width * 0.9)))
        by = to_px(height + margin * uniform(rng, (0.2, 0.7)))
        if by < plane_h - 6:
            cv2.circle(byte, (bx, by), max(3, round(margin * ps * 0.12)), contrast, -1, cv2.LINE_AA)
    rgb = byte.astype(np.float32)
    if layout.has_display and display is not None:
        bezel = round(0.035 * height * ps)
        x0, y0 = to_px(0.0), to_px(0.0)
        cv2.rectangle(
            rgb,
            (x0 - bezel, y0 - bezel),
            (x0 + display.shape[1] + bezel, y0 + display.shape[0] + bezel),
            palette.bezel,
            -1,
        )
        rgb[y0 : y0 + display.shape[0], x0 : x0 + display.shape[1]] = display
        cv2.rectangle(
            rgb, (x0, y0), (x0 + display.shape[1] - 1, y0 + display.shape[0] - 1), (15, 15, 15), max(1, round(ps))
        )
    rgb *= alpha[..., None]
    return rgb, alpha


# --- camera ----------------------------------------------------------------------------------------------------


def _rotation(yaw: float, pitch: float, roll: float) -> NDArray[np.float64]:
    cy, sy, cp, sp, cr, sr = (
        math.cos(yaw),
        math.sin(yaw),
        math.cos(pitch),
        math.sin(pitch),
        math.cos(roll),
        math.sin(roll),
    )
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return np.asarray(rz @ rx @ ry, dtype=np.float64)


def project(
    points: NDArray[np.float64], rotation: NDArray[np.float64], distance: float, focal: float
) -> NDArray[np.float64]:
    """Pinhole projection of plane points (x, y) relative to the display centre."""
    flat = np.concatenate([points, np.zeros((len(points), 1))], axis=1)
    camera = flat @ rotation.T + np.array([0.0, 0.0, distance])
    return np.asarray(focal * camera[:, :2] / camera[:, 2:3])


def compose_scene(
    layout: DisplayLayout,
    palette: DisplayPalette,
    rng: Rng,
    config: Config,
    *,
    with_masks: bool = False,
    labels_only: bool = False,
) -> Scene:
    cfg = config["scene"]
    out_w, out_h = int(config["image"]["width"]), int(config["image"]["height"])
    ss = int(cfg["supersample"])
    canvas_w, canvas_h = out_w * ss, out_h * ss

    low_fraction, high_fraction = cfg["display_fraction"]
    fraction = low_fraction + (high_fraction - low_fraction) * float(rng.random()) ** float(
        cfg["display_fraction_skew"]
    )
    yaw = math.radians(uniform(rng, (-cfg["yaw_deg"], cfg["yaw_deg"])))
    pitch = math.radians(uniform(rng, (-cfg["pitch_deg"], cfg["pitch_deg"])))
    roll = math.radians(uniform(rng, (-cfg["roll_deg"], cfg["roll_deg"])))
    rotation = _rotation(yaw, pitch, roll)
    distance = layout.width * uniform(rng, (2.0, 4.0))
    focal = distance  # unit scale at the display centre; the real scale is fitted below

    half_w, half_h = layout.width / 2, layout.height / 2
    display_corners = np.array([(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)])
    projected = project(display_corners, rotation, distance, focal)
    span = projected.max(axis=0) - projected.min(axis=0)
    fit = min(fraction * canvas_w / span[0], 0.95 * canvas_h / span[1])
    centre_offset = np.array(
        [uniform(rng, (-cfg["shift"], cfg["shift"])) * canvas_w, uniform(rng, (-cfg["shift"], cfg["shift"])) * canvas_h]
    )
    origin = np.array([canvas_w / 2, canvas_h / 2]) + centre_offset
    px_per_unit = fit * focal / distance

    plane_scale = max(0.2, px_per_unit)
    # The margin is a whole number of plane pixels so that the glyph polygons land exactly where they are drawn.
    margin = round(uniform(rng, (0.12, 0.45)) * layout.height * plane_scale) / plane_scale
    plane_w = round((layout.width + 2 * margin) * plane_scale)
    plane_h = round((layout.height + 2 * margin) * plane_scale)
    corners_nominal = np.array(
        [
            (-half_w - margin, -half_h - margin),
            (half_w + margin, -half_h - margin),
            (half_w + margin, half_h + margin),
            (-half_w - margin, half_h + margin),
        ]
    )
    destination = project(corners_nominal, rotation, distance, focal) * fit + origin
    source = np.array([(0, 0), (plane_w, 0), (plane_w, plane_h), (0, plane_h)], dtype=np.float32)
    # `homography` maps edge coordinates (pixel i covers [i, i + 1]); OpenCV warps pixel-centre coordinates.
    homography = np.asarray(cv2.getPerspectiveTransform(source, destination.astype(np.float32)), dtype=np.float64)
    to_edge = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
    warp_matrix = np.linalg.inv(to_edge) @ homography @ to_edge
    params: dict[str, object] = {
        "display_fraction": round(fraction, 4),
        "yaw_deg": round(math.degrees(yaw), 3),
        "pitch_deg": round(math.degrees(pitch), 3),
        "roll_deg": round(math.degrees(roll), 3),
        "casing_margin": round(margin / layout.height, 4),
    }
    if labels_only:
        labels, masks = _labels(layout, homography, plane_scale, margin, ss, (out_w, out_h), with_masks=with_masks)
        return Scene(np.zeros((out_h, out_w, 3), dtype=np.float32), labels, params, masks)

    display = render_display(layout, palette, plane_scale, rng) if layout.has_display else None
    plane_rgb, plane_alpha = _casing_plane(layout, palette, display, plane_scale, margin, rng)
    warped_rgb = cv2.warpPerspective(
        plane_rgb, warp_matrix, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0)
    )
    warped_alpha = cv2.warpPerspective(
        plane_alpha, warp_matrix, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR, borderValue=0
    )

    canvas = as_float32(cv2.resize(background(rng, out_w, out_h), (canvas_w, canvas_h), interpolation=cv2.INTER_LINEAR))
    shadow = as_float32(
        cv2.GaussianBlur(
            np.roll(warped_alpha, (round(0.01 * canvas_h), round(0.01 * canvas_w)), axis=(0, 1)),
            (0, 0),
            0.012 * canvas_w,
        )
    )
    canvas *= (1.0 - 0.35 * shadow)[..., None]
    canvas = as_float32(canvas * (1.0 - warped_alpha[..., None]) + warped_rgb)
    image = as_float32(cv2.resize(canvas, (out_w, out_h), interpolation=cv2.INTER_AREA)) / 255.0

    labels, masks = _labels(layout, homography, plane_scale, margin, ss, (out_w, out_h), with_masks=with_masks)
    return Scene(np.clip(image, 0.0, 1.0).astype(np.float32), labels, params, masks)


def _labels(
    layout: DisplayLayout,
    homography: NDArray[np.float64],
    plane_scale: float,
    margin: float,
    ss: int,
    size: tuple[int, int],
    *,
    with_masks: bool,
) -> tuple[list[GlyphLabel], list[NDArray[np.uint8]]]:
    out_w, out_h = size
    labels: list[GlyphLabel] = []
    masks: list[NDArray[np.uint8]] = []
    for placed in layout.glyphs:
        if not placed.labelled or not placed.geometry.lit:
            continue
        polygons = placed.geometry.lit_polygons
        points = (np.concatenate(polygons) + margin) * plane_scale
        mapped = cv2.perspectiveTransform(points.reshape(-1, 1, 2).astype(np.float32), homography).reshape(-1, 2) / ss
        x0, y0 = mapped.min(axis=0)
        x1, y1 = mapped.max(axis=0)
        area = max((x1 - x0) * (y1 - y0), 1e-9)
        cx0, cy0, cx1, cy1 = max(0.0, x0), max(0.0, y0), min(float(out_w), x1), min(float(out_h), y1)
        visible = max(0.0, cx1 - cx0) * max(0.0, cy1 - cy0) / area
        labels.append(
            GlyphLabel(placed.symbol, placed.role, (float(cx0), float(cy0), float(cx1), float(cy1)), float(visible))
        )
        if with_masks:
            mask = np.zeros((out_h, out_w), dtype=np.uint8)
            for polygon in polygons:
                poly_points = (polygon + margin) * plane_scale
                poly_mapped = (
                    cv2.perspectiveTransform(poly_points.reshape(-1, 1, 2).astype(np.float32), homography).reshape(
                        -1, 2
                    )
                    / ss
                )
                fill_polygons(mask, [poly_mapped.astype(np.float64)], 255, antialias=False)
            masks.append(mask)
    return labels, masks
