"""Letterbox and decoder: the numerical contract shared with the app (contract sections 3 and 6)."""

from __future__ import annotations

import numpy as np
import pytest

from tensoscan_ml.contract import CLASS_INDEX, PAD_VALUE, VALUES_PER_CANDIDATE
from tensoscan_ml.detector.postprocess import decode_output
from tensoscan_ml.detector.preprocess import letterbox, letterbox_transform


def test_letterbox_of_the_apps_portrait_roi_matches_the_kotlin_arithmetic() -> None:
    actual = letterbox_transform(384, 480)

    assert (actual.scaled_width, actual.scaled_height) == (256, 320)
    assert (actual.pad_left, actual.pad_top) == (32, 0)


def test_letterbox_pads_with_grey_114_and_normalises_to_unit_range() -> None:
    image = np.full((480, 384, 3), 255, dtype=np.uint8)

    model_input, transform = letterbox(image)

    assert model_input.shape == (1, 320, 320, 3) and model_input.dtype == np.float32
    assert model_input[0, 10, 5, 0] == pytest.approx(PAD_VALUE)
    assert model_input[0, 10, transform.pad_left + 5, 0] == pytest.approx(1.0)
    assert model_input[0, 10, transform.pad_left + transform.scaled_width + 2, 0] == pytest.approx(PAD_VALUE)


def test_letterbox_keeps_rgb_order_and_samples_pixel_centres() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[:, :, 0] = 255  # red in RGB

    model_input, _ = letterbox(image, input_size=8)

    assert model_input[0, 4, 4].tolist() == pytest.approx([1.0, 0.0, 0.0])


def test_letterbox_of_a_horizontal_gradient_is_monotonic_and_symmetric() -> None:
    gradient = np.tile(np.linspace(0, 255, 100, dtype=np.uint8), (100, 1))[:, :, None].repeat(3, axis=2)

    model_input, transform = letterbox(gradient, input_size=50)

    row = model_input[0, 25, transform.pad_left : transform.pad_left + transform.scaled_width, 0]
    assert np.all(np.diff(row) >= 0)
    assert row[0] == pytest.approx(0.0, abs=0.02) and row[-1] == pytest.approx(1.0, abs=0.02)


def _row(
    symbol: str, box_roi: tuple[float, float, float, float], score: float, transform, size: int = 320
) -> list[float]:
    """Model output row for a glyph whose box is given in ROI coordinates."""
    left, top, right, bottom = box_roi
    x0, x1 = left * transform.scaled_width + transform.pad_left, right * transform.scaled_width + transform.pad_left
    y0, y1 = top * transform.scaled_height + transform.pad_top, bottom * transform.scaled_height + transform.pad_top
    scores = [0.0] * 14
    scores[CLASS_INDEX[symbol]] = score
    return [(x0 + x1) / 2 / size, (y0 + y1) / 2 / size, (x1 - x0) / size, (y1 - y0) / size, *scores]


def test_decoder_undoes_the_letterbox_and_returns_roi_coordinates() -> None:
    transform = letterbox_transform(384, 480)
    output = np.array([_row("7", (0.25, 0.10, 0.40, 0.30), 0.93, transform)], dtype=np.float32)

    actual = decode_output(output, transform)

    assert len(actual) == 1
    assert actual[0].symbol == "7"
    assert actual[0].confidence == pytest.approx(0.93, abs=1e-6)
    box = actual[0].box
    assert (box.left, box.top, box.right, box.bottom) == pytest.approx((0.25, 0.10, 0.40, 0.30), abs=1e-5)


def test_decoder_applies_threshold_class_agnostic_nms_and_the_limit() -> None:
    transform = letterbox_transform(384, 480)
    rows = [
        _row("3", (0.20, 0.20, 0.30, 0.40), 0.90, transform),
        _row("8", (0.201, 0.20, 0.301, 0.40), 0.80, transform),  # same glyph, other class: suppressed
        _row("5", (0.50, 0.20, 0.60, 0.40), 0.39, transform),  # below the threshold
        _row("1", (0.70, 0.20, 0.78, 0.40), 0.55, transform),
    ]

    actual = decode_output(np.array(rows, dtype=np.float32), transform)

    assert [(d.symbol, round(d.confidence, 2)) for d in actual] == [("3", 0.90), ("1", 0.55)]
    assert len(decode_output(np.array(rows, dtype=np.float32), transform, max_detections=1)) == 1


def test_decoder_ignores_boxes_that_lie_entirely_in_the_letterbox_border() -> None:
    transform = letterbox_transform(384, 480)
    border = [10 / 320, 100 / 320, 8 / 320, 20 / 320, *([0.0] * 10), 0.9, *([0.0] * 3)]  # x in 6..14 < pad_left 32

    assert decode_output(np.array([border], dtype=np.float32), transform) == []


def test_decoder_accepts_the_batched_output_shape() -> None:
    transform = letterbox_transform(384, 480)
    output = np.array([_row("0", (0.2, 0.2, 0.3, 0.4), 0.9, transform)], dtype=np.float32)[None]

    assert output.shape[-1] == VALUES_PER_CANDIDATE
    assert len(decode_output(output, transform)) == 1
