"""Dummy models that respect the contract exactly, with no predictive value (T-B003-10b).

They let the app develop and test everything around the model (loading, sha256 check, letterbox, decoding, latency)
before any training happens:

* `random`: the real detector architecture with random weights; the heads are biased so that every anchor scores above
  the threshold, so the app sees a flood of garbage candidates and must survive the NMS and the 32-detection limit.
* `fixed`: ignores the picture and always "sees" the same reading in a 4:5 ROI (default 128/82/67) plus a duplicate box
  (to exercise the NMS) and a low-score box (to exercise the threshold). It makes the whole decode -> parse chain
  testable end to end.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from tensoscan_ml.contract import CLASS_INDEX, INPUT_SIZE, NUM_CLASSES, STRIDES, VALUES_PER_CANDIDATE
from tensoscan_ml.detector.preprocess import letterbox_transform
from tensoscan_ml.export.pipeline import ExportOptions, ExportResult, export_module
from tensoscan_ml.export.wrapper import OutputContractWrapper
from tensoscan_ml.model.yolox import YoloxNano
from tensoscan_ml.reading import Box, SymbolDetection

DUMMY_VERSION = "0.0.0-dummy"
DEFAULT_READING = (128, 82, 67)

# Layout of the golden case `stacked_three_rows_small_pulse` (coordinates normalised to a 4:5 ROI).
_ROWS = (
    # top, bottom, digit width, width of a "1", right edge
    (0.08, 0.38, 0.165, 0.075, 0.777),  # SYS
    (0.42, 0.72, 0.165, 0.075, 0.786),  # DIA
    (0.80, 0.94, 0.077, 0.040, 0.791),  # PUL
)
_GAP = 0.036
_SCORE = 0.95


def fixed_detections(reading: tuple[int, int, int] = DEFAULT_READING) -> list[SymbolDetection]:
    """Glyph detections (ROI coordinates) of a stacked SYS / DIA / small PUL display showing `reading`."""
    detections: list[SymbolDetection] = []
    for value, (top, bottom, width, one_width, right) in zip(reading, _ROWS, strict=True):
        widths = [one_width if digit == "1" else width for digit in str(value)]
        x = right - (sum(widths) + _GAP * (len(widths) - 1))
        for digit, glyph_width in zip(str(value), widths, strict=True):
            detections.append(SymbolDetection(digit, Box(x, top, x + glyph_width, bottom), _SCORE))
            x += glyph_width + _GAP
    return detections


def fixed_output(reading: tuple[int, int, int], input_size: int = INPUT_SIZE, candidates: int = 2100) -> np.ndarray:
    """The constant `[1, N, 4 + C]` tensor of the `fixed` dummy (boxes as the contract defines them)."""
    transform = letterbox_transform(384, 480, input_size)  # any 4:5 ROI gives the same transform ratios
    rows = np.zeros((candidates, VALUES_PER_CANDIDATE), dtype=np.float32)

    def put(index: int, symbol: str, box: Box, score: float) -> None:
        x0 = box.left * transform.scaled_width + transform.pad_left
        x1 = box.right * transform.scaled_width + transform.pad_left
        y0 = box.top * transform.scaled_height + transform.pad_top
        y1 = box.bottom * transform.scaled_height + transform.pad_top
        rows[index, :4] = [
            (x0 + x1) / 2 / input_size,
            (y0 + y1) / 2 / input_size,
            (x1 - x0) / input_size,
            (y1 - y0) / input_size,
        ]
        rows[index, 4 + CLASS_INDEX[symbol]] = score

    detections = fixed_detections(reading)
    for index, detection in enumerate(detections):
        put(index, detection.symbol, detection.box, detection.confidence)
    first = detections[0].box
    shifted = Box(first.left + 0.004, first.top + 0.004, first.right + 0.004, first.bottom + 0.004)
    put(len(detections), "7", shifted, 0.60)  # overlaps the first glyph: the class-agnostic NMS must drop it
    put(len(detections) + 1, "4", Box(0.05, 0.50, 0.12, 0.60), 0.30)  # below the 0.40 threshold
    return rows[None]


class FixedOutputWrapper(OutputContractWrapper):
    """Contract wrapper whose output ignores the picture (it still reads it, so the input stays in the graph)."""

    constant: torch.Tensor

    def __init__(self, reading: tuple[int, int, int], input_size: int = INPUT_SIZE) -> None:
        super().__init__(nn.Identity(), input_size, STRIDES, layout="nchw")
        self.register_buffer("constant", torch.from_numpy(fixed_output(reading, input_size, self.num_candidates)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.constant + torch.mean(x) * 1e-12


def random_wrapper(seed: int, score_bias: float = 1.0) -> OutputContractWrapper:
    torch.manual_seed(seed)
    model = YoloxNano()
    model.set_score_bias(score_bias)
    return OutputContractWrapper(model, INPUT_SIZE, STRIDES, layout="nchw").eval()


def build_dummy(
    out_dir: str | Path, kind: str, *, seed: int = 0, reading: tuple[int, int, int] = DEFAULT_READING
) -> ExportResult:
    """Exports one dummy model (`random` or `fixed`) to `out_dir`."""
    if kind == "random":
        module: OutputContractWrapper = random_wrapper(seed)
    elif kind == "fixed":
        module = FixedOutputWrapper(reading)
    else:
        raise ValueError("kind must be 'random' or 'fixed'")
    options = ExportOptions(version=f"{DUMMY_VERSION}-{kind}", dataset_version=f"dummy-{kind}", git_commit="dummy")
    extra = {
        "dummy": True,
        "kind": kind,
        "seed": seed,
        "reading": list(reading) if kind == "fixed" else None,
        "classes": NUM_CLASSES,
    }
    return export_module(module.eval(), out_dir, options, extra_info=extra)
