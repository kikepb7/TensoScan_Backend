"""Detector back-ends: `.tflite` (LiteRT), checkpoint and pre-computed `detections.jsonl`."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from tensoscan_ml.contract import CLASSES, MAX_DETECTIONS, NMS_IOU_THRESHOLD, SCORE_THRESHOLD
from tensoscan_ml.detector.postprocess import decode_output
from tensoscan_ml.detector.preprocess import letterbox
from tensoscan_ml.reading import SymbolDetection

RgbImage = NDArray[np.uint8]

#: The score threshold used to decode when measuring mAP: low, so the precision-recall curve is complete.
EVAL_SCORE_THRESHOLD = 0.05


class Detector(Protocol):
    """Anything that turns an RGB ROI crop into glyph detections normalised to that crop."""

    name: str

    def detect(self, image_rgb: RgbImage) -> list[SymbolDetection]: ...


@dataclass(frozen=True)
class Sidecar:
    """The parts of `digits-<version>.json` the tooling needs (contract section 5)."""

    classes: tuple[str, ...]
    num_candidates: int
    score_threshold: float
    nms_iou_threshold: float
    max_detections: int
    input_size: int

    @classmethod
    def load(cls, model_path: str | Path) -> Sidecar:
        path = Path(model_path).with_suffix(".json")
        if not path.is_file():
            return cls(CLASSES, 0, SCORE_THRESHOLD, NMS_IOU_THRESHOLD, MAX_DETECTIONS, 320)
        raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        recommended = raw.get("recommended", {})
        return cls(
            classes=tuple(raw.get("classes", CLASSES)),
            num_candidates=int(raw.get("output", {}).get("num_candidates", 0)),
            score_threshold=float(recommended.get("score_threshold", SCORE_THRESHOLD)),
            nms_iou_threshold=float(recommended.get("nms_iou_threshold", NMS_IOU_THRESHOLD)),
            max_detections=int(recommended.get("max_detections", MAX_DETECTIONS)),
            input_size=int(raw.get("input", {}).get("width", 320)),
        )


class RawModel(Protocol):
    """A model that maps the letterboxed `[1, S, S, 3]` input to the `[1, N, 4 + C]` contract output."""

    def run(self, model_input: NDArray[np.float32]) -> NDArray[np.float32]: ...


class ModelDetector:
    """Letterbox -> model -> app decoding. Shared by the `.tflite` and checkpoint back-ends."""

    def __init__(
        self, model: RawModel, name: str, sidecar: Sidecar, score_threshold: float = EVAL_SCORE_THRESHOLD
    ) -> None:
        self.model = model
        self.name = name
        self.sidecar = sidecar
        self.score_threshold = score_threshold

    def detect(self, image_rgb: RgbImage) -> list[SymbolDetection]:
        model_input, transform = letterbox(image_rgb, self.sidecar.input_size)
        output = self.model.run(model_input)
        return decode_output(
            output,
            transform,
            classes=self.sidecar.classes,
            score_threshold=self.score_threshold,
            nms_iou_threshold=self.sidecar.nms_iou_threshold,
            max_detections=self.sidecar.max_detections,
        )


class TfliteModel:
    """LiteRT interpreter on CPU (XNNPACK). Quantised inputs and outputs are converted with their parameters."""

    def __init__(self, path: str | Path, num_threads: int = 4) -> None:
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError as error:  # pragma: no cover - depends on the extra
            raise ImportError("running a .tflite needs the `runtime` extra: uv sync --extra runtime") from error
        self.interpreter = Interpreter(model_path=str(path), num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]

    def run(self, model_input: NDArray[np.float32]) -> NDArray[np.float32]:
        dtype = self.input["dtype"]
        if dtype == np.float32:
            feed = model_input
        else:  # uint8/int8 variant: quantise with the tensor's parameters
            scale, zero_point = self.input["quantization"]
            info = np.iinfo(dtype)
            feed = np.clip(np.round(model_input / scale + zero_point), info.min, info.max).astype(dtype)
        self.interpreter.set_tensor(self.input["index"], feed)
        self.interpreter.invoke()
        result = self.interpreter.get_tensor(self.output["index"])
        if result.dtype != np.float32:
            scale, zero_point = self.output["quantization"]
            result = (result.astype(np.float32) - zero_point) * scale
        return np.asarray(result, dtype=np.float32)


class JsonlDetections:
    """Detections computed elsewhere (e.g. dumped by the app in debug mode), looked up by image path or sha256."""

    def __init__(self, path: str | Path) -> None:
        self.name = f"detections:{Path(path).name}"
        self.by_key: dict[str, list[SymbolDetection]] = {}
        with Path(path).open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    key = raw.get("path") or raw["sha256"]
                    self.by_key[str(key)] = [SymbolDetection.from_dict(d) for d in raw["detections"]]
                except (KeyError, ValueError, TypeError) as error:
                    raise ValueError(f"{path}:{number}: invalid detections line ({error})") from error

    def lookup(self, path: str, sha256: str) -> list[SymbolDetection] | None:
        return self.by_key.get(path) if path in self.by_key else self.by_key.get(sha256)


def tflite_detector(path: str | Path, num_threads: int = 4) -> ModelDetector:
    return ModelDetector(TfliteModel(path, num_threads), f"tflite:{Path(path).name}", Sidecar.load(path))
