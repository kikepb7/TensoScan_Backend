"""Builds a detection source from the `--detector`/`--model` command-line pair."""

from __future__ import annotations

from tensoscan_ml.detector.runtime import Detector, JsonlDetections, tflite_detector


def detector_from_args(detector: str, model: str | None, threads: int = 4) -> Detector | JsonlDetections:
    """`checkpoint` and `tflite` need `model`; a value ending in `.jsonl` is a file of pre-computed detections."""
    if detector.endswith(".jsonl"):
        return JsonlDetections(detector)
    if detector == "tflite":
        if not model:
            raise ValueError("--detector tflite needs --model path/to/digits.tflite")
        return tflite_detector(model, threads)
    if detector == "checkpoint":
        if not model:
            raise ValueError("--detector checkpoint needs --model path/to/best.pt")
        from tensoscan_ml.model.runtime import checkpoint_detector

        return checkpoint_detector(model)
    raise ValueError("--detector must be checkpoint, tflite or a path to a .jsonl file")
