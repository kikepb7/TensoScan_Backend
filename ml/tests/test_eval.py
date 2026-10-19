"""T-B003-07: evaluation harness. Perfect detections give 100 %; mistakes are measured (REQ-B003-16)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tensoscan_ml.cli import main
from tensoscan_ml.contract import CLASS_INDEX
from tensoscan_ml.data.manifest import ManifestEntry, read_manifest, write_manifest
from tensoscan_ml.detector.preprocess import letterbox_transform
from tensoscan_ml.detector.runtime import JsonlDetections, ModelDetector, Sidecar
from tensoscan_ml.eval.metrics import detection_metrics, judge, make_result, reading_metrics
from tensoscan_ml.eval.runner import evaluate_manifest
from tensoscan_ml.reading import Box, FrameReading, FrameStatus, SymbolDetection
from tensoscan_ml.synth.params import load_synth_config
from tensoscan_ml.synth.writer import write_dataset


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small synthetic dataset where the official parser reproduces every label (flat, readable, compatible)."""
    root = tmp_path_factory.mktemp("synth")
    config = load_synth_config(overrides={"scene": {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}})
    write_dataset(root, config, {"val": 60}, seed=11)
    entries = read_manifest(root / "val" / "manifest.jsonl")
    keep = [
        e
        for e in entries
        if e.extra["readable"] and e.extra["parser_compatible"] and e.status in ("COMPLETE", "DEVICE_ERROR")
    ]
    write_manifest(root / "val" / "perfect.jsonl", keep)
    assert len(keep) >= 12
    return root / "val"


def _perfect_jsonl(dataset: Path, target: Path, drop_digit: bool = False) -> Path:
    entries = read_manifest(dataset / "perfect.jsonl")
    with target.open("w", encoding="utf-8") as handle:
        for index, entry in enumerate(entries):
            detections = []
            for line in (dataset / str(entry.labels)).read_text(encoding="utf-8").splitlines():
                cls, cx, cy, w, h = line.split()
                symbol = list(CLASS_INDEX)[int(cls)]
                box = [
                    float(cx) - float(w) / 2,
                    float(cy) - float(h) / 2,
                    float(cx) + float(w) / 2,
                    float(cy) + float(h) / 2,
                ]
                detections.append({"symbol": symbol, "box": box, "confidence": 0.95})
            if drop_digit and index % 2 == 0 and entry.status == "COMPLETE":
                digits = [i for i, d in enumerate(detections) if d["symbol"].isdigit()]
                del detections[digits[0]]  # a missed glyph: the reading can no longer be right
            handle.write(json.dumps({"path": entry.path, "detections": detections}) + "\n")
    return target


def test_perfect_detections_give_a_hundred_percent_everywhere(dataset: Path, tmp_path: Path) -> None:
    source = JsonlDetections(_perfect_jsonl(dataset, tmp_path / "det.jsonl"))

    report = evaluate_manifest(dataset / "perfect.jsonl", source, tmp_path / "out")

    assert report["reading"]["frame"]["accuracy"] == 1.0
    assert report["reading"]["video"]["accuracy"] == 1.0
    assert report["reading"]["worst_device_accuracy"] == 1.0
    assert report["detection"]["map50"] == pytest.approx(1.0)
    assert report["detection"]["map50_95"] == pytest.approx(1.0)
    assert all(ap in (None, pytest.approx(1.0)) for ap in report["detection"]["ap50_by_class"].values())
    assert all(check["pass"] for check in report["assessment"]["checks"].values())
    assert report["worst"] == []
    assert report["assessment"]["notes"], "synthetic data must carry the 'does not count' note"
    assert (tmp_path / "out" / "report.md").read_text(encoding="utf-8").startswith("# Evaluation report")
    saved = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))
    assert saved["reading"]["frame"]["correct"] == saved["reading"]["frame"]["frames"] > 0


def test_a_model_detector_through_letterbox_and_decoder_also_reaches_a_hundred_percent(
    dataset: Path, tmp_path: Path
) -> None:
    """The `checkpoint`/`tflite` path: pixels -> letterbox -> model -> app decoder -> parser."""
    entries = read_manifest(dataset / "perfect.jsonl")
    labels = [
        [line.split() for line in (dataset / str(e.labels)).read_text(encoding="utf-8").splitlines()] for e in entries
    ]

    class OracleModel:
        """Emits the contract output for the labels of the image it is shown (calls arrive in manifest order)."""

        def __init__(self) -> None:
            self.calls = 0

        def run(self, model_input: np.ndarray) -> np.ndarray:
            transform = letterbox_transform(384, 480)
            rows = np.zeros((2100, 18), dtype=np.float32)
            for index, (cls, cx, cy, w, h) in enumerate(labels[self.calls]):
                x0 = (float(cx) - float(w) / 2) * transform.scaled_width + transform.pad_left
                x1 = (float(cx) + float(w) / 2) * transform.scaled_width + transform.pad_left
                y0 = (float(cy) - float(h) / 2) * transform.scaled_height + transform.pad_top
                y1 = (float(cy) + float(h) / 2) * transform.scaled_height + transform.pad_top
                rows[index, :4] = [(x0 + x1) / 640, (y0 + y1) / 640, (x1 - x0) / 320, (y1 - y0) / 320]
                rows[index, 4 + int(cls)] = 0.9
            self.calls += 1
            return rows[None]

    detector = ModelDetector(OracleModel(), "oracle", Sidecar.load(tmp_path / "none.tflite"))

    report = evaluate_manifest(dataset / "perfect.jsonl", detector, tmp_path / "out", write_failures=False)

    assert report["reading"]["frame"]["accuracy"] == 1.0
    assert report["detection"]["map50"] == pytest.approx(1.0, abs=0.01)


def test_mistakes_lower_the_accuracy_and_the_worst_images_are_listed(dataset: Path, tmp_path: Path) -> None:
    jsonl = _perfect_jsonl(dataset, tmp_path / "det.jsonl", drop_digit=True)

    report = evaluate_manifest(dataset / "perfect.jsonl", JsonlDetections(jsonl), tmp_path / "out")

    assert report["reading"]["frame"]["accuracy"] < 1.0
    assert report["worst"] and report["worst"][0]["got"]["status"]
    assert any(p.suffix == ".jpg" for p in (tmp_path / "out" / "failures").iterdir())
    assert any(not check["pass"] for check in report["assessment"]["checks"].values())


def test_cli_evaluates_a_detections_file_and_flags_thresholds_it_misses(
    dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    good = _perfect_jsonl(dataset, tmp_path / "good.jsonl")

    assert (
        main(
            [
                "eval",
                "--detector",
                str(good),
                "--manifest",
                str(dataset / "perfect.jsonl"),
                "--out",
                str(tmp_path / "o"),
            ]
        )
        == 0
    )
    assert "reading per frame" in capsys.readouterr().out
    assert (tmp_path / "o" / "run.json").is_file()


def test_missing_detections_for_an_image_is_an_error(dataset: Path, tmp_path: Path) -> None:
    empty = tmp_path / "det.jsonl"
    empty.write_text(json.dumps({"path": "images/other.jpg", "detections": []}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no detections in the source"):
        evaluate_manifest(dataset / "perfect.jsonl", JsonlDetections(empty), tmp_path / "out")


def _det(symbol: str, box: tuple[float, float, float, float], confidence: float = 0.9) -> SymbolDetection:
    return SymbolDetection(symbol, Box(*box), confidence)


def test_judge_distinguishes_values_statuses_and_unlabelled_images() -> None:
    reading = ManifestEntry("a", "0", "d", "COMPLETE", 128, 82, 67)
    error = ManifestEntry("b", "0", "d", "DEVICE_ERROR")
    nothing = ManifestEntry("c", "0", "d", None)

    assert judge(reading, FrameReading(FrameStatus.COMPLETE)) == (False, 3)
    assert judge(error, FrameReading(FrameStatus.DEVICE_ERROR)) == (True, 0)
    assert judge(error, FrameReading(FrameStatus.NO_READING)) == (False, 1)
    assert judge(nothing, FrameReading(FrameStatus.NO_READING)) == (None, 0)


def test_video_accuracy_uses_the_majority_vote_of_complete_frames() -> None:
    def frames(values: list[tuple[int, int, int]]) -> list[Any]:
        results = []
        for index, (sys, dia, pul) in enumerate(values):
            entry = ManifestEntry(f"f{index}", "0", "omron", "COMPLETE", 128, 82, 67, "good", "v1", index)
            detections = []
            x = 0.1
            for row, number in enumerate((sys, dia, pul)):
                for digit in str(number):
                    detections.append(_det(digit, (x, 0.1 + 0.3 * row, x + 0.07, 0.3 + 0.3 * row)))
                    x += 0.075
                x = 0.1
            results.append(make_result(entry, detections, None))
        return results

    right_majority = reading_metrics(frames([(128, 82, 67), (128, 82, 67), (128, 82, 68)]))
    wrong_majority = reading_metrics(frames([(128, 82, 68), (128, 82, 68), (128, 82, 67)]))

    assert right_majority["frame"]["accuracy"] == pytest.approx(2 / 3)
    assert right_majority["video"]["accuracy"] == 1.0
    assert wrong_majority["video"]["accuracy"] == 0.0


def test_map_is_one_for_exact_boxes_and_zero_when_boxes_miss_by_more_than_the_iou() -> None:
    entry = ManifestEntry("a", "0", "d", None)
    truth = [(CLASS_INDEX["8"], Box(0.1, 0.1, 0.2, 0.3))]
    exact = make_result(entry, [_det("8", (0.1, 0.1, 0.2, 0.3))], truth)
    shifted = make_result(entry, [_det("8", (0.18, 0.1, 0.28, 0.3))], truth)
    wrong_class = make_result(entry, [_det("3", (0.1, 0.1, 0.2, 0.3))], truth)

    assert detection_metrics([exact])["map50"] == pytest.approx(1.0)  # type: ignore[index]
    assert detection_metrics([shifted])["map50"] == 0.0  # type: ignore[index]
    result = detection_metrics([wrong_class])
    assert result is not None and result["map50"] == 0.0
    confusion = result["confusion"]["matrix"]
    assert confusion[CLASS_INDEX["8"]][CLASS_INDEX["3"]] == 1


def test_map_penalises_a_loose_box_at_high_iou_only() -> None:
    entry = ManifestEntry("a", "0", "d", None)
    truth = [(CLASS_INDEX["8"], Box(0.1, 0.1, 0.2, 0.3))]
    loose = make_result(entry, [_det("8", (0.1, 0.1, 0.2, 0.36))], truth)  # IoU about 0.77

    result = detection_metrics([loose])

    assert result is not None
    assert result["map50"] == pytest.approx(1.0)
    assert 0.0 < result["map50_95"] < 1.0
