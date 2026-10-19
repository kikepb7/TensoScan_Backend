"""T-B003-10b: dummy models that respect the contract (needs the `export` extra)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("onnx2tf")
pytest.importorskip("ai_edge_litert")
pytest.importorskip("tensorflow")


from tensoscan_ml.cli import main
from tensoscan_ml.config import contracts_dir
from tensoscan_ml.contract import CLASSES, CONTRACT_VERSION
from tensoscan_ml.data.manifest import ManifestEntry, write_manifest
from tensoscan_ml.detector.runtime import ModelDetector, Sidecar, TfliteModel
from tensoscan_ml.export.dummy import build_dummy, fixed_detections
from tensoscan_ml.export.tflite_inspect import inspect_file
from tensoscan_ml.reading import parse_detections

pytestmark = pytest.mark.export

CONTRACT_SIDECAR_KEYS = {
    "name",
    "version",
    "contract_version",
    "input",
    "output",
    "classes",
    "recommended",
    "sha256",
    "training",
    "metrics",
}


@pytest.fixture(scope="module")
def dummies(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("dummy")
    for kind in ("random", "fixed"):
        build_dummy(root, kind, seed=1)
    return root


@pytest.mark.parametrize("kind", ["random", "fixed"])
def test_dummy_models_have_the_exact_io_of_the_contract_and_only_builtin_ops(dummies: Path, kind: str) -> None:
    info = inspect_file(dummies / f"digits-0.0.0-dummy-{kind}.tflite")

    assert info.custom_operators == () and info.flex_operators == ()
    assert [(t.name, t.shape, t.dtype) for t in info.inputs] == [("input", (1, 320, 320, 3), "FLOAT32")]
    assert [(t.name, t.shape, t.dtype) for t in info.outputs] == [("output", (1, 2100, 18), "FLOAT32")]
    assert info.size_bytes < 3 * 1024 * 1024


@pytest.mark.parametrize("kind", ["random", "fixed"])
def test_dummy_sidecar_follows_the_contract_and_carries_the_real_sha256(dummies: Path, kind: str) -> None:
    import hashlib

    tflite = dummies / f"digits-0.0.0-dummy-{kind}.tflite"
    sidecar = json.loads(tflite.with_suffix(".json").read_text(encoding="utf-8"))

    assert set(sidecar) == CONTRACT_SIDECAR_KEYS
    assert sidecar["contract_version"] == CONTRACT_VERSION and sidecar["name"] == "digit-detector"
    assert sidecar["classes"] == list(CLASSES)
    assert sidecar["sha256"] == hashlib.sha256(tflite.read_bytes()).hexdigest()
    assert sidecar["input"] == {
        "width": 320,
        "height": 320,
        "channels": 3,
        "dtype": "float32",
        "range": [0, 1],
        "letterbox_pad": 0.447,
    }
    assert sidecar["output"] == {"num_candidates": 2100, "box_format": "cxcywh_normalized", "scores": "sigmoid"}
    assert sidecar["recommended"] == {"score_threshold": 0.4, "nms_iou_threshold": 0.45, "max_detections": 32}
    assert (tflite.parent / f"{tflite.stem}.model-card.md").is_file()


def test_the_sidecar_matches_the_documented_example_structure() -> None:
    text = (contracts_dir() / "digit-detector-model.md").read_text(encoding="utf-8")
    start, end = text.index('"name": "digit-detector"'), text.index("```", text.index('"name": "digit-detector"'))
    example = json.loads("{" + text[start:end].rstrip().rstrip("}").rstrip() + "}")

    assert set(example) == CONTRACT_SIDECAR_KEYS


def test_the_fixed_dummy_reports_the_same_reading_through_the_apps_decoder_and_parser(dummies: Path) -> None:
    tflite = dummies / "digits-0.0.0-dummy-fixed.tflite"
    detector = ModelDetector(TfliteModel(tflite), "fixed", Sidecar.load(tflite), score_threshold=0.4)
    picture = np.random.default_rng(0).integers(0, 255, (480, 384, 3), dtype=np.uint8)

    detections = detector.detect(picture)
    reading = parse_detections(detections)

    assert len(detections) == len(fixed_detections())  # the duplicate (NMS) and the 0.30 box (threshold) are gone
    assert reading.status.value == "COMPLETE" and reading.values == (128, 82, 67)


def test_the_fixed_dummy_ignores_the_picture(dummies: Path) -> None:
    tflite = dummies / "digits-0.0.0-dummy-fixed.tflite"
    model = TfliteModel(tflite)

    first = model.run(np.zeros((1, 320, 320, 3), dtype=np.float32))
    second = model.run(np.ones((1, 320, 320, 3), dtype=np.float32))

    assert np.abs(first - second).max() < 1e-6


def test_the_random_dummy_floods_the_app_with_candidates_but_respects_the_detection_limit(dummies: Path) -> None:
    tflite = dummies / "digits-0.0.0-dummy-random.tflite"
    raw = TfliteModel(tflite).run(np.random.default_rng(0).random((1, 320, 320, 3), dtype=np.float32))
    detector = ModelDetector(TfliteModel(tflite), "random", Sidecar.load(tflite), score_threshold=0.4)

    assert raw.shape == (1, 2100, 18) and np.isfinite(raw).all()
    assert float(raw[..., 4:].min()) >= 0.0 and float(raw[..., 4:].max()) <= 1.0
    assert float(raw[..., :4].min()) >= 0.0 and float(raw[..., :4].max()) <= 1.0
    detections = detector.detect(np.full((480, 384, 3), 128, dtype=np.uint8))
    assert 1 <= len(detections) <= 32


def test_eval_on_the_tflite_fixed_dummy_reads_the_reading_end_to_end(
    dummies: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The exported `.tflite` goes through the same pre/post-processing as the app (REQ-B003-17)."""
    entries = []
    for index in range(3):
        path = tmp_path / "images" / f"{index}.jpg"
        path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(path), np.random.default_rng(index).integers(0, 255, (480, 384, 3), dtype=np.uint8))
        entries.append(
            ManifestEntry(f"images/{index}.jpg", "0" * 64, "dummy", "COMPLETE", 128, 82, 67, "good", f"v{index}", 0)
        )
    write_manifest(tmp_path / "manifest.jsonl", entries)

    actual = main(
        [
            "eval",
            "--detector",
            "tflite",
            "--model",
            str(dummies / "digits-0.0.0-dummy-fixed.tflite"),
            "--manifest",
            str(tmp_path / "manifest.jsonl"),
            "--out",
            str(tmp_path / "eval"),
        ]
    )

    report = json.loads((tmp_path / "eval" / "report.json").read_text(encoding="utf-8"))
    assert report["reading"]["frame"]["accuracy"] == 1.0
    assert report["reading"]["video"]["accuracy"] == 1.0
    assert actual in (0, 1)  # one device: the worst-device/other thresholds may still be reported
    assert "reading per frame: 3/3" in capsys.readouterr().out


@pytest.mark.parametrize("kind", ["random", "fixed"])
def test_the_committed_fixture_is_intact_and_below_the_size_limit(kind: str) -> None:
    """The Android app copies these files: the sidecar hash must match and they must stay small."""
    import hashlib

    from tensoscan_ml.config import ml_root

    tflite = ml_root() / "fixtures" / "dummy" / f"digits-0.0.0-dummy-{kind}.tflite"
    sidecar = json.loads(tflite.with_suffix(".json").read_text(encoding="utf-8"))

    assert sidecar["sha256"] == hashlib.sha256(tflite.read_bytes()).hexdigest()
    assert tflite.stat().st_size < 3 * 1024 * 1024
    assert inspect_file(tflite).custom_operators == ()
