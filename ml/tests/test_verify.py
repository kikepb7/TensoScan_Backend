"""T-B003-11: `tensoscan-ml verify` (CA-B003-05)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("ai_edge_litert")

from tensoscan_ml.cli import main
from tensoscan_ml.config import ml_root
from tensoscan_ml.data.manifest import ManifestEntry, write_manifest
from tensoscan_ml.export.tflite_inspect import ModelInfo, TensorInfo
from tensoscan_ml.verify import checks
from tensoscan_ml.verify.checks import FAIL, INFO, PASS, SKIP, VerifyReport, verify_model
from tensoscan_ml.verify.golden import make_synthetic_golden

pytestmark = pytest.mark.runtime

FIXTURES = ml_root() / "fixtures" / "dummy"
FIXED = FIXTURES / "digits-0.0.0-dummy-fixed.tflite"
RANDOM = FIXTURES / "digits-0.0.0-dummy-random.tflite"


def _copy(model: Path, target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy(model, target / model.name)
    shutil.copy(model.with_suffix(".json"), target / model.with_suffix(".json").name)
    return target / model.name


def _golden_manifest(root: Path, reading: tuple[int, int, int] = (128, 82, 67), count: int = 20) -> Path:
    entries = []
    (root / "images").mkdir(parents=True, exist_ok=True)
    for index in range(count):
        cv2.imwrite(
            str(root / "images" / f"{index}.jpg"),
            np.random.default_rng(index).integers(0, 255, (480, 384, 3), dtype=np.uint8),
        )
        entries.append(
            ManifestEntry(f"images/{index}.jpg", "0" * 64, "golden", "COMPLETE", *reading, "good", f"v{index}", 0)
        )
    write_manifest(root / "golden.jsonl", entries)
    return root / "golden.jsonl"


def _by_name(report: VerifyReport) -> dict[str, str]:
    return {check.name: check.status for check in report.checks}


def test_a_conformant_model_passes_every_check(tmp_path: Path) -> None:
    """CA-B003-05: shapes, ops, classes, size, sidecar and the golden set all pass; latency is informational."""
    report = verify_model(FIXED, golden=_golden_manifest(tmp_path), runs=10)

    status = _by_name(report)
    assert report.passed
    assert {
        status[name]
        for name in (
            "input",
            "output",
            "input.name",
            "output.name",
            "operators",
            "classes.count",
            "classes.order",
            "size",
            "golden",
        )
    } == {PASS}
    assert status["sidecar.sha256"] == PASS and status["sidecar.contract_version"] == PASS
    assert status["latency"] == INFO
    assert report.latency is not None and 0 < report.latency["p50_ms"] <= report.latency["p95_ms"]
    assert report.latency["runs"] == 10 and report.latency["threads"] == 4
    assert "CONV_2D" not in report.operators or len(report.operators) > 0


def test_the_random_model_fails_the_golden_set_because_it_has_no_predictive_value(tmp_path: Path) -> None:
    report = verify_model(RANDOM, golden=_golden_manifest(tmp_path), runs=5)

    status = _by_name(report)
    assert status["golden"] == FAIL and not report.passed
    assert status["operators"] == PASS and status["input"] == PASS


def test_without_a_golden_set_the_check_is_reported_as_skipped() -> None:
    report = verify_model(FIXED, runs=3)

    assert _by_name(report)["golden"] == SKIP
    assert report.passed


def test_a_sidecar_with_the_wrong_hash_classes_or_contract_major_fails(tmp_path: Path) -> None:
    model = _copy(FIXED, tmp_path)
    sidecar_path = model.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["sha256"] = "0" * 64
    sidecar["classes"] = [*sidecar["classes"][1:], sidecar["classes"][0]]
    sidecar["contract_version"] = "2.0.0"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    status = _by_name(verify_model(model, runs=3))

    assert status["sidecar.sha256"] == FAIL
    assert status["classes.order"] == FAIL
    assert status["sidecar.contract_version"] == FAIL


def test_a_missing_sidecar_fails(tmp_path: Path) -> None:
    model = tmp_path / FIXED.name
    shutil.copy(FIXED, model)

    report = verify_model(model, runs=3)

    assert _by_name(report)["sidecar"] == FAIL and not report.passed


def test_missing_model_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        verify_model(tmp_path / "nope.tflite")


def _info(**overrides: object) -> ModelInfo:
    base: dict[str, object] = {
        "operators": ("CONV_2D", "LOGISTIC"),
        "custom_operators": (),
        "inputs": (TensorInfo("input", (1, 320, 320, 3), "FLOAT32"),),
        "outputs": (TensorInfo("output", (1, 2100, 18), "FLOAT32"),),
        "size_bytes": 1_000_000,
    }
    return ModelInfo(**{**base, **overrides})  # type: ignore[arg-type]


def test_flex_or_custom_operators_and_an_oversized_model_fail() -> None:
    report = VerifyReport("m", "h")
    checks.check_ops_and_size(report, _info(custom_operators=("FlexErf",), size_bytes=7 * 1024 * 1024))

    assert _by_name(report) == {"operators": FAIL, "size": FAIL}


def test_wrong_input_shape_dtype_and_class_count_fail() -> None:
    report = VerifyReport("m", "h")
    wrong = _info(
        inputs=(TensorInfo("serving_default_input:0", (1, 224, 224, 3), "FLOAT32"),),
        outputs=(TensorInfo("Identity", (1, 2100, 16), "FLOAT32"),),
    )

    checks.check_io(report, wrong, None)

    status = _by_name(report)
    assert status["input"] == FAIL and status["input.name"] == FAIL
    assert status["output"] == FAIL and status["output.name"] == FAIL and status["classes.count"] == FAIL


def test_an_int8_sidecar_expects_a_uint8_input() -> None:
    report = VerifyReport("m", "h")
    sidecar = {"input": {"dtype": "uint8"}, "output": {"num_candidates": 2100}, "classes": []}

    checks.check_io(report, _info(), sidecar)

    assert _by_name(report)["input"] == FAIL  # a float32 model does not satisfy an int8 sidecar


def test_latency_is_informational_unless_it_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    slow = {"p50_ms": 80.0, "p95_ms": 120.0, "runs": 1, "threads": 4, "delegate": "x", "indicative": True}
    monkeypatch.setattr(checks, "measure_latency", lambda *_args: slow)
    relaxed, strict = VerifyReport("m", "h"), VerifyReport("m", "h")

    checks.check_latency(relaxed, FIXED, 1, 4, enforce=False)
    checks.check_latency(strict, FIXED, 1, 4, enforce=True)

    assert _by_name(relaxed)["latency"] == INFO and relaxed.passed
    assert _by_name(strict)["latency"] == FAIL and not strict.passed


def test_cli_prints_the_checks_writes_the_json_report_and_sets_the_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    golden = _golden_manifest(tmp_path)
    report_path = tmp_path / "report.json"

    ok = main(["verify", "--model", str(FIXED), "--golden", str(golden), "--runs", "5", "--report", str(report_path)])
    bad = main(
        [
            "verify",
            "--model",
            str(RANDOM),
            "--golden",
            str(golden),
            "--runs",
            "5",
            "--report",
            str(tmp_path / "bad.json"),
        ]
    )

    output = capsys.readouterr().out
    assert (ok, bad) == (0, 1)
    assert "VERIFIED" in output and "FAILED" in output
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert (
        saved["passed"] is True
        and saved["sha256"]
        and {c["name"] for c in saved["checks"]} >= {"input", "output", "operators", "golden"}
    )


def test_the_synthetic_golden_set_has_twenty_readable_frames_with_known_readings(tmp_path: Path) -> None:
    from tensoscan_ml.data.manifest import read_manifest

    entries = read_manifest(make_synthetic_golden(tmp_path))

    assert len(entries) == 20
    assert all(e.status == "COMPLETE" and e.sys and e.dia and e.pul for e in entries)
    assert len({e.reading for e in entries}) > 5
