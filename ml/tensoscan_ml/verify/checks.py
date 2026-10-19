"""The checks of `tensoscan-ml verify` (T-B003-11, CA-B003-05)."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tensoscan_ml.contract import (
    CLASSES,
    CONTRACT_VERSION,
    INPUT_SIZE,
    MAX_LATENCY_P50_MS,
    MAX_LATENCY_P95_MS,
    MAX_MODEL_BYTES,
)
from tensoscan_ml.detector.runtime import ModelDetector, Sidecar, TfliteModel
from tensoscan_ml.export.tflite_inspect import ModelInfo, inspect_file
from tensoscan_ml.provenance import file_sha256

PASS, FAIL, SKIP, INFO = "pass", "fail", "skip", "info"


@dataclass
class Check:
    name: str
    status: str
    detail: str
    value: Any = None


@dataclass
class VerifyReport:
    model: str
    sha256: str
    checks: list[Check] = field(default_factory=list)
    latency: dict[str, Any] | None = None
    operators: list[str] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str, value: Any = None) -> None:
        self.checks.append(Check(name, status, detail, value))

    @property
    def passed(self) -> bool:
        return all(check.status != FAIL for check in self.checks)

    def to_json(self) -> str:
        payload = {**asdict(self), "passed": self.passed}
        return json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"


def check_sidecar(report: VerifyReport, model: Path, sidecar_path: Path) -> dict[str, Any] | None:
    if not sidecar_path.is_file():
        report.add("sidecar", FAIL, f"{sidecar_path.name} not found next to the model")
        return None
    sidecar: dict[str, Any] = json.loads(sidecar_path.read_text(encoding="utf-8"))
    report.add(
        "sidecar.sha256", PASS if sidecar.get("sha256") == report.sha256 else FAIL, "sha256 of the file vs the sidecar"
    )
    major = str(sidecar.get("contract_version", "")).split(".")[0]
    report.add(
        "sidecar.contract_version",
        PASS if major == CONTRACT_VERSION.split(".")[0] else FAIL,
        f"{sidecar.get('contract_version')} (this tooling implements {CONTRACT_VERSION})",
    )
    return sidecar


def check_io(report: VerifyReport, info: ModelInfo, sidecar: dict[str, Any] | None) -> None:
    int8 = bool(sidecar and sidecar["input"]["dtype"] == "uint8")
    (model_input,), (model_output,) = info.inputs, info.outputs
    expected_input = (1, INPUT_SIZE, INPUT_SIZE, 3)
    ok = model_input.shape == expected_input and model_input.dtype == ("UINT8" if int8 else "FLOAT32")
    report.add(
        "input",
        PASS if ok else FAIL,
        f"{model_input.name} {model_input.shape} {model_input.dtype}, expected {expected_input}",
    )
    report.add("input.name", PASS if model_input.name == "input" else FAIL, f"tensor named '{model_input.name}'")
    candidates = int(sidecar["output"]["num_candidates"]) if sidecar else model_output.shape[1]
    values = 4 + len(CLASSES)
    ok = model_output.shape == (1, candidates, values) and model_output.dtype == "FLOAT32"
    report.add(
        "output",
        PASS if ok else FAIL,
        f"{model_output.name} {model_output.shape} {model_output.dtype}, expected (1, {candidates}, {values})",
    )
    report.add("output.name", PASS if model_output.name == "output" else FAIL, f"tensor named '{model_output.name}'")
    report.add(
        "classes.count",
        PASS if model_output.shape[-1] == values else FAIL,
        f"{model_output.shape[-1] - 4} classes, expected {len(CLASSES)}",
    )
    if sidecar is not None:
        report.add(
            "classes.order",
            PASS if sidecar["classes"] == list(CLASSES) else FAIL,
            "sidecar class list vs the contract's order",
        )


def check_ops_and_size(report: VerifyReport, info: ModelInfo) -> None:
    report.operators = sorted(info.operator_set)
    if info.custom_operators:
        report.add("operators", FAIL, f"custom/Flex operators: {', '.join(info.custom_operators)}")
    else:
        report.add(
            "operators", PASS, f"{len(info.operator_set)} builtin operator types, no Flex or custom", report.operators
        )
    report.add(
        "size",
        PASS if info.size_bytes <= MAX_MODEL_BYTES else FAIL,
        f"{info.size_bytes / 1024:.0f} KiB (limit {MAX_MODEL_BYTES // 1024} KiB)",
        info.size_bytes,
    )


def measure_latency(model_path: Path, runs: int, threads: int) -> dict[str, Any]:
    """Wall-clock latency of `invoke` on this CPU with the XNNPACK delegate (indicative: the reference is a phone)."""
    model = TfliteModel(model_path, num_threads=threads)
    rng = np.random.default_rng(0)
    sample = rng.random((1, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32)
    for _ in range(5):
        model.run(sample)
    timings = []
    for _ in range(runs):
        started = time.perf_counter()
        model.run(sample)
        timings.append((time.perf_counter() - started) * 1000.0)
    timings.sort()
    return {
        "p50_ms": round(statistics.median(timings), 2),
        "p95_ms": round(timings[min(len(timings) - 1, int(0.95 * len(timings)))], 2),
        "runs": runs,
        "threads": threads,
        "delegate": "XNNPACK (LiteRT default on CPU)",
        "indicative": True,
    }


def check_latency(report: VerifyReport, model_path: Path, runs: int, threads: int, *, enforce: bool) -> None:
    latency = measure_latency(model_path, runs, threads)
    report.latency = latency
    within = latency["p50_ms"] <= MAX_LATENCY_P50_MS and latency["p95_ms"] <= MAX_LATENCY_P95_MS
    status = (PASS if within else FAIL) if enforce else INFO
    report.add(
        "latency",
        status,
        f"p50 {latency['p50_ms']} ms, p95 {latency['p95_ms']} ms on this CPU ({threads} threads); "
        f"budget on the reference phone {MAX_LATENCY_P50_MS}/{MAX_LATENCY_P95_MS} ms",
        latency,
    )


def check_golden(report: VerifyReport, model_path: Path, manifest: Path | None, min_accuracy: float) -> None:
    """Runs the model on a golden set (about 20 frames with known readings) and compares the parsed readings."""
    if manifest is None:
        report.add("golden", SKIP, "no golden set given (--golden manifest.jsonl | synthetic)")
        return
    from tensoscan_ml.eval.runner import evaluate_manifest

    sidecar = Sidecar.load(model_path)
    detector = ModelDetector(TfliteModel(model_path), f"tflite:{model_path.name}", sidecar)
    import tempfile

    with tempfile.TemporaryDirectory() as scratch:
        result = evaluate_manifest(manifest, detector, scratch, write_failures=False)
    frame = result["reading"]["frame"]
    accuracy = frame["accuracy"]
    ok = accuracy is not None and accuracy >= min_accuracy
    report.add(
        "golden",
        PASS if ok else FAIL,
        f"{frame['correct']}/{frame['frames']} readings reproduced (required {min_accuracy:.0%})",
        accuracy,
    )


def verify_model(
    model_path: str | Path,
    *,
    golden: Path | None = None,
    runs: int = 50,
    threads: int = 4,
    min_golden_accuracy: float = 0.9,
    enforce_latency: bool = False,
) -> VerifyReport:
    model = Path(model_path)
    if not model.is_file():
        raise FileNotFoundError(f"model '{model}' not found")
    report = VerifyReport(str(model), file_sha256(model))
    sidecar = check_sidecar(report, model, model.with_suffix(".json"))
    info = inspect_file(model)
    check_io(report, info, sidecar)
    check_ops_and_size(report, info)
    try:
        check_latency(report, model, runs, threads, enforce=enforce_latency)
        check_golden(report, model, golden, min_golden_accuracy)
    except Exception as error:
        report.add("runtime", FAIL, f"LiteRT could not run the model: {error}")
    return report
