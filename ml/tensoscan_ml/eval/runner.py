"""Runs a detector over a manifest and writes `report.json`, `report.md` and the worst failures (REQ-B003-16)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tensoscan_ml.contract import CLASSES
from tensoscan_ml.data.manifest import ManifestEntry, read_manifest
from tensoscan_ml.detector.runtime import Detector, JsonlDetections
from tensoscan_ml.eval.metrics import (
    FrameResult,
    app_view,
    assess,
    detection_metrics,
    make_result,
    read_yolo_boxes,
    reading_metrics,
)
from tensoscan_ml.reading import SymbolDetection

WORST_EXAMPLES = 20


def evaluate_manifest(
    manifest_path: str | Path,
    source: Detector | JsonlDetections,
    out_dir: str | Path,
    *,
    limit: int | None = None,
    write_failures: bool = True,
) -> dict[str, Any]:
    """Evaluates every entry of the manifest with `source` and writes the reports to `out_dir`."""
    manifest = Path(manifest_path)
    entries = read_manifest(manifest)[:limit]
    if not entries:
        raise ValueError(f"{manifest} has no entries")
    results: list[FrameResult] = []
    missing: list[str] = []
    for entry in entries:
        image_path = manifest.parent / entry.path
        detections = _detections_for(source, entry, image_path)
        if detections is None:
            missing.append(entry.path)
            continue
        truth = (
            read_yolo_boxes(manifest.parent / entry.labels)
            if entry.labels and (manifest.parent / entry.labels).is_file()
            else None
        )
        results.append(make_result(entry, detections, truth))
    if missing:
        raise ValueError(f"{len(missing)} image(s) have no detections in the source, e.g. {missing[0]}")

    reading = reading_metrics(results)
    detection = detection_metrics(results)
    devices = sorted({e.device for e in entries})
    synthetic = devices == ["synthetic"]
    assessment = assess(reading, detection, len(devices), synthetic)
    report: dict[str, Any] = {
        "detector": source.name,
        "manifest": str(manifest),
        "images": len(results),
        "devices": devices,
        "synthetic": synthetic,
        "reading": reading,
        "detection": detection,
        "assessment": asdict(assessment),
        "worst": _worst(results),
    }
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    (target / "report.md").write_text(render_markdown(report), encoding="utf-8")
    _write_detections(target / "detections.jsonl", results)
    if write_failures:
        _write_failures(target / "failures", manifest.parent, results, report["worst"])
    return report


def _detections_for(
    source: Detector | JsonlDetections, entry: ManifestEntry, image_path: Path
) -> list[SymbolDetection] | None:
    if isinstance(source, JsonlDetections):
        return source.lookup(entry.path, entry.sha256)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"cannot read image {image_path}")
    return source.detect(np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), dtype=np.uint8))


def _worst(results: list[FrameResult]) -> list[dict[str, Any]]:
    wrong = [r for r in results if r.correct is False]
    wrong.sort(key=lambda r: (-r.wrong_fields, r.mean_confidence))
    return [
        {
            "path": r.entry.path,
            "device": r.entry.device,
            "condition": r.entry.condition,
            "expected": r.entry.status if r.entry.status != "COMPLETE" else list(r.entry.reading),
            "got": {"status": r.reading.status.value, "values": list(r.reading.values)},
        }
        for r in wrong[:WORST_EXAMPLES]
    ]


def _write_detections(path: Path, results: list[FrameResult]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            payload = {
                "path": result.entry.path,
                "detections": [
                    {
                        "symbol": d.symbol,
                        "box": [d.box.left, d.box.top, d.box.right, d.box.bottom],
                        "confidence": round(d.confidence, 5),
                    }
                    for d in result.detections
                ],
            }
            handle.write(json.dumps(payload) + "\n")


def _write_failures(directory: Path, root: Path, results: list[FrameResult], worst: list[dict[str, Any]]) -> None:
    if not worst:
        return
    directory.mkdir(parents=True, exist_ok=True)
    by_path = {r.entry.path: r for r in results}
    for rank, item in enumerate(worst, start=1):
        result = by_path[item["path"]]
        image = cv2.imread(str(root / result.entry.path))
        if image is None:
            continue
        height, width = image.shape[:2]
        for detection in app_view(result.detections):
            p1 = (round(detection.box.left * width), round(detection.box.top * height))
            p2 = (round(detection.box.right * width), round(detection.box.bottom * height))
            cv2.rectangle(image, p1, p2, (0, 200, 0), 1)
            cv2.putText(
                image,
                detection.symbol,
                (p1[0], max(12, p1[1] - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        label = f"want {item['expected']} got {item['got']['values']} {item['got']['status']}"
        cv2.putText(image, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(directory / f"{rank:02d}.jpg"), image)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f} %"


def render_markdown(report: dict[str, Any]) -> str:
    reading, detection, assessment = report["reading"], report["detection"], report["assessment"]
    lines = [
        "# Evaluation report",
        "",
        f"- Detector: `{report['detector']}`",
        f"- Manifest: `{report['manifest']}` ({report['images']} images, devices: {', '.join(report['devices'])})",
        f"- Channel: **{assessment['channel']}**",
        "",
        "## Acceptance thresholds",
        "",
        "| Metric | Value | Threshold | Result |",
        "|--------|-------|-----------|--------|",
    ]
    for name, check in assessment["checks"].items():
        lines.append(
            f"| {name} | {_pct(check['value'])} | {_pct(check['threshold'])} | {'pass' if check['pass'] else 'FAIL'} |"
        )
    lines += [f"- {note}" for note in assessment["notes"]]
    lines += [
        "",
        "## Reading accuracy",
        "",
        f"- Per frame: {_pct(reading['frame']['accuracy'])} "
        f"({reading['frame']['correct']}/{reading['frame']['frames']})",
        f"- Per video (majority of COMPLETE frames): {_pct(reading['video']['accuracy'])} "
        f"({reading['video']['correct']}/{reading['video']['videos']})",
        f"- Worst device: {_pct(reading['worst_device_accuracy'])}",
        "",
        "| Device | Frames | Accuracy |",
        "|--------|--------|----------|",
    ]
    lines += [f"| {name} | {v['frames']} | {_pct(v['accuracy'])} |" for name, v in reading["by_device"].items()]
    lines += ["", "| Condition | Frames | Accuracy |", "|-----------|--------|----------|"]
    lines += [f"| {name} | {v['frames']} | {_pct(v['accuracy'])} |" for name, v in reading["by_condition"].items()]
    if reading["by_status"]:
        lines += ["", "| Status shown | Frames | Accuracy |", "|--------------|--------|----------|"]
        lines += [f"| {name} | {v['frames']} | {_pct(v['accuracy'])} |" for name, v in reading["by_status"].items()]
    if detection is not None:
        lines += [
            "",
            "## Glyph detection",
            "",
            f"- mAP@0.5: {_pct(detection['map50'])} · mAP@0.5:0.95: {_pct(detection['map50_95'])} "
            f"({detection['images']} images with boxes)",
            "",
            "| Class | AP@0.5 |",
            "|-------|--------|",
        ]
        lines += [f"| {name} | {_pct(ap)} |" for name, ap in detection["ap50_by_class"].items()]
        labels = detection["confusion"]["labels"]
        lines += [
            "",
            "### Confusion matrix (rows: truth, columns: prediction)",
            "",
            "| | " + " | ".join(labels) + " |",
            "|" + "---|" * (len(labels) + 1),
        ]
        for label, row in zip(labels, detection["confusion"]["matrix"], strict=True):
            lines.append(f"| **{label}** | " + " | ".join(str(v) for v in row) + " |")
    if report["worst"]:
        lines += [
            "",
            f"## Worst {len(report['worst'])} images",
            "",
            "| Image | Expected | Got |",
            "|-------|----------|-----|",
        ]
        lines += [
            f"| {w['path']} | {w['expected']} | {w['got']['values']} {w['got']['status']} |" for w in report["worst"]
        ]
    return "\n".join(lines) + "\n"


__all__ = ["CLASSES", "evaluate_manifest", "render_markdown"]
