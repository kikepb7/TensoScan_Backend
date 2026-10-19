"""Metrics of the evaluation harness (requirements section 1 and design section 6)."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from tensoscan_ml.contract import CLASS_INDEX, CLASSES
from tensoscan_ml.data.manifest import ManifestEntry
from tensoscan_ml.reading import Box, FrameReading, FrameStatus, SymbolDetection, parse_detections

#: Acceptance thresholds of requirements section 1.
ACCEPTANCE: Final = {
    "reading_exact_match_frame": 0.90,
    "reading_exact_match_video": 0.97,
    "reading_exact_match_worst_device": 0.85,
    "map50_real": 0.90,
    "map50_synthetic": 0.95,
}
IOU_THRESHOLDS: Final = tuple(round(0.5 + 0.05 * step, 2) for step in range(10))
APP_SCORE_THRESHOLD: Final = 0.40
APP_MAX_DETECTIONS: Final = 32
PRODUCTION_MIN_DEVICES: Final = 3


@dataclass
class FrameResult:
    """One evaluated image."""

    entry: ManifestEntry
    detections: list[SymbolDetection]  # decoded with a low threshold (mAP uses all of them)
    reading: FrameReading
    ground_truth: list[tuple[int, Box]] | None = None
    correct: bool | None = None
    wrong_fields: int = 0

    @property
    def mean_confidence(self) -> float:
        used = [d.confidence for d in self.detections if d.confidence >= APP_SCORE_THRESHOLD]
        return float(np.mean(used)) if used else 0.0


def app_view(detections: Sequence[SymbolDetection]) -> list[SymbolDetection]:
    """What the app would see: score threshold 0.40 and at most 32 detections, best first."""
    kept = sorted((d for d in detections if d.confidence >= APP_SCORE_THRESHOLD), key=lambda d: -d.confidence)
    return kept[:APP_MAX_DETECTIONS]


def read_yolo_boxes(path: Path) -> list[tuple[int, Box]]:
    boxes: list[tuple[int, Box]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, cx, cy, w, h = int(parts[0]), *(float(v) for v in parts[1:5])
        boxes.append((cls, Box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
    return boxes


def judge(entry: ManifestEntry, reading: FrameReading) -> tuple[bool | None, int]:
    """Is the parsed reading what the label says? Returns `(correct, wrong_fields)`; `None` without ground truth."""
    if entry.status is None:
        return None, 0
    if entry.status != "COMPLETE":
        return reading.status.value == entry.status, 0 if reading.status.value == entry.status else 1
    expected = (entry.sys, entry.dia, entry.pul)
    wrong = sum(1 for got, want in zip(reading.values, expected, strict=True) if got != want)
    return reading.status is FrameStatus.COMPLETE and wrong == 0, wrong


def make_result(
    entry: ManifestEntry, detections: list[SymbolDetection], ground_truth: list[tuple[int, Box]] | None
) -> FrameResult:
    reading = parse_detections(app_view(detections))
    correct, wrong = judge(entry, reading)
    return FrameResult(entry, detections, reading, ground_truth, correct, wrong)


# --- reading accuracy ------------------------------------------------------------------------------------------


def _accuracy(results: Sequence[FrameResult]) -> dict[str, Any]:
    total = len(results)
    right = sum(1 for r in results if r.correct)
    return {"frames": total, "correct": right, "accuracy": right / total if total else None}


def reading_metrics(results: Sequence[FrameResult]) -> dict[str, Any]:
    """Exact-match accuracy per frame, per video (majority vote of COMPLETE frames), device, condition and status."""
    readings = [r for r in results if r.entry.status == "COMPLETE"]
    others = [r for r in results if r.entry.status not in (None, "COMPLETE")]

    by_video: dict[str, list[FrameResult]] = defaultdict(list)
    for result in readings:
        by_video[result.entry.video_id or result.entry.path].append(result)
    video_correct = 0
    for frames in by_video.values():
        votes = Counter(f.reading.values for f in frames if f.reading.status is FrameStatus.COMPLETE)
        if votes and votes.most_common(1)[0][0] == frames[0].entry.reading:
            video_correct += 1

    def grouped(key: Any, subset: Sequence[FrameResult]) -> dict[str, Any]:
        groups: dict[str, list[FrameResult]] = defaultdict(list)
        for result in subset:
            groups[str(key(result.entry))].append(result)
        return {name: _accuracy(items) for name, items in sorted(groups.items())}

    by_device = grouped(lambda e: e.device, readings)
    device_scores = [v["accuracy"] for v in by_device.values() if v["accuracy"] is not None]
    return {
        "frame": _accuracy(readings),
        "video": {
            "videos": len(by_video),
            "correct": video_correct,
            "accuracy": video_correct / len(by_video) if by_video else None,
        },
        "by_device": by_device,
        "worst_device_accuracy": min(device_scores) if device_scores else None,
        "by_condition": grouped(lambda e: e.condition, readings),
        "by_status": grouped(lambda e: e.status, others),
        "field_errors": {
            "any": sum(1 for r in readings if not r.correct),
            "one_field": sum(1 for r in readings if not r.correct and r.wrong_fields == 1),
            "two_or_more": sum(1 for r in readings if not r.correct and r.wrong_fields >= 2),
        },
    }


# --- detection quality (needs boxes) ---------------------------------------------------------------------------


def _iou_matrix(predicted: Sequence[Box], truth: Sequence[Box]) -> np.ndarray[Any, Any]:
    matrix = np.zeros((len(predicted), len(truth)), dtype=np.float64)
    for i, p in enumerate(predicted):
        for j, t in enumerate(truth):
            width = min(p.right, t.right) - max(p.left, t.left)
            height = min(p.bottom, t.bottom) - max(p.top, t.top)
            if width > 0 and height > 0:
                inter = width * height
                matrix[i, j] = inter / (p.width * p.height + t.width * t.height - inter)
    return matrix


def _average_precision(scores: list[float], matched: list[bool], total_truth: int) -> float:
    """101-point interpolated AP of one class at one IoU threshold."""
    if total_truth == 0:
        return float("nan")
    if not scores:
        return 0.0
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    hits = np.asarray(matched, dtype=np.float64)[order]
    true_positives = np.cumsum(hits)
    false_positives = np.cumsum(1.0 - hits)
    recall = true_positives / total_truth
    precision = true_positives / np.maximum(true_positives + false_positives, 1e-12)
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])
    points = np.linspace(0.0, 1.0, 101)
    interpolated = [
        float(precision[np.searchsorted(recall, point, side="left")])
        if point <= (recall[-1] if len(recall) else 0)
        else 0.0
        for point in points
    ]
    return float(np.mean(interpolated))


def detection_metrics(results: Sequence[FrameResult]) -> dict[str, Any] | None:
    """mAP@0.5, mAP@0.5:0.95, per-class AP@0.5 and the glyph confusion matrix, over images that have boxes."""
    labelled = [r for r in results if r.ground_truth is not None]
    if not labelled:
        return None
    classes = len(CLASSES)
    ap_by_threshold: dict[float, list[float]] = {t: [] for t in IOU_THRESHOLDS}
    per_class_ap50: dict[str, float | None] = {}
    for cls in range(classes):
        for threshold in IOU_THRESHOLDS:
            scores: list[float] = []
            matched: list[bool] = []
            total_truth = 0
            for result in labelled:
                assert result.ground_truth is not None  # noqa: S101  (filtered above)
                truth = [box for c, box in result.ground_truth if c == cls]
                total_truth += len(truth)
                predicted = sorted(
                    (d for d in result.detections if CLASS_INDEX[d.symbol] == cls), key=lambda d: -d.confidence
                )
                used: set[int] = set()
                ious = _iou_matrix([d.box for d in predicted], truth)
                for i, detection in enumerate(predicted):
                    best, best_iou = -1, threshold
                    for j in range(len(truth)):
                        if j not in used and ious[i, j] >= best_iou:
                            best, best_iou = j, ious[i, j]
                    scores.append(detection.confidence)
                    matched.append(best >= 0)
                    if best >= 0:
                        used.add(best)
            ap = _average_precision(scores, matched, total_truth)
            if not np.isnan(ap):
                ap_by_threshold[threshold].append(ap)
                if threshold == IOU_THRESHOLDS[0]:
                    per_class_ap50[CLASSES[cls]] = ap
            elif threshold == IOU_THRESHOLDS[0]:
                per_class_ap50[CLASSES[cls]] = None
    map50 = float(np.mean(ap_by_threshold[IOU_THRESHOLDS[0]])) if ap_by_threshold[IOU_THRESHOLDS[0]] else None
    all_maps = [float(np.mean(v)) for v in ap_by_threshold.values() if v]
    return {
        "images": len(labelled),
        "map50": map50,
        "map50_95": float(np.mean(all_maps)) if all_maps else None,
        "ap50_by_class": per_class_ap50,
        "confusion": confusion_matrix(labelled),
    }


def confusion_matrix(results: Sequence[FrameResult]) -> dict[str, Any]:
    """Rows: true class, columns: predicted class; the last row/column is `background` (missed / spurious)."""
    size = len(CLASSES) + 1
    matrix = np.zeros((size, size), dtype=np.int64)
    for result in results:
        assert result.ground_truth is not None  # noqa: S101
        predicted = app_view(result.detections)
        ious = _iou_matrix([d.box for d in predicted], [box for _, box in result.ground_truth])
        taken: set[int] = set()
        for j, (cls, _) in enumerate(result.ground_truth):
            candidates = [(ious[i, j], i) for i in range(len(predicted)) if i not in taken and ious[i, j] >= 0.5]
            if candidates:
                _, i = max(candidates)
                taken.add(i)
                matrix[cls, CLASS_INDEX[predicted[i].symbol]] += 1
            else:
                matrix[cls, size - 1] += 1
        for i, detection in enumerate(predicted):
            if i not in taken:
                matrix[size - 1, CLASS_INDEX[detection.symbol]] += 1
    return {"labels": [*CLASSES, "background"], "matrix": matrix.tolist()}


# --- assessment ------------------------------------------------------------------------------------------------


@dataclass
class Assessment:
    checks: dict[str, dict[str, Any]] = field(default_factory=dict)
    channel: str = "beta"
    notes: list[str] = field(default_factory=list)


def assess(reading: dict[str, Any], detection: dict[str, Any] | None, devices: int, synthetic: bool) -> Assessment:
    """Compares the metrics with the acceptance thresholds. Synthetic data never counts for acceptance (C-ML-04)."""
    result = Assessment(channel="production-eligible" if devices >= PRODUCTION_MIN_DEVICES else "beta")

    def check(name: str, value: float | None, threshold: float) -> None:
        result.checks[name] = {"value": value, "threshold": threshold, "pass": value is not None and value >= threshold}

    check("reading_exact_match_frame", reading["frame"]["accuracy"], ACCEPTANCE["reading_exact_match_frame"])
    check("reading_exact_match_video", reading["video"]["accuracy"], ACCEPTANCE["reading_exact_match_video"])
    check(
        "reading_exact_match_worst_device",
        reading["worst_device_accuracy"],
        ACCEPTANCE["reading_exact_match_worst_device"],
    )
    if detection is not None:
        key = "map50_synthetic" if synthetic else "map50_real"
        check(key, detection["map50"], ACCEPTANCE[key])
    if synthetic:
        result.notes.append(
            "Synthetic data: these numbers do not count for acceptance (C-ML-04); evaluate on real data."
        )
    if devices < PRODUCTION_MIN_DEVICES:
        result.notes.append(f"{devices} device(s): metrics are optimistic (beta channel, no unseen devices).")
    return result
