"""Model-assisted pre-annotation of real frames for CVAT (REQ-B003-13, T-B003-12).

Selects about 300 frames spread over devices, conditions and videos, runs the best available detector with a low score
threshold and writes YOLO label files plus a CVAT "YOLO 1.1" annotations zip. A human then corrects the boxes in CVAT;
`check_annotations` then compares the corrected boxes with the known readings (through the official parser) to catch
annotation mistakes.
"""

from __future__ import annotations

import csv
import json
import shutil
import zipfile
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tensoscan_ml.contract import CLASS_INDEX, CLASSES
from tensoscan_ml.data.manifest import ManifestEntry, read_manifest, write_manifest
from tensoscan_ml.detector.runtime import Detector, JsonlDetections, ModelDetector
from tensoscan_ml.eval.metrics import read_yolo_boxes
from tensoscan_ml.reading import SymbolDetection, parse_detections

PRELABEL_SCORE_THRESHOLD = 0.25
DEFAULT_COUNT = 300


@dataclass
class PrelabelReport:
    selected: int = 0
    detections: int = 0
    empty_images: list[str] = field(default_factory=list)
    crowded_images: list[str] = field(default_factory=list)
    by_device: dict[str, int] = field(default_factory=dict)
    by_condition: dict[str, int] = field(default_factory=dict)


def select_frames(entries: Sequence[ManifestEntry], count: int = DEFAULT_COUNT, seed: int = 42) -> list[ManifestEntry]:
    """Round-robin over videos, taking evenly spaced frames from each, so devices and conditions all appear."""
    videos: dict[str, list[ManifestEntry]] = defaultdict(list)
    for entry in entries:
        videos[f"{entry.device}|{entry.condition}|{entry.video_id or entry.path}"].append(entry)
    order = sorted(videos)
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    queues: dict[str, list[ManifestEntry]] = {}
    for key in order:
        frames = sorted(videos[key], key=lambda e: (e.frame_idx or 0, e.path))
        queues[key] = _spread(frames)
    selected: list[ManifestEntry] = []
    while len(selected) < min(count, len(entries)):
        progressed = False
        for key in order:
            if queues[key] and len(selected) < count:
                selected.append(queues[key].pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def _spread(frames: list[ManifestEntry]) -> list[ManifestEntry]:
    """Frames reordered so that taking a prefix gives evenly spaced samples (bit-reversal style bisection)."""
    ordered: list[ManifestEntry] = []
    seen: set[int] = set()
    step = len(frames)
    while step >= 1 and len(ordered) < len(frames):
        for index in range(0, len(frames), max(step, 1)):
            if index not in seen:
                seen.add(index)
                ordered.append(frames[index])
        step //= 2
    return ordered


def yolo_lines(detections: Sequence[SymbolDetection]) -> list[str]:
    lines = []
    for detection in detections:
        box = detection.box
        lines.append(
            f"{CLASS_INDEX[detection.symbol]} {(box.left + box.right) / 2:.6f} {(box.top + box.bottom) / 2:.6f} "
            f"{box.width:.6f} {box.height:.6f}"
        )
    return lines


def prelabel(
    manifest_path: str | Path,
    source: Detector | JsonlDetections,
    out_dir: str | Path,
    *,
    count: int = DEFAULT_COUNT,
    seed: int = 42,
    score_threshold: float = PRELABEL_SCORE_THRESHOLD,
) -> PrelabelReport:
    """Writes `images/`, `labels/`, `selected.jsonl`, the CVAT YOLO 1.1 files and `confidences.csv` to `out_dir`."""
    manifest = Path(manifest_path)
    entries = read_manifest(manifest)
    if not entries:
        raise ValueError(f"{manifest} has no entries")
    target = Path(out_dir)
    (target / "images").mkdir(parents=True, exist_ok=True)
    (target / "labels").mkdir(parents=True, exist_ok=True)
    if isinstance(source, ModelDetector):
        source.score_threshold = score_threshold  # decode low: the human removes the false positives

    report = PrelabelReport()
    chosen: list[ManifestEntry] = []
    confidences: list[tuple[str, int, float, float]] = []
    names: list[str] = []
    for entry in select_frames(entries, count, seed):
        image_path = manifest.parent / entry.path
        name = f"{entry.video_id or Path(entry.path).stem}__{entry.frame_idx if entry.frame_idx is not None else 0:06d}"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"cannot read {image_path}")
        if isinstance(source, JsonlDetections):
            found = source.lookup(entry.path, entry.sha256) or []
        else:
            found = source.detect(np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), dtype=np.uint8))
        found = [d for d in found if d.confidence >= score_threshold]
        if image_path.suffix.lower() in (".jpg", ".jpeg"):
            shutil.copyfile(image_path, target / "images" / f"{name}.jpg")
        else:
            cv2.imwrite(str(target / "images" / f"{name}.jpg"), image)
        (target / "labels" / f"{name}.txt").write_text(
            "".join(line + "\n" for line in yolo_lines(found)), encoding="utf-8"
        )
        chosen.append(
            ManifestEntry(
                path=f"images/{name}.jpg",
                sha256=entry.sha256,
                device=entry.device,
                status=entry.status,
                sys=entry.sys,
                dia=entry.dia,
                pul=entry.pul,
                condition=entry.condition,
                video_id=entry.video_id,
                frame_idx=entry.frame_idx,
                labels=f"labels/{name}.txt",
                extra={**entry.extra, "prelabelled": True, "source_path": entry.path},
            )
        )
        names.append(name)
        report.selected += 1
        report.detections += len(found)
        report.by_device[entry.device] = report.by_device.get(entry.device, 0) + 1
        report.by_condition[str(entry.condition)] = report.by_condition.get(str(entry.condition), 0) + 1
        mean = float(np.mean([d.confidence for d in found])) if found else 0.0
        confidences.append(
            (name, len(found), round(mean, 3), round(min((d.confidence for d in found), default=0.0), 3))
        )
        if not found:
            report.empty_images.append(name)
        elif len(found) > 20:
            report.crowded_images.append(name)

    write_manifest(target / "selected.jsonl", chosen)
    _write_cvat_files(target, names)
    with (target / "confidences.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "detections", "mean_confidence", "min_confidence"])
        least_confident_first = sorted(confidences, key=lambda row: (row[2], row[0]))
        writer.writerows(least_confident_first)
    (target / "prelabel_report.json").write_text(
        json.dumps(vars(report), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _write_cvat_files(target: Path, names: list[str]) -> None:
    """CVAT "YOLO 1.1" annotations: `obj.names`, `obj.data`, `train.txt` and `obj_train_data/<name>.txt`."""
    (target / "obj.names").write_text("".join(f"{name}\n" for name in CLASSES), encoding="utf-8")
    (target / "obj.data").write_text(
        f"classes = {len(CLASSES)}\ntrain = data/train.txt\nnames = data/obj.names\nbackup = backup/\n",
        encoding="utf-8",
    )
    (target / "train.txt").write_text("".join(f"data/obj_train_data/{name}.jpg\n" for name in names), encoding="utf-8")
    with zipfile.ZipFile(target / "cvat_yolo11.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(target / "obj.names", "obj.names")
        archive.write(target / "obj.data", "obj.data")
        archive.write(target / "train.txt", "train.txt")
        for name in names:
            archive.write(target / "labels" / f"{name}.txt", f"obj_train_data/{name}.txt")


# --- checking the corrected annotations -----------------------------------------------------------------------


def check_annotations(selected_manifest: str | Path) -> dict[str, Any]:
    """Compares the annotated boxes of each frame with its known reading using the official parser.

    A mismatch does not always mean a wrong box (see finding F1 in the README: the parser may not join a "1" to its
    neighbour), but it is the quickest way to find a missing, wrong or misclassified glyph.
    """
    manifest = Path(selected_manifest)
    problems: list[dict[str, Any]] = []
    checked = ok = 0
    for entry in read_manifest(manifest):
        if not entry.labels:
            continue
        path = manifest.parent / entry.labels
        if not path.is_file():
            problems.append({"image": entry.path, "problem": "label file missing"})
            continue
        checked += 1
        boxes = read_yolo_boxes(path)
        outside = any(b.width <= 0 or b.height <= 0 or b.left < -1e-6 or b.right > 1 + 1e-6 for _, b in boxes)
        if outside or any(not 0 <= c < len(CLASSES) for c, _ in boxes):
            problems.append({"image": entry.path, "problem": "invalid class or box outside the image"})
            continue
        detections = [SymbolDetection(CLASSES[c], b, 1.0) for c, b in boxes]
        reading = parse_detections(detections)
        if entry.status == "COMPLETE":
            matches = reading.status.value == "COMPLETE" and reading.values == entry.reading
        else:
            matches = entry.status is None or reading.status.value == entry.status
        if matches:
            ok += 1
        else:
            problems.append(
                {
                    "image": entry.path,
                    "problem": "annotated boxes do not reproduce the known reading",
                    "expected": entry.status if entry.status != "COMPLETE" else list(entry.reading),
                    "parsed": {"status": reading.status.value, "values": list(reading.values)},
                }
            )
    return {"checked": checked, "reproduce_reading": ok, "problems": problems}
