"""Real-capture ingestion: videos and photos -> cropped, de-duplicated frames + manifest (REQ-B003-10, design 5).

Nothing is dropped silently: every rejected file, frame or CSV row ends in `ingest_report.json` with its reason.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from tensoscan_ml.data.manifest import ManifestEntry, write_manifest
from tensoscan_ml.data.naming import (
    CaptureLabel,
    NamingError,
    Roi,
    is_media,
    is_video,
    parse_labels_csv,
    parse_name,
)
from tensoscan_ml.data.phash import hamming, phash
from tensoscan_ml.provenance import combine_hashes, file_sha256

BgrImage = NDArray[np.uint8]
ROI_MODES = ("center", "none", "manual")

# Guide rectangle of the app (`RoiGeometry`, spec A-002 6.1).
ROI_WIDTH_FRACTION = 0.80
ROI_HEIGHT_TO_WIDTH = 1.25
ROI_MAX_HEIGHT_FRACTION = 0.60
ROI_CENTER_Y_FRACTION = 0.45


@dataclass
class IngestOptions:
    fps: float = 3.0
    blur_threshold: float = 60.0
    hash_distance: int = 4
    roi_mode: str = "center"
    max_side: int = 640
    jpeg_quality: int = 95
    drop_blurry_photos: bool = False
    dedupe_photos: bool = False
    preview: bool = False
    preview_limit: int = 48

    def __post_init__(self) -> None:
        if self.roi_mode not in ROI_MODES:
            raise ValueError(f"roi_mode must be one of {', '.join(ROI_MODES)}")
        if self.fps <= 0:
            raise ValueError("fps must be positive")


@dataclass
class Rejection:
    """Something that did not make it into the dataset, and why."""

    source: str
    code: str
    detail: str
    frame_idx: int | None = None


@dataclass
class SourceStats:
    kind: str
    frames_read: int = 0
    kept: int = 0
    blurry: int = 0
    duplicates: int = 0


@dataclass
class IngestReport:
    dataset_version: str
    options: dict[str, Any]
    sources: dict[str, SourceStats] = field(default_factory=dict)
    rejections: list[Rejection] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    entries: list[ManifestEntry] = field(default_factory=list)
    preview_path: str | None = None

    def summary(self) -> dict[str, Any]:
        kept = sum(stats.kept for stats in self.sources.values())
        return {
            "dataset_version": self.dataset_version,
            "files": len(self.sources),
            "frames_kept": kept,
            "frames_blurry": sum(stats.blurry for stats in self.sources.values()),
            "frames_duplicate": sum(stats.duplicates for stats in self.sources.values()),
            "rejections": len(self.rejections),
        }

    def to_json(self) -> str:
        payload = {
            "summary": self.summary(),
            "options": self.options,
            "sources": {name: vars(stats) for name, stats in sorted(self.sources.items())},
            "rejections": [vars(r) for r in self.rejections],
            "warnings": self.warnings,
            "preview": self.preview_path,
        }
        return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


# --- geometry and quality --------------------------------------------------------------------------------------


def center_roi(width: int, height: int) -> tuple[int, int, int, int]:
    """The app's guide rectangle as `(x, y, w, h)` pixels: 80 % of the width, 4:5, centred at 45 % of the height."""
    roi_w = ROI_WIDTH_FRACTION * width
    roi_h = min(ROI_HEIGHT_TO_WIDTH * roi_w, ROI_MAX_HEIGHT_FRACTION * height)
    center_y = min(max(ROI_CENTER_Y_FRACTION * height, roi_h / 2), height - roi_h / 2)
    left, top = (width - roi_w) / 2, center_y - roi_h / 2
    return round(left), round(top), max(1, round(roi_w)), max(1, round(roi_h))


def roi_rect(image: BgrImage, mode: str, manual: Roi | None) -> tuple[int, int, int, int]:
    height, width = image.shape[:2]
    if mode == "none":
        return 0, 0, width, height
    if mode == "manual":
        if manual is None:
            raise NamingError("MISSING_ROI", "roi-mode manual needs roi_x, roi_y, roi_w and roi_h in labels.csv")
        if manual.x + manual.width > width or manual.y + manual.height > height:
            raise NamingError("ROI_OUTSIDE", f"the ROI exceeds the {width}x{height} image")
        return manual.x, manual.y, manual.width, manual.height
    return center_roi(width, height)


def crop_and_scale(image: BgrImage, rect: tuple[int, int, int, int], max_side: int) -> BgrImage:
    x, y, w, h = rect
    crop = image[y : y + h, x : x + w]
    longest = max(crop.shape[:2])
    if longest > max_side:
        scale = max_side / longest
        crop = np.asarray(cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), dtype=np.uint8)
    return np.ascontiguousarray(crop)


def blur_score(image: BgrImage) -> float:
    """Variance of the Laplacian: the lower, the blurrier."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


# --- reading media ---------------------------------------------------------------------------------------------


def iter_video_frames(path: Path, fps: float) -> Iterator[tuple[int, BgrImage]]:
    """Frames of a video at about `fps` frames per second as `(source_frame_index, image)`."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise NamingError("UNREADABLE_VIDEO", f"OpenCV could not open '{path.name}'")
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        if not source_fps or source_fps != source_fps or source_fps <= 0:
            source_fps = 30.0
        step = max(1.0, source_fps / fps)
        next_pick = 0.0
        index = 0
        while True:
            ok = capture.grab()
            if not ok:
                break
            if index + 1e-9 >= next_pick:
                ok, frame = capture.retrieve()
                if not ok:
                    break
                yield index, np.asarray(frame, dtype=np.uint8)
                next_pick += step
            index += 1
    finally:
        capture.release()


def read_image(path: Path) -> BgrImage:
    """Reads a photo honouring its EXIF orientation (OpenCV does by default)."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise NamingError("UNREADABLE_IMAGE", f"OpenCV could not read '{path.name}'")
    return np.asarray(image, dtype=np.uint8)


# --- planning --------------------------------------------------------------------------------------------------


@dataclass
class Job:
    """A media file with what it shows."""

    path: Path
    label: CaptureLabel
    roi: Roi | None


def plan_jobs(src: Path, labels_csv: Path | None, report: IngestReport) -> list[Job]:
    """Pairs every media file under `src` with a label, from `labels.csv` or from the file name."""
    media = sorted(p for p in src.rglob("*") if p.is_file() and is_media(p))
    by_relative = {p.relative_to(src).as_posix(): p for p in media}
    jobs: list[Job] = []
    used: set[str] = set()

    if labels_csv is not None:
        rows, problems = parse_labels_csv(labels_csv)
        for problem in problems:
            used.add(Path(problem.file).as_posix())  # already reported: do not also try to parse its name
            report.rejections.append(
                Rejection(problem.file or f"{labels_csv.name}:{problem.line}", problem.code, problem.message)
            )
        for row in rows:
            key = Path(row.file).as_posix()
            path = (
                by_relative.get(key)
                or by_relative.get(Path(row.file).name)
                or _find_by_name(media, Path(row.file).name)
            )
            if path is None:
                report.rejections.append(
                    Rejection(row.file, "FILE_NOT_FOUND", f"'{row.file}' of labels.csv is not under {src}")
                )
                continue
            used.add(path.relative_to(src).as_posix())
            jobs.append(Job(path, row.label, row.roi))

    for path in media:
        relative = path.relative_to(src).as_posix()
        if relative in used:
            continue
        try:
            jobs.append(Job(path, parse_name(path.name), None))
        except NamingError as error:
            report.rejections.append(Rejection(relative, error.code, error.message))
    return jobs


def _find_by_name(media: list[Path], name: str) -> Path | None:
    matches = [p for p in media if p.name == name]
    return matches[0] if len(matches) == 1 else None


# --- the ingestion itself --------------------------------------------------------------------------------------


def ingest(
    src: str | Path,
    out: str | Path,
    options: IngestOptions,
    *,
    labels_csv: str | Path | None = None,
    dataset_version: str = "ds-real",
) -> IngestReport:
    """Ingests `src` into `out` (frames + `manifest.jsonl` + `ingest_report.json`). With `options.preview` it only
    writes `preview.jpg` and the report."""
    source = Path(src)
    target = Path(out)
    if not source.is_dir():
        raise FileNotFoundError(f"source directory '{source}' does not exist")
    csv_path = (
        Path(labels_csv) if labels_csv else (source / "labels.csv" if (source / "labels.csv").is_file() else None)
    )
    report = IngestReport(dataset_version, dict(vars(options)))
    jobs = plan_jobs(source, csv_path, report)
    if not jobs and not report.rejections:
        raise ValueError(f"no videos or images found under '{source}'")

    target.mkdir(parents=True, exist_ok=True)
    stem_counts = Counter(job.path.stem for job in jobs)
    previews: list[tuple[BgrImage, BgrImage, str]] = []
    for job in jobs:
        relative = job.path.relative_to(source).as_posix()
        try:
            _process(job, relative, stem_counts[job.path.stem] > 1, target, options, report, previews)
        except NamingError as error:
            report.rejections.append(Rejection(relative, error.code, error.message))
            report.sources.pop(relative, None)

    if options.preview:
        sheet = contact_sheet(previews[: options.preview_limit])
        preview_path = target / "preview.jpg"
        if sheet is not None:
            cv2.imwrite(str(preview_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
            report.preview_path = preview_path.name
        report.entries.clear()
    else:
        write_manifest(target / "manifest.jsonl", report.entries)
    (target / "ingest_report.json").write_text(report.to_json(), encoding="utf-8")
    return report


def _process(
    job: Job,
    relative: str,
    stem_clash: bool,
    target: Path,
    options: IngestOptions,
    report: IngestReport,
    previews: list[tuple[BgrImage, BgrImage, str]],
) -> None:
    video = is_video(job.path)
    stats = SourceStats("video" if video else "image")
    report.sources[relative] = stats
    video_id = relative.replace("/", "_").rsplit(".", 1)[0] if stem_clash else job.path.stem
    frames: Iterator[tuple[int, BgrImage]] = (
        iter_video_frames(job.path, options.fps) if video else iter([(0, read_image(job.path))])
    )
    kept_hashes: list[int] = []
    source_hash = file_sha256(job.path)

    for frame_idx, frame in frames:
        stats.frames_read += 1
        rect = roi_rect(frame, options.roi_mode, job.roi)
        crop = crop_and_scale(frame, rect, options.max_side)
        if options.preview:  # one frame per file is enough to check the framing
            if len(previews) < options.preview_limit:
                previews.append((_with_roi(frame, rect), crop, _caption(job.label)))
            stats.kept += 1
            break
        score = blur_score(crop)
        if score < options.blur_threshold and (video or options.drop_blurry_photos):
            stats.blurry += 1
            report.rejections.append(
                Rejection(relative, "BLURRY", f"Laplacian variance {score:.1f} < {options.blur_threshold}", frame_idx)
            )
            continue
        if score < options.blur_threshold:
            report.warnings.append(f"{relative}: blurry photo kept (Laplacian variance {score:.1f})")
        if video or options.dedupe_photos:
            digest = phash(crop)
            if any(hamming(digest, other) <= options.hash_distance for other in kept_hashes):
                stats.duplicates += 1
                report.rejections.append(
                    Rejection(relative, "DUPLICATE", f"pHash within {options.hash_distance} of a kept frame", frame_idx)
                )
                continue
            kept_hashes.append(digest)
        frame_rel = f"frames/{video_id}/{frame_idx:06d}.jpg"
        destination = target / frame_rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(destination), crop, [cv2.IMWRITE_JPEG_QUALITY, options.jpeg_quality])
        stats.kept += 1
        report.entries.append(
            _entry(frame_rel, destination, job, video_id, frame_idx, source_hash, score, rect, options)
        )

    if not options.preview and stats.kept == 0:
        report.rejections.append(Rejection(relative, "NO_FRAMES_KEPT", "every frame of the file was rejected"))


def _entry(
    frame_rel: str,
    destination: Path,
    job: Job,
    video_id: str,
    frame_idx: int,
    source_hash: str,
    score: float,
    rect: tuple[int, int, int, int],
    options: IngestOptions,
) -> ManifestEntry:
    label = job.label
    return ManifestEntry(
        path=frame_rel,
        sha256=file_sha256(destination),
        device=label.device,
        status=label.status,
        sys=label.sys,
        dia=label.dia,
        pul=label.pul,
        condition=label.condition,
        video_id=video_id,
        frame_idx=frame_idx,
        extra={
            "source": job.path.name,
            "source_sha256": source_hash,
            "roi_mode": options.roi_mode,
            "roi": list(rect),
            "blur_score": round(score, 1),
        },
    )


# --- preview ---------------------------------------------------------------------------------------------------


def _caption(label: CaptureLabel) -> str:
    values = "error" if label.sys is None else f"{label.sys}/{label.dia}/{label.pul}"
    return f"{label.device} {values} {label.condition}"


def _with_roi(frame: BgrImage, rect: tuple[int, int, int, int]) -> BgrImage:
    """The full image with the ROI rectangle drawn on it."""
    marked = frame.copy()
    x, y, w, h = rect
    thickness = max(2, round(max(frame.shape[:2]) / 250))
    cv2.rectangle(marked, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), thickness)
    return np.asarray(marked, dtype=np.uint8)


def contact_sheet(items: list[tuple[BgrImage, BgrImage, str]], tile: int = 220, columns: int = 4) -> BgrImage | None:
    """Pairs of (original with its ROI, the crop) with the label of each, for checking the framing at a glance."""
    if not items:
        return None
    cell_w, cell_h = tile * 2 + 6, tile + 22
    rows = -(-len(items) // columns)
    sheet = np.full((rows * cell_h, columns * cell_w, 3), 255, dtype=np.uint8)
    for index, (original, crop, caption) in enumerate(items):
        row, column = divmod(index, columns)
        x0, y0 = column * cell_w, row * cell_h
        for offset, image in enumerate((original, crop)):
            scale = min(tile / image.shape[1], tile / image.shape[0])
            small = np.asarray(
                cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), dtype=np.uint8
            )
            sheet[
                y0 + 20 : y0 + 20 + small.shape[0], x0 + offset * (tile + 6) : x0 + offset * (tile + 6) + small.shape[1]
            ] = small
        cv2.putText(sheet, caption[:44], (x0 + 2, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return sheet


def dataset_hash(entries: list[ManifestEntry]) -> str:
    """Hash of the dataset: sha256 over the sorted `(path, sha256)` pairs."""
    return combine_hashes((entry.path, entry.sha256) for entry in entries)


def copy_manifest(manifest: Path, directory: Path, name: str) -> Path:
    """Copies the manifest to the committed manifests directory (`ml/datasets/manifests/<name>.jsonl`)."""
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{name}.jsonl"
    shutil.copyfile(manifest, destination)
    return destination
