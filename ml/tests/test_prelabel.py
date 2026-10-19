"""T-B003-12: model-assisted pre-annotation for CVAT and the check of the corrected boxes (REQ-B003-13)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tensoscan_ml.cli import main
from tensoscan_ml.contract import CLASSES
from tensoscan_ml.data.manifest import ManifestEntry, read_manifest, write_manifest
from tensoscan_ml.data.prelabel import check_annotations, prelabel, select_frames, yolo_lines
from tensoscan_ml.detector.preprocess import letterbox_transform
from tensoscan_ml.detector.runtime import JsonlDetections, ModelDetector, Sidecar
from tensoscan_ml.reading import Box, SymbolDetection
from tensoscan_ml.synth.params import load_synth_config
from tensoscan_ml.synth.writer import write_dataset


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("prelabel")
    config = load_synth_config(overrides={"scene": {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}})
    write_dataset(root, config, {"val": 40}, seed=31)
    entries = [
        e
        for e in read_manifest(root / "val" / "manifest.jsonl")
        if e.status == "COMPLETE" and e.extra["readable"] and e.extra["parser_compatible"]
    ]
    write_manifest(root / "val" / "good.jsonl", entries)
    assert len(entries) >= 8
    return root / "val"


def _perfect_jsonl(dataset: Path, target: Path, confidence: float = 0.9) -> Path:
    with target.open("w", encoding="utf-8") as handle:
        for entry in read_manifest(dataset / "good.jsonl"):
            detections = []
            for line in (dataset / str(entry.labels)).read_text(encoding="utf-8").splitlines():
                cls, cx, cy, w, h = line.split()
                box = [
                    float(cx) - float(w) / 2,
                    float(cy) - float(h) / 2,
                    float(cx) + float(w) / 2,
                    float(cy) + float(h) / 2,
                ]
                detections.append({"symbol": CLASSES[int(cls)], "box": box, "confidence": confidence})
            handle.write(json.dumps({"path": entry.path, "detections": detections}) + "\n")
    return target


def _video_entries() -> list[ManifestEntry]:
    entries = []
    for device, condition, video, frames in (("a", "good", "va", 9), ("a", "dim", "vb", 5), ("b", "glare", "vc", 3)):
        for index in range(frames):
            entries.append(
                ManifestEntry(
                    f"frames/{video}/{index * 10:06d}.jpg",
                    "0" * 64,
                    device,
                    "COMPLETE",
                    128,
                    82,
                    67,
                    condition,
                    video,
                    index * 10,
                )
            )
    return entries


def test_selection_is_round_robin_over_videos_so_every_device_and_condition_appears() -> None:
    selected = select_frames(_video_entries(), count=6, seed=1)

    assert len(selected) == 6
    assert {e.video_id for e in selected} == {"va", "vb", "vc"}
    assert {e.device for e in selected} == {"a", "b"}
    assert sorted(e.video_id for e in selected).count("va") == 2


def test_frames_of_a_video_are_spread_along_it_not_taken_from_the_start() -> None:
    selected = [e for e in select_frames(_video_entries(), count=100, seed=1) if e.video_id == "va"]

    assert [e.frame_idx for e in selected][:3] == [0, 40, 80]  # start, middle, end before filling in the gaps


def test_selection_is_reproducible_and_never_exceeds_what_exists() -> None:
    first = [e.path for e in select_frames(_video_entries(), count=7, seed=5)]
    again = [e.path for e in select_frames(_video_entries(), count=7, seed=5)]

    assert first == again
    assert len(select_frames(_video_entries(), count=500)) == 17
    assert len({e.path for e in select_frames(_video_entries(), count=500)}) == 17


def test_yolo_lines_use_the_contract_class_indices() -> None:
    lines = yolo_lines(
        [SymbolDetection("8", Box(0.1, 0.2, 0.3, 0.6), 0.9), SymbolDetection("colon", Box(0.4, 0.2, 0.5, 0.6), 0.8)]
    )

    assert lines[0].split()[0] == "8" and lines[1].split()[0] == "10"
    assert [float(v) for v in lines[0].split()[1:]] == pytest.approx([0.2, 0.4, 0.2, 0.4])


def test_prelabel_writes_images_yolo_labels_cvat_files_and_a_usable_manifest(dataset: Path, tmp_path: Path) -> None:
    source = JsonlDetections(_perfect_jsonl(dataset, tmp_path / "det.jsonl"))
    out = tmp_path / "pre"

    report = prelabel(dataset / "good.jsonl", source, out, count=5)

    assert report.selected == 5 and report.detections > 5
    assert len(list((out / "images").glob("*.jpg"))) == 5 and len(list((out / "labels").glob("*.txt"))) == 5
    assert (out / "obj.names").read_text(encoding="utf-8").split() == list(CLASSES)
    assert "classes = 14" in (out / "obj.data").read_text(encoding="utf-8")
    names = [line.split("/")[-1] for line in (out / "train.txt").read_text(encoding="utf-8").split()]
    assert len(names) == 5 and all(n.endswith(".jpg") for n in names)
    with zipfile.ZipFile(out / "cvat_yolo11.zip") as archive:
        members = set(archive.namelist())
    assert {"obj.names", "obj.data", "train.txt"} <= members
    assert sum(1 for m in members if m.startswith("obj_train_data/")) == 5
    selected = read_manifest(out / "selected.jsonl")
    assert len(selected) == 5 and all(
        e.labels and (out / e.labels).is_file() and (out / e.path).is_file() for e in selected
    )
    assert (out / "confidences.csv").read_text(encoding="utf-8").startswith("image,detections")


def test_detections_below_the_prelabel_threshold_are_left_out(dataset: Path, tmp_path: Path) -> None:
    low = JsonlDetections(_perfect_jsonl(dataset, tmp_path / "low.jsonl", confidence=0.2))

    report = prelabel(dataset / "good.jsonl", low, tmp_path / "pre", count=3, score_threshold=0.25)

    assert report.detections == 0 and len(report.empty_images) == 3


def test_a_model_detector_is_decoded_with_the_low_prelabel_threshold(dataset: Path, tmp_path: Path) -> None:
    transform = letterbox_transform(384, 480)

    class OneBox:
        def run(self, model_input: np.ndarray) -> np.ndarray:
            rows = np.zeros((2100, 18), dtype=np.float32)
            rows[0, :4] = [160 / 320, 160 / 320, 40 / 320, 60 / 320]
            rows[0, 4 + 5] = 0.30  # between the prelabel threshold (0.25) and the app's (0.40)
            del model_input
            return rows[None]

    detector = ModelDetector(OneBox(), "one", Sidecar.load(tmp_path / "x.tflite"))
    assert transform.scaled_width == 256

    report = prelabel(dataset / "good.jsonl", detector, tmp_path / "pre", count=2, score_threshold=0.25)

    assert report.detections == 2  # one box per frame, kept although the app would drop it


def test_corrected_annotations_that_reproduce_the_readings_pass_the_check(dataset: Path, tmp_path: Path) -> None:
    out = tmp_path / "pre"
    prelabel(dataset / "good.jsonl", JsonlDetections(_perfect_jsonl(dataset, tmp_path / "det.jsonl")), out, count=6)

    result = check_annotations(out / "selected.jsonl")

    assert result["checked"] == 6 and result["reproduce_reading"] == 6 and result["problems"] == []


def test_the_check_flags_missing_digits_bad_classes_and_missing_files(dataset: Path, tmp_path: Path) -> None:
    out = tmp_path / "pre"
    prelabel(dataset / "good.jsonl", JsonlDetections(_perfect_jsonl(dataset, tmp_path / "det.jsonl")), out, count=6)
    selected = read_manifest(out / "selected.jsonl")
    first, second, third = (out / str(e.labels) for e in selected[:3])
    first.write_text(
        "".join(first.read_text(encoding="utf-8").splitlines(keepends=True)[:-3]), encoding="utf-8"
    )  # glyphs lost
    second.write_text("99 0.5 0.5 0.1 0.1\n", encoding="utf-8")  # class out of range
    third.unlink()

    result = check_annotations(out / "selected.jsonl")

    problems = {p["image"]: p["problem"] for p in result["problems"]}
    assert problems[selected[0].path] == "annotated boxes do not reproduce the known reading"
    assert problems[selected[1].path] == "invalid class or box outside the image"
    assert problems[selected[2].path] == "label file missing"
    assert result["reproduce_reading"] == 3


def test_cli_prelabels_then_checks(dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    det = _perfect_jsonl(dataset, tmp_path / "det.jsonl")
    out = tmp_path / "pre"

    assert (
        main(
            [
                "prelabel",
                "--manifest",
                str(dataset / "good.jsonl"),
                "--detector",
                str(det),
                "--out",
                str(out),
                "--count",
                "4",
            ]
        )
        == 0
    )
    assert "cvat_yolo11.zip" in capsys.readouterr().out
    assert main(["prelabel", "--check", str(out / "selected.jsonl")]) == 0
    assert "reproduce their known reading" in capsys.readouterr().out
    assert (out / "annotation_check.json").is_file()


def test_cli_without_the_required_arguments_explains_what_is_missing() -> None:
    assert main(["prelabel"]) == 2
