"""T-B003-06: ingestion of synthetic videos and photos built inside the tests (CA-B003-04, CA-B003-08)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tensoscan_ml.cli import main
from tensoscan_ml.data.ingest import IngestOptions, center_roi, ingest
from tensoscan_ml.data.manifest import read_manifest
from tensoscan_ml.data.phash import hamming, phash
from tests.media import blurry, scene, write_image, write_video


def _video_with_static_moving_and_blurry_parts() -> list[np.ndarray]:
    """3 s at 10 fps: 1 s of an unchanging scene, 1 s of changing scenes, 1 s of blurred frames."""
    portrait = (540, 960)  # like a phone held upright: the ROI is then 4:5
    still = [scene(1, portrait)] * 10
    moving = [scene(10 + i, portrait) for i in range(10)]
    soft = [blurry(scene(30 + i, portrait)) for i in range(10)]
    return still + moving + soft


def test_a_video_gives_cropped_frames_without_blurry_or_duplicates_and_a_manifest(tmp_path: Path) -> None:
    """CA-B003-04."""
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    write_video(src / "omron-m3__128-82-67__dim__01.mp4", _video_with_static_moving_and_blurry_parts())

    report = ingest(src, out, IngestOptions(fps=3.0))

    entries = read_manifest(out / "manifest.jsonl")
    assert entries, "frames were expected"
    assert {(e.sys, e.dia, e.pul, e.device, e.condition, e.status) for e in entries} == {
        (128, 82, 67, "omron-m3", "dim", "COMPLETE")
    }
    assert {e.video_id for e in entries} == {"omron-m3__128-82-67__dim__01"}
    stats = report.sources["omron-m3__128-82-67__dim__01.mp4"]
    assert stats.blurry >= 2  # the blurred second
    assert stats.duplicates >= 1  # the static second
    assert stats.kept == len(entries)
    assert 3 <= len(entries) <= 5
    for entry in entries:
        assert len(entry.sha256) == 64
        crop = cv2.imread(str(out / entry.path))
        assert crop is not None
        assert max(crop.shape[:2]) <= 640
        assert abs(crop.shape[0] / crop.shape[1] - 1.25) < 0.01  # 4:5 portrait ROI
    json_report = json.loads((out / "ingest_report.json").read_text(encoding="utf-8"))
    assert json_report["summary"]["frames_blurry"] == stats.blurry


def test_manifest_sha256_matches_the_files_and_paths_are_relative(tmp_path: Path) -> None:
    from tensoscan_ml.provenance import file_sha256

    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    write_video(src / "omron-m3__128-82-67__good__01.mp4", [scene(i) for i in range(12)])
    ingest(src, out, IngestOptions(fps=5.0))

    for entry in read_manifest(out / "manifest.jsonl"):
        assert not Path(entry.path).is_absolute()
        assert file_sha256(out / entry.path) == entry.sha256


def test_photos_with_labels_csv_are_processed_as_they_are(tmp_path: Path) -> None:
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    for index, name in enumerate(["IMG_001.jpg", "IMG_002.jpg", "IMG_003.jpg"]):
        write_image(src / name, scene(index))
    (src / "labels.csv").write_text(
        "file,device,sys,dia,pul,condition\n"
        "IMG_001.jpg,omron-m3,128,82,67,good\n"
        "IMG_002.jpg,omron-m3,141,94,78,glare\n"
        "IMG_003.jpg,omron-m3,error,,,dim\n",
        encoding="utf-8",
    )

    report = ingest(src, out, IngestOptions(roi_mode="none"))

    entries = {e.video_id: e for e in read_manifest(out / "manifest.jsonl")}
    assert set(entries) == {"IMG_001", "IMG_002", "IMG_003"}
    assert entries["IMG_001"].reading == (128, 82, 67)
    assert entries["IMG_003"].status == "DEVICE_ERROR"
    assert report.rejections == []
    first = cv2.imread(str(out / entries["IMG_001"].path))
    assert first.shape[:2] == (480, 640)  # whole image, scaled so that the long side is 640


def test_center_roi_matches_the_apps_guide_rectangle() -> None:
    x, y, w, h = center_roi(1080, 1920)

    assert w == round(0.8 * 1080)
    assert h == round(1.25 * w)
    assert x == round((1080 - w) / 2)
    assert abs((y + h / 2) - 0.45 * 1920) <= 1


def test_manual_roi_mode_crops_the_rectangle_of_labels_csv(tmp_path: Path) -> None:
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    write_image(src / "IMG_001.jpg", scene(0))
    (src / "labels.csv").write_text(
        "file,device,sys,dia,pul,condition,roi_x,roi_y,roi_w,roi_h\nIMG_001.jpg,omron-m3,128,82,67,good,100,50,400,500\n",
        encoding="utf-8",
    )

    ingest(src, out, IngestOptions(roi_mode="manual"))

    entry = read_manifest(out / "manifest.jsonl")[0]
    crop = cv2.imread(str(out / entry.path))
    assert crop.shape[:2] == (500, 400)
    assert entry.extra["roi"] == [100, 50, 400, 500]


def test_nothing_is_dropped_silently_bad_names_missing_files_and_missing_roi_are_reported(tmp_path: Path) -> None:
    """CA-B003-08: every problem ends in the report."""
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    write_image(src / "IMG_001.jpg", scene(1))
    write_image(src / "random-name.jpg", scene(2))
    write_image(src / "IMG_004.jpg", scene(4))
    (src / "labels.csv").write_text(
        "file,device,sys,dia,pul,condition\n"
        "IMG_001.jpg,omron-m3,128,82,67,good\n"
        "IMG_404.jpg,omron-m3,128,82,67,good\n"
        "IMG_004.jpg,omron-m3,128,82,67,night\n",
        encoding="utf-8",
    )

    report = ingest(src, out, IngestOptions(roi_mode="manual"))

    codes = {(r.source, r.code) for r in report.rejections}
    assert ("random-name.jpg", "BAD_NAME") in codes
    assert ("IMG_404.jpg", "FILE_NOT_FOUND") in codes
    assert ("IMG_004.jpg", "BAD_CONDITION") in codes
    assert ("IMG_001.jpg", "MISSING_ROI") in codes
    assert read_manifest(out / "manifest.jsonl") == []
    saved = json.loads((out / "ingest_report.json").read_text(encoding="utf-8"))
    assert len(saved["rejections"]) == len(report.rejections) == 4


def test_preview_writes_a_contact_sheet_and_no_frames(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CA-B003-08: `ingest --preview` gives the contact sheet and reports every rejection."""
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    for index in range(3):
        write_image(src / f"IMG_00{index}.jpg", scene(index))
    write_image(src / "oops.jpg", scene(9))
    (src / "labels.csv").write_text(
        "file,device,sys,dia,pul,condition\n"
        + "".join(f"IMG_00{i}.jpg,omron-m3,{120 + i},8{i},7{i},good\n" for i in range(3)),
        encoding="utf-8",
    )

    actual = main(["ingest", "--src", str(src), "--out", str(out), "--preview"])

    assert actual == 0
    sheet = cv2.imread(str(out / "preview.jpg"))
    assert sheet is not None and sheet.shape[1] > 400
    assert not (out / "frames").exists()
    report = json.loads((out / "ingest_report.json").read_text(encoding="utf-8"))
    assert [r["source"] for r in report["rejections"]] == ["oops.jpg"]
    assert "BAD_NAME" in capsys.readouterr().out


def test_blurry_photos_are_kept_by_default_but_flagged(tmp_path: Path) -> None:
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    write_image(src / "omron-m3__128-82-67__good__01.jpg", blurry(scene(3)))

    kept = ingest(src, out, IngestOptions(roi_mode="none"))
    dropped = ingest(src, tmp_path / "out2", IngestOptions(roi_mode="none", drop_blurry_photos=True))

    assert len(kept.entries) == 1 and any("blurry photo kept" in w for w in kept.warnings)
    assert dropped.entries == [] and {r.code for r in dropped.rejections} == {"BLURRY", "NO_FRAMES_KEPT"}


def test_perceptual_hash_tells_similar_from_different_images() -> None:
    first, again, other = scene(1), scene(1), scene(7)
    noisy = np.clip(first.astype(np.int16) + np.random.default_rng(0).integers(-6, 7, first.shape), 0, 255).astype(
        np.uint8
    )

    assert hamming(phash(first), phash(again)) == 0
    assert hamming(phash(first), phash(noisy)) <= 4
    assert hamming(phash(first), phash(other)) > 4


def test_ingest_cli_commits_the_manifest_only_when_asked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tensoscan_ml.commands.ingest as command

    committed = tmp_path / "manifests"
    monkeypatch.setattr(command, "manifests_dir", lambda: committed)
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    write_video(src / "omron-m3__128-82-67__good__01.mp4", [scene(i) for i in range(12)])

    assert main(["ingest", "--src", str(src), "--out", str(out), "--dataset-version", "ds-test"]) == 0
    assert not committed.exists()
    assert (
        main(["ingest", "--src", str(src), "--out", str(out), "--dataset-version", "ds-test", "--commit-manifest"]) == 0
    )
    assert (committed / "ds-test.jsonl").is_file()
    assert (out / "run.json").is_file()


def test_ingest_cli_fails_when_nothing_is_kept(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    write_image(src / "garbage.jpg", scene(1))

    assert main(["ingest", "--src", str(src), "--out", str(tmp_path / "out")]) == 1
