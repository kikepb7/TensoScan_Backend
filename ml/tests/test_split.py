"""T-B003-06: leak-free split by device or by reading (REQ-B003-12)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tensoscan_ml.cli import main
from tensoscan_ml.data.manifest import ManifestEntry, read_manifest, write_manifest
from tensoscan_ml.data.split import SplitOptions, split_entries


def _entries(devices: list[str], readings: list[tuple[int, int, int]], frames: int = 5) -> list[ManifestEntry]:
    entries = []
    for device in devices:
        for sys, dia, pul in readings:
            for take in range(2):
                video = f"{device}__{sys}-{dia}-{pul}__good__{take:02d}"
                for frame in range(frames):
                    entries.append(
                        ManifestEntry(
                            path=f"frames/{video}/{frame:06d}.jpg",
                            sha256="0" * 64,
                            device=device,
                            status="COMPLETE",
                            sys=sys,
                            dia=dia,
                            pul=pul,
                            condition="good",
                            video_id=video,
                            frame_idx=frame,
                        )
                    )
    return entries


READINGS = [(110 + i, 70 + i % 9, 60 + i) for i in range(12)]


def test_with_three_devices_at_least_one_is_only_in_test() -> None:
    entries = _entries(["omron-m3", "beurer-bm27", "medisana-bu510"], READINGS)

    result = split_entries(entries, SplitOptions(seed=1))

    held_out = result.report["held_out_devices"]
    assert len(held_out) >= 1
    for device in held_out:
        assert {e.device for e in result.assignments["test"]} >= {device}
        assert all(e.device != device for split in ("train", "val") for e in result.assignments[split])
    assert result.report["channel"] == "production-eligible"
    assert result.report["leaks"] == []
    assert sum(len(v) for v in result.assignments.values()) == len(entries)


def test_with_one_device_no_reading_is_shared_between_splits_and_the_channel_is_beta() -> None:
    entries = _entries(["omron-m3"], READINGS)

    result = split_entries(entries, SplitOptions(seed=3))

    readings = {split: {e.reading for e in items} for split, items in result.assignments.items()}
    assert not readings["train"] & readings["test"]
    assert not readings["train"] & readings["val"]
    assert not readings["val"] & readings["test"]
    assert all(readings[split] for split in readings)
    assert result.report["channel"] == "beta"
    assert any("optimistic" in w for w in result.report["warnings"])
    assert result.report["leaks"] == []


def test_all_frames_of_a_video_stay_together() -> None:
    result = split_entries(_entries(["omron-m3", "beurer-bm27"], READINGS), SplitOptions(seed=5))

    where: dict[str | None, set[str]] = {}
    for split, items in result.assignments.items():
        for entry in items:
            where.setdefault(entry.video_id, set()).add(split)
    assert all(len(splits) == 1 for splits in where.values())


def test_the_same_seed_gives_the_same_split_and_another_seed_a_different_one() -> None:
    entries = _entries(["omron-m3"], READINGS)

    first = split_entries(entries, SplitOptions(seed=7))
    again = split_entries(entries, SplitOptions(seed=7))
    other = split_entries(entries, SplitOptions(seed=8))

    names = lambda result: {s: sorted(e.path for e in items) for s, items in result.assignments.items()}  # noqa: E731
    assert names(first) == names(again)
    assert names(first) != names(other)


def test_the_ratios_are_respected_approximately() -> None:
    result = split_entries(_entries(["omron-m3"], [(100 + i, 60 + i % 20, 55 + i) for i in range(40)]), SplitOptions())

    total = sum(len(v) for v in result.assignments.values())
    assert len(result.assignments["train"]) / total == pytest.approx(0.70, abs=0.1)
    assert len(result.assignments["test"]) / total == pytest.approx(0.15, abs=0.1)


def test_one_single_reading_cannot_be_split_without_leakage() -> None:
    with pytest.raises(ValueError, match="at least 2 distinct readings"):
        split_entries(_entries(["omron-m3"], [(128, 82, 67)]), SplitOptions())


def test_error_videos_are_grouped_by_video_and_do_not_break_the_split() -> None:
    entries = _entries(["omron-m3"], READINGS)
    entries += [
        ManifestEntry(f"frames/e{v}/{f}.jpg", "0" * 64, "omron-m3", "DEVICE_ERROR", video_id=f"e{v}", frame_idx=f)
        for v in range(4)
        for f in range(3)
    ]

    result = split_entries(entries, SplitOptions(seed=2))

    for video in {f"e{v}" for v in range(4)}:
        homes = {s for s, items in result.assignments.items() if any(e.video_id == video for e in items)}
        assert len(homes) == 1


def test_invalid_ratios_are_rejected() -> None:
    with pytest.raises(ValueError, match="add up to 1"):
        SplitOptions(ratios=(0.5, 0.5, 0.5))


def test_split_cli_writes_the_three_manifests_and_the_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, _entries(["omron-m3"], READINGS))

    actual = main(["split", "--manifest", str(manifest)])

    assert actual == 0
    for name in ("train", "val", "test"):
        assert read_manifest(tmp_path / f"{name}.jsonl")
    assert (tmp_path / "split_report.json").is_file()
    assert "beta" in capsys.readouterr().out
