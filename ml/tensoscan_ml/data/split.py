"""Leak-free train/val/test split (REQ-B003-12).

With three or more devices at least one device is held out entirely for test. With fewer, the unit of the split is the
**reading** (the SYS/DIA/PUL combination, across every video and photo that shows it), so no reading appears on both
sides; the report then marks the evaluation as `beta`.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tensoscan_ml.data.manifest import ManifestEntry, write_manifest
from tensoscan_ml.data.naming import STATUS_ERROR

SPLITS = ("train", "val", "test")
BETA_MIN_READINGS = 30  # requirements section 1: channel `beta` needs >= 30 distinct readings
PRODUCTION_MIN_DEVICES = 3


@dataclass(frozen=True)
class SplitOptions:
    seed: int = 42
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)
    holdout_fraction: float = 0.20

    def __post_init__(self) -> None:
        if len(self.ratios) != 3 or abs(sum(self.ratios) - 1.0) > 1e-6 or min(self.ratios) < 0:
            raise ValueError("ratios must be three non-negative numbers that add up to 1")


@dataclass
class SplitResult:
    assignments: dict[str, list[ManifestEntry]]
    report: dict[str, Any] = field(default_factory=dict)


def _device_of(entry: ManifestEntry) -> str:
    return entry.device


def _reading_group(entry: ManifestEntry) -> str:
    """Frames of one reading, and of one video when there is no reading, travel together."""
    if entry.status == STATUS_ERROR or entry.sys is None:
        return f"{entry.status or 'unlabelled'}|{entry.device}|{entry.video_id or entry.path}"
    return f"{entry.sys}-{entry.dia}-{entry.pul}"


def _assign_groups(
    groups: dict[str, list[ManifestEntry]], ratios: tuple[float, float, float], rng: random.Random
) -> dict[str, str]:
    """Greedy assignment of whole groups to splits, largest deficit first (reproducible with the seed)."""
    names = sorted(groups)
    rng.shuffle(names)
    names.sort(key=lambda name: -len(groups[name]))  # big groups first: they are the hardest to balance
    total = sum(len(groups[name]) for name in names)
    targets = {split: ratio * total for split, ratio in zip(SPLITS, ratios, strict=True)}
    loaded = dict.fromkeys(SPLITS, 0)
    chosen: dict[str, str] = {}
    active = [split for split, ratio in zip(SPLITS, ratios, strict=True) if ratio > 0]
    for index, name in enumerate(names):
        # make sure every active split gets at least one group while groups last
        empty = [split for split in active if loaded[split] == 0]
        if empty and len(names) - index <= len(empty):
            split = empty[0]
        else:
            split = max(active, key=lambda candidate: targets[candidate] - loaded[candidate])
        chosen[name] = split
        loaded[split] += len(groups[name])
    return chosen


def split_entries(entries: list[ManifestEntry], options: SplitOptions) -> SplitResult:
    """Assigns every entry to `train`, `val` or `test` without leakage."""
    if not entries:
        raise ValueError("the manifest is empty")
    rng = random.Random(options.seed)  # noqa: S311  (reproducible shuffling, not cryptography)
    devices = sorted({_device_of(e) for e in entries})
    warnings: list[str] = []
    assignments: dict[str, list[ManifestEntry]] = {split: [] for split in SPLITS}
    held_out: list[str] = []

    if len(devices) >= PRODUCTION_MIN_DEVICES:
        count = max(1, math.ceil(len(devices) * options.holdout_fraction))
        shuffled = devices[:]
        rng.shuffle(shuffled)
        held_out = sorted(shuffled[:count])
        for entry in entries:
            if _device_of(entry) in held_out:
                assignments["test"].append(entry)
        rest = [e for e in entries if _device_of(e) not in held_out]
        groups = _group(rest, lambda e: f"{_device_of(e)}|{_reading_group(e)}")
        train_val = (options.ratios[0], options.ratios[1], 0.0)
        scale = sum(train_val)
        chosen = _assign_groups(groups, (train_val[0] / scale, train_val[1] / scale, 0.0), rng)
        for name, split in chosen.items():
            assignments[split].extend(groups[name])
        mode = "device-holdout"
    else:
        groups = _group(entries, _reading_group)
        if len(groups) < 2:
            raise ValueError(
                "need at least 2 distinct readings (or videos with an error) to split without leakage; "
                f"found {len(groups)}"
            )
        ratios = options.ratios
        if len(groups) == 2:
            warnings.append("only 2 distinct readings: no validation split")
            ratios = (ratios[0] / (ratios[0] + ratios[2]), 0.0, ratios[2] / (ratios[0] + ratios[2]))
        chosen = _assign_groups(groups, ratios, rng)
        for name, split in chosen.items():
            assignments[split].extend(groups[name])
        mode = "reading-groups"
        warnings.append(
            f"only {len(devices)} device(s): the test metrics are optimistic (unseen readings of known devices), "
            "channel `beta`"
        )

    report = _report(assignments, devices, held_out, mode, warnings)
    return SplitResult(assignments, report)


def _group(entries: list[ManifestEntry], key: Any) -> dict[str, list[ManifestEntry]]:
    groups: dict[str, list[ManifestEntry]] = defaultdict(list)
    for entry in entries:
        groups[key(entry)].append(entry)
    return dict(groups)


def _readings(entries: list[ManifestEntry]) -> set[tuple[int | None, int | None, int | None]]:
    return {e.reading for e in entries if e.status != STATUS_ERROR and e.sys is not None}


def _report(
    assignments: dict[str, list[ManifestEntry]], devices: list[str], held_out: list[str], mode: str, warnings: list[str]
) -> dict[str, Any]:
    leaks: list[str] = []
    for index, first in enumerate(SPLITS):
        for second in SPLITS[index + 1 :]:
            shared = _readings(assignments[first]) & _readings(assignments[second])
            if mode == "reading-groups" and shared:
                leaks.append(f"{len(shared)} reading(s) in both {first} and {second}")
    train_val_devices = {e.device for s in ("train", "val") for e in assignments[s]}
    if held_out and set(held_out) & train_val_devices:
        leaks.append("a held-out device also appears in train or val")
    all_readings = _readings([e for items in assignments.values() for e in items])
    channel = "production-eligible" if len(devices) >= PRODUCTION_MIN_DEVICES and held_out else "beta"
    if len(all_readings) < BETA_MIN_READINGS:
        warnings.append(
            f"{len(all_readings)} distinct readings: the beta channel asks for at least {BETA_MIN_READINGS}"
        )
    if len(devices) < PRODUCTION_MIN_DEVICES:
        warnings.append(f"{len(devices)} device(s): `production` needs at least {PRODUCTION_MIN_DEVICES}")
    return {
        "mode": mode,
        "channel": channel,
        "devices": devices,
        "held_out_devices": held_out,
        "distinct_readings": len(all_readings),
        "splits": {
            split: {
                "frames": len(items),
                "devices": sorted({e.device for e in items}),
                "readings": len(_readings(items)),
                "videos": len({e.video_id for e in items}),
            }
            for split, items in assignments.items()
        },
        "leaks": leaks,
        "warnings": warnings,
    }


def write_split(result: SplitResult, out_dir: str | Path) -> dict[str, Path]:
    """Writes `train.jsonl`, `val.jsonl`, `test.jsonl` and `split_report.json` (paths stay relative to `out_dir`)."""
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for split, items in result.assignments.items():
        if not items:
            continue
        written[split] = target / f"{split}.jsonl"
        write_manifest(written[split], items)
    report_path = target / "split_report.json"
    report_path.write_text(json.dumps(result.report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    written["report"] = report_path
    return written
