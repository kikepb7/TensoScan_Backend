"""A golden set of synthetic frames whose reading the official parser reproduces from the exact labels."""

from __future__ import annotations

from pathlib import Path

from tensoscan_ml.data.manifest import read_manifest, write_manifest
from tensoscan_ml.synth.params import load_synth_config
from tensoscan_ml.synth.writer import write_dataset

GOLDEN_SEED = 2024
GOLDEN_SIZE = 20


def make_synthetic_golden(directory: Path, count: int = GOLDEN_SIZE, seed: int = GOLDEN_SEED) -> Path:
    """Generates frames without perspective, readable and with parser-friendly fonts; returns the golden manifest."""
    config = load_synth_config(overrides={"scene": {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}})
    write_dataset(directory / "pool", config, {"val": count * 4}, seed=seed)
    entries = [
        e
        for e in read_manifest(directory / "pool" / "val" / "manifest.jsonl")
        if e.status == "COMPLETE" and e.extra["readable"] and e.extra["parser_compatible"]
    ][:count]
    if len(entries) < count:
        raise RuntimeError(f"could only build {len(entries)} of {count} golden frames")
    golden = directory / "pool" / "val" / "golden.jsonl"
    write_manifest(golden, entries)
    return golden
