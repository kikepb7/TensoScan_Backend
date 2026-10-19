"""`tensoscan-ml split`: leak-free train/val/test split of an ingested dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from tensoscan_ml.commands import add_common_arguments
from tensoscan_ml.config import default_config_path, load_yaml
from tensoscan_ml.console import echo, fail, warn

HELP = "Split an ingested dataset into train, val and test without leakage (by device or by reading)"


def configure(parser: argparse.ArgumentParser) -> None:
    add_common_arguments(parser)
    parser.add_argument("--manifest", required=True, help="manifest.jsonl written by `ingest`")
    parser.add_argument("--out", help="where to write train/val/test.jsonl (default: next to the manifest)")
    parser.add_argument("--seed", type=int, help="random seed (default: the config's)")


def run(args: argparse.Namespace) -> int:
    from tensoscan_ml.data.manifest import read_manifest
    from tensoscan_ml.data.split import SplitOptions, split_entries, write_split

    section = load_yaml(default_config_path("ingest", "default"))["split"]
    if args.config:
        section = {**section, **load_yaml(args.config).get("split", {})}
    ratios = section["ratios"]
    options = SplitOptions(
        seed=int(args.seed if args.seed is not None else section["seed"]),
        ratios=(float(ratios[0]), float(ratios[1]), float(ratios[2])),
        holdout_fraction=float(section["holdout_fraction"]),
    )
    manifest = Path(args.manifest)
    result = split_entries(read_manifest(manifest), options)
    written = write_split(result, Path(args.out) if args.out else manifest.parent)
    report = result.report
    echo(f"Split mode: {report['mode']} | channel: {report['channel']} | devices: {', '.join(report['devices'])}")
    for split, info in report["splits"].items():
        echo(f"  {split}: {info['frames']} frames, {info['readings']} readings, {info['videos']} videos")
    for message in report["warnings"]:
        warn(message)
    echo(f"Report: {written['report']}")
    if report["leaks"]:
        for leak in report["leaks"]:
            fail(f"leakage: {leak}")
        return 1
    return 0
