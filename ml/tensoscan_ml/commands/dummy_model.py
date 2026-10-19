"""`tensoscan-ml dummy-model`: random-weights `.tflite` + sidecar that respects the contract exactly."""

from __future__ import annotations

import argparse
from pathlib import Path

from tensoscan_ml.commands import add_common_arguments
from tensoscan_ml.config import ml_root
from tensoscan_ml.console import echo

HELP = "Build dummy .tflite models (random weights, or a fixed reading) that respect the contract, for app development"


def configure(parser: argparse.ArgumentParser) -> None:
    add_common_arguments(parser)
    parser.add_argument("--out", help="output folder (default: ml/fixtures/dummy)")
    parser.add_argument("--kind", choices=("random", "fixed", "both"), default="both")
    parser.add_argument("--seed", type=int, default=0, help="seed of the random weights")
    parser.add_argument("--reading", default="128-82-67", help="reading the `fixed` model reports, as SYS-DIA-PUL")


def run(args: argparse.Namespace) -> int:
    from tensoscan_ml.export.dummy import build_dummy

    parts = args.reading.split("-")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise ValueError("--reading must look like 128-82-67")
    reading = (int(parts[0]), int(parts[1]), int(parts[2]))
    out_dir = Path(args.out) if args.out else ml_root() / "fixtures" / "dummy"
    for kind in ("random", "fixed") if args.kind == "both" else (args.kind,):
        result = build_dummy(out_dir, kind, seed=args.seed, reading=reading)
        echo(f"{kind}: {result.tflite} ({result.tflite.stat().st_size / 1024:.0f} KiB, sha256 {result.sha256[:16]}...)")
    return 0
