"""`tensoscan-ml synth`: generate synthetic seven-segment display images with exact YOLO labels (T-B003-05)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2

from tensoscan_ml.commands import add_common_arguments
from tensoscan_ml.config import apply_overrides, data_dir
from tensoscan_ml.console import echo
from tensoscan_ml.provenance import write_run_record
from tensoscan_ml.synth.params import load_synth_config

HELP = "Generate synthetic seven-segment display images with exact YOLO labels"


def configure(parser: argparse.ArgumentParser) -> None:
    add_common_arguments(parser)
    parser.add_argument("--out", help="output directory (default: $TENSOSCAN_ML_DATA_DIR/synth/<dataset_version>)")
    parser.add_argument("--seed", type=int, help="random seed (default: the config's)")
    parser.add_argument("--train", type=int, help="number of training images")
    parser.add_argument("--val", type=int, help="number of validation images")
    parser.add_argument("--test", type=int, help="number of test images")
    parser.add_argument("--workers", type=int, default=1, help="processes (the output does not depend on it)")
    parser.add_argument("--gallery", metavar="PNG", help="write a contact sheet of annotated samples and exit")
    parser.add_argument("--gallery-count", type=int, default=50, help="samples in the gallery (default 50)")


def run(args: argparse.Namespace) -> int:
    from tensoscan_ml.synth.gallery import render_gallery
    from tensoscan_ml.synth.writer import write_dataset

    config = load_synth_config(args.config)
    config = apply_overrides(config, args.overrides)
    seed = int(args.seed if args.seed is not None else config["seed"])

    if args.gallery:
        sheet = render_gallery(config, seed, args.gallery_count)
        target = Path(args.gallery)
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 9])
        echo(f"Gallery of {args.gallery_count} samples written to {target} ({target.stat().st_size / 1024:.0f} KiB).")
        return 0

    counts = {
        "train": args.train if args.train is not None else int(config["dataset"]["train"]),
        "val": args.val if args.val is not None else int(config["dataset"]["val"]),
        "test": args.test if args.test is not None else int(config["dataset"]["test"]),
    }
    out_dir = Path(args.out) if args.out else data_dir() / "synth" / str(config["dataset_version"])
    workers = max(1, min(args.workers, os.cpu_count() or 1))
    info = write_dataset(out_dir, config, counts, seed=seed, workers=workers)
    write_run_record(
        out_dir,
        "synth",
        config=config,
        seed=seed,
        dataset_version=str(config["dataset_version"]),
        dataset_hash=info["dataset_hash"],
    )
    echo(f"Wrote {sum(counts.values())} samples to {out_dir} (dataset hash {info['dataset_hash'][:16]}).")
    for split, summary in info["summary"].items():
        echo(f"  {split}: {summary['count']} images, {summary['readable']} readable by the parser")
    return 0
