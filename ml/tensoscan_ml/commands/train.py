"""`tensoscan-ml train`: two-stage training (synthetic pre-training, then real fine-tuning)."""

from __future__ import annotations

import argparse
from pathlib import Path

from tensoscan_ml.commands import add_common_arguments
from tensoscan_ml.config import default_config_path, load_config
from tensoscan_ml.console import echo

HELP = "Train the digit detector in two stages (synthetic pre-training, then fine-tuning with real frames)"


def configure(parser: argparse.ArgumentParser) -> None:
    add_common_arguments(parser)
    parser.add_argument("--out", help="runs folder (default: config out_dir)")
    parser.add_argument("--stage", action="append", dest="stages", help="run only this stage (repeatable)")
    parser.add_argument(
        "--require-real", action="store_true", help="fail instead of skipping stages that need real data"
    )
    parser.add_argument("--init-from", help="start from the weights of a checkpoint")
    parser.add_argument("--max-steps", type=int, help="stop after this many optimiser steps (debugging)")


def run(args: argparse.Namespace) -> int:
    from tensoscan_ml.model.train import run_training

    config = load_config(args.config or default_config_path("train", "yolox-nano-v1"), args.overrides)
    summary = run_training(
        config,
        out_root=args.out,
        require_real=args.require_real,
        only_stages=args.stages,
        init_from=args.init_from,
        max_steps=args.max_steps,
    )
    echo(f"Run folder: {summary.run_dir}")
    echo(
        f"Stages run: {', '.join(summary.stages_run) or 'none'}; skipped: {', '.join(summary.stages_skipped) or 'none'}"
    )
    for stage, path in summary.checkpoints.items():
        echo(f"  best checkpoint of {stage}: {Path(path)}")
    return 0
