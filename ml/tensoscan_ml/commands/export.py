"""`tensoscan-ml export`: checkpoint -> ONNX -> `.tflite` with the contract sidecar and model-card template."""

from __future__ import annotations

import argparse
from pathlib import Path

from tensoscan_ml.commands import add_common_arguments
from tensoscan_ml.config import apply_overrides, default_config_path, load_yaml
from tensoscan_ml.console import echo

HELP = "Export a checkpoint to digits-<version>.tflite (float16 by default, int8 optional) with its sidecar"


def configure(parser: argparse.ArgumentParser) -> None:
    add_common_arguments(parser)
    parser.add_argument("--checkpoint", required=True, help="best.pt of a training run")
    parser.add_argument("--out", required=True, help="output folder")
    parser.add_argument("--version", help="semver of the model, e.g. 1.0.0")
    parser.add_argument("--precision", choices=("float16", "float32", "int8"), help="default: float16")
    parser.add_argument("--int8-images", help="folder with ~300 representative frames (required for int8)")
    parser.add_argument("--dataset-version", help="dataset version recorded in the sidecar")
    parser.add_argument("--keep-work", action="store_true", help="keep the ONNX and SavedModel intermediates")


def run(args: argparse.Namespace) -> int:
    from tensoscan_ml.export.pipeline import ExportOptions, export_checkpoint
    from tensoscan_ml.provenance import write_run_record

    config = apply_overrides(load_yaml(args.config or default_config_path("export", "v1")), args.overrides)
    options = ExportOptions(
        version=args.version or str(config["version"]),
        precision=args.precision or str(config["precision"]),
        opset=int(config["onnx_opset"]),
        int8_images=Path(args.int8_images) if args.int8_images else None,
        int8_count=int(config["int8_samples"]),
        dataset_version=args.dataset_version or str(config["dataset_version"]),
        keep_work=args.keep_work,
    )
    result = export_checkpoint(args.checkpoint, args.out, options)
    write_run_record(args.out, "export", config=config, extra={"checkpoint": args.checkpoint, "sha256": result.sha256})
    echo(f"{result.tflite} ({result.tflite.stat().st_size / 1024:.0f} KiB, sha256 {result.sha256})")
    echo(f"sidecar: {result.sidecar}")
    echo(f"model card (fill in the TODO fields): {result.model_card}")
    return 0
