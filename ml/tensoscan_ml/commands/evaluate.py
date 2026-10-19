"""`tensoscan-ml eval`: evaluate any detector against a manifest with the official parser."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from tensoscan_ml.commands import add_common_arguments
from tensoscan_ml.console import echo, fail
from tensoscan_ml.provenance import write_run_record

if TYPE_CHECKING:
    from tensoscan_ml.detector.runtime import Detector, JsonlDetections

HELP = "Evaluate a checkpoint, a .tflite or a detections.jsonl against a manifest (official parser, mAP, reports)"


def configure(parser: argparse.ArgumentParser) -> None:
    add_common_arguments(parser)
    parser.add_argument(
        "--detector",
        required=True,
        help="checkpoint | tflite | path to a detections.jsonl (with checkpoint/tflite pass --model)",
    )
    parser.add_argument("--model", help="the .pt checkpoint or the .tflite file")
    parser.add_argument("--manifest", required=True, help="manifest.jsonl (paths relative to its folder)")
    parser.add_argument("--out", help="folder for report.md, report.json, detections.jsonl and failures/")
    parser.add_argument("--limit", type=int, help="evaluate only the first N entries")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads of the .tflite interpreter")


def run(args: argparse.Namespace) -> int:
    from tensoscan_ml.detector.runtime import JsonlDetections, tflite_detector
    from tensoscan_ml.eval.runner import evaluate_manifest

    manifest = Path(args.manifest)
    source: Detector | JsonlDetections
    if args.detector.endswith(".jsonl"):
        source = JsonlDetections(args.detector)
    elif args.detector == "tflite":
        if not args.model:
            raise ValueError("--detector tflite needs --model path/to/digits.tflite")
        source = tflite_detector(args.model, args.threads)
    elif args.detector == "checkpoint":
        if not args.model:
            raise ValueError("--detector checkpoint needs --model path/to/best.pt")
        from tensoscan_ml.model.runtime import checkpoint_detector

        source = checkpoint_detector(args.model)
    else:
        raise ValueError("--detector must be checkpoint, tflite or a path to a .jsonl file")

    out_dir = Path(args.out) if args.out else manifest.parent / "eval" / source.name.replace(":", "_")
    report = evaluate_manifest(manifest, source, out_dir, limit=args.limit)
    write_run_record(
        out_dir, "eval", config={"detector": args.detector, "model": args.model, "manifest": str(manifest)}
    )
    reading = report["reading"]
    frame, video = reading["frame"], reading["video"]
    echo(f"{report['detector']} on {report['images']} images ({report['assessment']['channel']})")
    echo(f"  reading per frame: {frame['correct']}/{frame['frames']} | per video: {video['correct']}/{video['videos']}")
    if report["detection"]:
        echo(f"  mAP@0.5 {report['detection']['map50']:.3f}")
    failed = [name for name, check in report["assessment"]["checks"].items() if not check["pass"]]
    echo(f"Reports in {out_dir}")
    if failed:
        fail("below the acceptance thresholds: " + ", ".join(failed))
    return 0
