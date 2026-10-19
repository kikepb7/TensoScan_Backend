"""`tensoscan-ml prelabel`: model-assisted pre-annotation of real frames for CVAT, and a check of the result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tensoscan_ml.commands import add_common_arguments
from tensoscan_ml.console import echo

HELP = "Pre-annotate ~300 real frames with a model (YOLO + CVAT zip), or --check the corrected annotations"


def configure(parser: argparse.ArgumentParser) -> None:
    add_common_arguments(parser)
    parser.add_argument("--manifest", help="manifest.jsonl of the ingested frames")
    parser.add_argument("--detector", help="checkpoint | tflite | path to a detections.jsonl")
    parser.add_argument("--model", help="the .pt checkpoint or the .tflite file")
    parser.add_argument("--out", help="output folder (images/, labels/, cvat_yolo11.zip, selected.jsonl...)")
    parser.add_argument("--count", type=int, default=300, help="frames to select (default 300)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-threshold", type=float, default=0.25, help="keep detections at or above this score")
    parser.add_argument(
        "--check", metavar="SELECTED_JSONL", help="validate corrected labels against the known readings"
    )


def run(args: argparse.Namespace) -> int:
    from tensoscan_ml.data.prelabel import check_annotations, prelabel

    if args.check:
        result = check_annotations(args.check)
        echo(f"{result['reproduce_reading']}/{result['checked']} annotated frames reproduce their known reading")
        for problem in result["problems"][:25]:
            echo(
                f"  {problem['image']}: {problem['problem']} {problem.get('expected', '')} {problem.get('parsed', '')}"
            )
        check_path = Path(args.check).with_name("annotation_check.json")
        check_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        echo(f"Report: {check_path}")
        return 0 if not result["problems"] else 1

    if not (args.manifest and args.detector and args.out):
        raise ValueError("prelabel needs --manifest, --detector and --out (or use --check)")
    from tensoscan_ml.detector.factory import detector_from_args

    source = detector_from_args(args.detector, args.model)
    report = prelabel(
        args.manifest, source, args.out, count=args.count, seed=args.seed, score_threshold=args.score_threshold
    )
    echo(f"{report.selected} frames pre-annotated ({report.detections} boxes) in {args.out}")
    echo(f"  frames without detections: {len(report.empty_images)} | with more than 20: {len(report.crowded_images)}")
    echo(
        f"Upload {Path(args.out) / 'images'} to CVAT and import {Path(args.out) / 'cvat_yolo11.zip'} (format YOLO 1.1)."
    )
    return 0
