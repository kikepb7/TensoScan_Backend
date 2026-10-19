"""`tensoscan-ml verify`: check an exported `.tflite` against the model contract."""

from __future__ import annotations

import argparse
from pathlib import Path

from tensoscan_ml.commands import add_common_arguments
from tensoscan_ml.console import echo

HELP = "Verify a .tflite: I/O, builtin ops, classes, size, CPU latency and a golden set; machine-readable report"


def configure(parser: argparse.ArgumentParser) -> None:
    add_common_arguments(parser)
    parser.add_argument("--model", required=True, help="digits-<version>.tflite (the sidecar .json sits next to it)")
    parser.add_argument(
        "--golden",
        help="manifest.jsonl with about 20 frames of known reading, or `synthetic` to generate 20 parser-friendly ones",
    )
    parser.add_argument("--min-golden-accuracy", type=float, default=0.9)
    parser.add_argument("--runs", type=int, default=50, help="timed inferences for the latency")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--enforce-latency", action="store_true", help="fail when the CPU latency is over the budget")
    parser.add_argument("--report", help="where to write the JSON report (default: <model>.verify.json)")


def run(args: argparse.Namespace) -> int:
    import tempfile

    from tensoscan_ml.verify.checks import FAIL, verify_model
    from tensoscan_ml.verify.golden import make_synthetic_golden

    model = Path(args.model)
    scratch = tempfile.TemporaryDirectory()
    golden = None
    if args.golden == "synthetic":
        golden = make_synthetic_golden(Path(scratch.name))
    elif args.golden:
        golden = Path(args.golden)
    report = verify_model(
        model,
        golden=golden,
        runs=args.runs,
        threads=args.threads,
        min_golden_accuracy=args.min_golden_accuracy,
        enforce_latency=args.enforce_latency,
    )
    scratch.cleanup()
    target = Path(args.report) if args.report else model.with_suffix(".verify.json")
    target.write_text(report.to_json(), encoding="utf-8")
    for check in report.checks:
        echo(f"  [{check.status:>4}] {check.name}: {check.detail}")
    failing = sum(c.status == FAIL for c in report.checks)
    echo(f"{'VERIFIED' if report.passed else 'FAILED'}: {failing} failing check(s). Report: {target}")
    return 0 if report.passed else 1
