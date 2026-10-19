"""`tensoscan-ml` command line (REQ-B003-03).

Every subcommand lives in `tensoscan_ml/commands/<name>.py` and exposes `HELP`, `configure(parser)` and
`run(args) -> int`. Command modules import their heavy dependencies (PyTorch, TensorFlow...) inside `run`, so
`tensoscan-ml --help` and the fast tests only need the light core.
"""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Sequence

from tensoscan_ml import __version__
from tensoscan_ml.console import fail

#: Subcommand name -> module under `tensoscan_ml.commands`.
COMMANDS: dict[str, str] = {
    "synth": "synth",
    "ingest": "ingest",
    "split": "split",
    "prelabel": "prelabel",
    "train": "train",
    "eval": "evaluate",
    "export": "export",
    "verify": "verify",
    "dummy-model": "dummy_model",
    "release": "release",
    "audit-licenses": "audit_licenses",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tensoscan-ml",
        description="TensoScan digit-detector tooling (spec B-003).",
    )
    parser.add_argument("--version", action="version", version=f"tensoscan-ml {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>", required=True)
    for name, module_name in COMMANDS.items():
        module = importlib.import_module(f"tensoscan_ml.commands.{module_name}")
        sub = subparsers.add_parser(name, help=module.HELP, description=module.HELP)
        module.configure(sub)
        sub.set_defaults(_run=module.run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args._run(args))
    except (ValueError, FileNotFoundError, NotADirectoryError, ImportError) as error:
        fail(str(error))
        return 2
