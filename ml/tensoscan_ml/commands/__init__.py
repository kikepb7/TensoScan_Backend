"""One module per `tensoscan-ml` subcommand."""

from __future__ import annotations

import argparse

from tensoscan_ml.console import fail


def not_implemented(task: str) -> int:
    """Exit code of a subcommand whose task is not part of the current delivery."""
    fail(f"this command is not implemented yet ({task})")
    return 2


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """`--config` and `--set section.key=value` shared by the commands driven by a YAML file."""
    parser.add_argument("--config", help="YAML file with the settings of this run")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set train.epochs=3 (repeatable)",
    )
