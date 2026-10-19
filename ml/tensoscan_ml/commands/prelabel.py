"""`tensoscan-ml` subcommand `prelabel`."""

from __future__ import annotations

import argparse

from tensoscan_ml.commands import add_common_arguments, not_implemented

HELP = "Pre-annotate real frames with a model for CVAT (T-B003-12)"


def configure(parser: argparse.ArgumentParser) -> None:
    add_common_arguments(parser)


def run(args: argparse.Namespace) -> int:
    del args
    return not_implemented(HELP)
