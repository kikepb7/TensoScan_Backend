"""`tensoscan-ml` subcommand `train`."""

from __future__ import annotations

import argparse

from tensoscan_ml.commands import add_common_arguments, not_implemented

HELP = "Train the digit detector in two stages (T-B003-09)"


def configure(parser: argparse.ArgumentParser) -> None:
    add_common_arguments(parser)


def run(args: argparse.Namespace) -> int:
    del args
    return not_implemented(HELP)
