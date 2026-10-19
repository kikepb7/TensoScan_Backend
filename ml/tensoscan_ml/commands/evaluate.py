"""`tensoscan-ml` subcommand `evaluate`."""

from __future__ import annotations

import argparse

from tensoscan_ml.commands import add_common_arguments, not_implemented

HELP = "Evaluate any detector against a manifest with the official parser (T-B003-07)"


def configure(parser: argparse.ArgumentParser) -> None:
    add_common_arguments(parser)


def run(args: argparse.Namespace) -> int:
    del args
    return not_implemented(HELP)
