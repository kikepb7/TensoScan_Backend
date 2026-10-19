"""`tensoscan-ml` subcommand `split`."""

from __future__ import annotations

import argparse

from tensoscan_ml.commands import add_common_arguments, not_implemented

HELP = "Split an ingested dataset into train, val and test without leakage (T-B003-06)"


def configure(parser: argparse.ArgumentParser) -> None:
    add_common_arguments(parser)


def run(args: argparse.Namespace) -> int:
    del args
    return not_implemented(HELP)
