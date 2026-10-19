"""`tensoscan-ml` subcommand `ingest`."""

from __future__ import annotations

import argparse

from tensoscan_ml.commands import add_common_arguments, not_implemented

HELP = "Turn videos and photos with labels into a cropped, de-duplicated dataset with a manifest (T-B003-06)"


def configure(parser: argparse.ArgumentParser) -> None:
    add_common_arguments(parser)


def run(args: argparse.Namespace) -> int:
    del args
    return not_implemented(HELP)
