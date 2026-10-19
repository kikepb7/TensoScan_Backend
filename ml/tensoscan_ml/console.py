"""Terminal output of the CLI (the lint rules forbid bare `print`)."""

from __future__ import annotations

import sys


def echo(message: str = "") -> None:
    sys.stdout.write(message + "\n")


def warn(message: str) -> None:
    sys.stderr.write(f"warning: {message}\n")


def fail(message: str) -> None:
    sys.stderr.write(f"error: {message}\n")
