"""`tensoscan-ml audit-licenses`: fail when an installed dependency carries a strong copyleft licence."""

from __future__ import annotations

import argparse

from tensoscan_ml.audit.licenses import audit_installed
from tensoscan_ml.console import echo, fail

HELP = "Fail when an installed dependency carries a strong copyleft licence (GPL, AGPL)"


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--quiet", action="store_true", help="only print violations")


def run(args: argparse.Namespace) -> int:
    report = audit_installed()
    echo(f"Checked {report.checked} distributions.")
    if not args.quiet:
        for finding in report.warnings:
            echo(f"  warning  {finding.package} {finding.version}: {finding.reason} ({'; '.join(finding.licenses)})")
    for finding in report.violations:
        fail(f"{finding.package} {finding.version}: {finding.reason}")
    if report.violations:
        return 1
    echo("OK: no GPL/AGPL dependency.")
    return 0
