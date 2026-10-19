from __future__ import annotations

import pytest

from tensoscan_ml.audit.licenses import audit_declarations, audit_installed, is_strong_copyleft
from tensoscan_ml.cli import main


@pytest.mark.parametrize(
    "declaration",
    [
        "GPL-3.0-only",
        "GPL-2.0-or-later",
        "AGPL-3.0",
        "AGPL-3.0-or-later",
        "GNU Affero General Public License v3",
        "OSI Approved :: GNU General Public License v3 (GPLv3)",
        "GNU General Public License",
    ],
)
def test_strong_copyleft_declarations_are_detected(declaration: str) -> None:
    assert is_strong_copyleft(declaration)


@pytest.mark.parametrize(
    "declaration",
    [
        "MIT",
        "Apache-2.0",
        "BSD-3-Clause",
        "LGPL-3.0-or-later",
        "GNU Lesser General Public License v3 (LGPLv3)",
        "GNU Library or Lesser General Public License (LGPL)",
        "HPND",
        "Mozilla Public License 2.0 (MPL 2.0)",
    ],
)
def test_permissive_and_weak_copyleft_declarations_pass(declaration: str) -> None:
    assert not is_strong_copyleft(declaration)


def test_audit_fails_on_gpl_and_agpl_dependencies() -> None:
    items = [
        ("fine", "1.0", ("MIT",)),
        ("ultralytics", "8.0", ("AGPL-3.0",)),
        ("gplthing", "2.0", ("GPL-3.0-only",)),
    ]

    actual = audit_declarations(items)

    assert [finding.package for finding in actual.violations] == ["ultralytics", "gplthing"]
    assert actual.checked == 3


def test_dual_licence_with_a_permissive_side_is_accepted() -> None:
    actual = audit_declarations([("dual", "1", ("MIT OR GPL-2.0-only",))])

    assert actual.violations == []


def test_lgpl_and_proprietary_build_tools_are_warnings_not_failures() -> None:
    items = [
        ("lib", "1", ("LGPL-3.0-or-later",)),
        ("nvidia-cublas", "13", ("NVIDIA Proprietary Software",)),
        ("mystery", "1", ()),
    ]

    actual = audit_declarations(items)

    assert actual.violations == []
    assert {finding.package for finding in actual.warnings} == {"lib", "nvidia-cublas", "mystery"}


def test_installed_environment_has_no_strong_copyleft_dependency() -> None:
    report = audit_installed()

    assert [f"{finding.package}: {finding.reason}" for finding in report.violations] == []


def test_audit_command_succeeds_on_a_clean_environment(capsys: pytest.CaptureFixture[str]) -> None:
    actual = main(["audit-licenses", "--quiet"])

    assert actual == 0
    assert "no GPL/AGPL" in capsys.readouterr().out
