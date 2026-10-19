"""Licence audit of the installed dependencies (commercial app => permissive licences only, C-ML-01 / design §8).

The audit reads the metadata of every installed distribution and **fails on strong copyleft** (GPL, AGPL). Weak
copyleft (LGPL) and proprietary build-time tooling (the NVIDIA CUDA wheels PyTorch pulls on Linux) are reported as
warnings: none of it ships in the app, but a human should know.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from importlib import metadata

#: Distributions that are reviewed and accepted on purpose, with the reason. Empty on purpose: add sparingly.
#: This project itself is not a third-party dependency.
OWN_PACKAGE = "tensoscan-ml"

ALLOWLIST: Mapping[str, str] = {}

_AGPL = re.compile(r"affero|(?<![A-Za-z])agpl", re.IGNORECASE)
_GPL_TOKEN = re.compile(r"(?<![A-Za-z])gpl", re.IGNORECASE)
_GNU_GPL = re.compile(r"gnu general public licen[cs]e", re.IGNORECASE)
_LESSER = re.compile(r"lesser|library|(?<![A-Za-z])lgpl", re.IGNORECASE)
_LGPL = re.compile(r"lgpl|lesser", re.IGNORECASE)
_PROPRIETARY = re.compile(r"proprietary|nvidia", re.IGNORECASE)
_LICENSE_FIELD_MAX = 300


@dataclass(frozen=True)
class Finding:
    package: str
    version: str
    licenses: tuple[str, ...]
    level: str  # "violation" | "warning"
    reason: str


@dataclass
class AuditReport:
    checked: int = 0
    findings: list[Finding] = field(default_factory=list)

    @property
    def violations(self) -> list[Finding]:
        return [finding for finding in self.findings if finding.level == "violation"]

    @property
    def warnings(self) -> list[Finding]:
        return [finding for finding in self.findings if finding.level == "warning"]


def license_strings(dist: metadata.Distribution) -> tuple[str, ...]:
    """Licence declarations of a distribution: SPDX expression, trove classifiers and a short `License` field."""
    meta = dist.metadata
    found: list[str] = []
    expression = meta.get("License-Expression")
    if expression:
        found.append(expression)
    found.extend(
        classifier.split("::", 1)[1].strip()
        for classifier in meta.get_all("Classifier") or []
        if classifier.startswith("License ::") and "::" in classifier
    )
    free_text = meta.get("License")
    if free_text and free_text.strip().upper() != "UNKNOWN":
        # Some wheels paste the whole licence text here: only the head says which licence it is.
        found.append(" ".join(free_text.split())[:_LICENSE_FIELD_MAX])
    return tuple(dict.fromkeys(found))


def is_strong_copyleft(declaration: str) -> bool:
    """True for GPL/AGPL declarations; LGPL ("Lesser"/"Library") does not count."""
    if _AGPL.search(declaration):
        return True
    if _LESSER.search(declaration):
        return False
    return bool(_GPL_TOKEN.search(declaration) or _GNU_GPL.search(declaration))


def _has_permissive_alternative(declaration: str) -> bool:
    """`MIT OR GPL-2.0` lets the consumer pick the permissive side."""
    if " OR " not in declaration.upper():
        return False
    return any(not is_strong_copyleft(part) for part in re.split(r"\s+OR\s+", declaration, flags=re.IGNORECASE))


def audit_distribution(name: str, version: str, declarations: tuple[str, ...]) -> Finding | None:
    if name.lower() in ALLOWLIST:
        return None
    copyleft = [d for d in declarations if is_strong_copyleft(d)]
    if copyleft and not any(_has_permissive_alternative(d) for d in copyleft):
        # A package that also declares a permissive licence in another field (e.g. MIT classifier next to a GPL
        # mention in free text) is still reported: the reviewer decides and may allow-list it with a reason.
        return Finding(name, version, declarations, "violation", f"strong copyleft licence: {copyleft[0]}")
    if any(_LGPL.search(d) for d in declarations):
        return Finding(name, version, declarations, "warning", "weak copyleft (LGPL): fine if dynamically linked")
    if any(_PROPRIETARY.search(d) for d in declarations):
        return Finding(name, version, declarations, "warning", "proprietary licence: build-time tooling only")
    if not declarations:
        return Finding(name, version, declarations, "warning", "no licence declared in the metadata")
    return None


def audit_declarations(items: Iterable[tuple[str, str, tuple[str, ...]]]) -> AuditReport:
    report = AuditReport()
    for name, version, declarations in items:
        report.checked += 1
        finding = audit_distribution(name, version, declarations)
        if finding is not None:
            report.findings.append(finding)
    return report


def audit_installed() -> AuditReport:
    """Audits every distribution of the current environment."""
    items = [
        (dist.metadata.get("Name") or "unknown", dist.version, license_strings(dist))
        for dist in metadata.distributions()
        if dist.metadata.get("Name") != OWN_PACKAGE
    ]
    return audit_declarations(sorted(items))
