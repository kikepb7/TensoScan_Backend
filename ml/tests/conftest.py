from __future__ import annotations

import json
from typing import Any

import pytest

from tensoscan_ml.config import contracts_dir


@pytest.fixture(scope="session")
def golden_cases() -> dict[str, Any]:
    """The shared golden cases, read straight from `specs/contracts/` (never copied)."""
    path = contracts_dir() / "reading-golden-cases.json"
    with path.open(encoding="utf-8") as handle:
        loaded: dict[str, Any] = json.load(handle)
    return loaded
