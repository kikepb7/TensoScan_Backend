from __future__ import annotations

import contextlib
import json
from typing import Any

import pytest

from tensoscan_ml.config import contracts_dir

# Importing torch._dynamo for the first time *after* TensorFlow is loaded in the same process crashes (the triton
# extension clashes with it). The export tests load TensorFlow, the training tests create optimisers, so load it first.
with contextlib.suppress(ImportError):
    import torch._dynamo  # noqa: F401


@pytest.fixture(scope="session")
def golden_cases() -> dict[str, Any]:
    """The shared golden cases, read straight from `specs/contracts/` (never copied)."""
    path = contracts_dir() / "reading-golden-cases.json"
    with path.open(encoding="utf-8") as handle:
        loaded: dict[str, Any] = json.load(handle)
    return loaded
