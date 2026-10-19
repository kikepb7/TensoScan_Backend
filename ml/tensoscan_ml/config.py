"""Locations and YAML configuration shared by every command (REQ-B003-02, REQ-B003-11)."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

DATA_DIR_ENV = "TENSOSCAN_ML_DATA_DIR"
CONTRACTS_DIR_ENV = "TENSOSCAN_CONTRACTS_DIR"

Config = dict[str, Any]


def ml_root() -> Path:
    """The `ml/` directory (where `configs/` and `datasets/` live)."""
    return Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    return ml_root().parent


def contracts_dir() -> Path:
    """Shared contracts, read straight from `specs/contracts/` and never copied (spec T-B003-02)."""
    override = os.environ.get(CONTRACTS_DIR_ENV)
    return Path(override) if override else repo_root() / "specs" / "contracts"


def data_dir() -> Path:
    """Where datasets, runs and exports live. It is outside git by definition (C-ML-03)."""
    override = os.environ.get(DATA_DIR_ENV)
    return Path(override).expanduser() if override else ml_root() / "data"


def manifests_dir() -> Path:
    """Committed manifests (hashes only, no images)."""
    return ml_root() / "datasets" / "manifests"


def load_yaml(path: str | Path) -> Config:
    with Path(path).open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: the top level of a config must be a mapping")
    return loaded


def deep_merge(base: Config, override: Config) -> Config:
    """A new mapping with `override` laid over `base`, merging nested mappings."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def apply_overrides(config: Config, assignments: list[str]) -> Config:
    """Applies `a.b.c=value` assignments (values parsed as YAML scalars) to a copy of `config`."""
    result = copy.deepcopy(config)
    for assignment in assignments:
        key, separator, raw = assignment.partition("=")
        if not separator or not key:
            raise ValueError(f"Override '{assignment}' must look like section.key=value")
        node = result
        parts = key.split(".")
        for part in parts[:-1]:
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError(f"Override '{assignment}': '{part}' is not a section")
            node = child
        node[parts[-1]] = yaml.safe_load(raw)
    return result


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    return apply_overrides(load_yaml(path), overrides or [])


def default_config_path(section: str, name: str) -> Path:
    """`ml/configs/<section>/<name>.yaml`."""
    return ml_root() / "configs" / section / f"{name}.yaml"
