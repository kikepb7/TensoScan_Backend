"""Access to the synthetic-generator configuration and its random draws."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from tensoscan_ml.config import Config, deep_merge, default_config_path, load_yaml

Rng = np.random.Generator


def load_synth_config(path: str | None = None, overrides: Config | None = None) -> Config:
    """The default config laid over by `path` (if any) and `overrides`."""
    config = load_yaml(default_config_path("synth", "default"))
    if path is not None:
        config = deep_merge(config, load_yaml(path))
    if overrides:
        config = deep_merge(config, overrides)
    validate_synth_config(config)
    return config


def validate_synth_config(config: Config) -> None:
    layouts = config["layouts"]
    shares = {k: v for k, v in layouts.items() if k in TEMPLATE_NAMES}
    if set(shares) != set(TEMPLATE_NAMES):
        raise ValueError(f"layouts must define the templates {sorted(TEMPLATE_NAMES)}")
    if abs(sum(shares.values()) - 1.0) > 1e-6:
        raise ValueError(f"layout shares must add up to 1, got {sum(shares.values()):.4f}")
    values = config["values"]
    if abs(sum(values.values()) - 1.0) > 1e-6:
        raise ValueError("value-range shares must add up to 1")
    image = config["image"]
    if image["format"] not in ("jpg", "png"):
        raise ValueError("image.format must be jpg or png")


TEMPLATE_NAMES: tuple[str, ...] = (
    "stacked_small_pulse",
    "three_rows_equal",
    "sys_dia_row_pulse_below",
    "clock_date",
    "error",
    "measuring",
    "negative",
    "out_of_coverage",
)


def uniform(rng: Rng, bounds: Sequence[float]) -> float:
    low, high = bounds
    return float(rng.uniform(low, high))


def chance(rng: Rng, probability: float) -> bool:
    return bool(rng.random() < probability)


def choose_weighted(rng: Rng, shares: dict[str, Any], names: Sequence[str]) -> str:
    weights = np.array([float(shares[name]) for name in names])
    return str(names[int(rng.choice(len(names), p=weights / weights.sum()))])
