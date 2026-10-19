from __future__ import annotations

from pathlib import Path

import pytest

from tensoscan_ml.config import (
    DATA_DIR_ENV,
    apply_overrides,
    contracts_dir,
    data_dir,
    deep_merge,
    load_config,
    manifests_dir,
)


def test_deep_merge_overlays_nested_mappings_without_mutating_inputs() -> None:
    base = {"a": {"x": 1, "y": 2}, "b": 1}
    override = {"a": {"y": 3}, "c": 4}

    actual = deep_merge(base, override)

    assert actual == {"a": {"x": 1, "y": 3}, "b": 1, "c": 4}
    assert base == {"a": {"x": 1, "y": 2}, "b": 1}


def test_overrides_parse_yaml_scalars_and_create_sections() -> None:
    config = {"train": {"epochs": 10}}

    actual = apply_overrides(config, ["train.epochs=3", "train.amp=true", "data.name=abc"])

    assert actual == {"train": {"epochs": 3, "amp": True}, "data": {"name": "abc"}}


def test_override_without_equals_sign_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"section\.key=value"):
        apply_overrides({}, ["epochs"])


def test_load_config_reads_yaml_and_applies_overrides(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("seed: 1\nsynth:\n  count: 5\n", encoding="utf-8")

    actual = load_config(path, ["synth.count=7"])

    assert actual == {"seed": 1, "synth": {"count": 7}}


def test_data_dir_follows_the_environment_variable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))

    assert data_dir() == tmp_path


def test_contracts_and_manifests_dirs_exist_in_the_repository() -> None:
    assert (contracts_dir() / "reading-golden-cases.json").is_file()
    assert manifests_dir().parent.name == "datasets"
