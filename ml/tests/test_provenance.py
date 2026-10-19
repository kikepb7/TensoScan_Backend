from __future__ import annotations

import json
from pathlib import Path

from tensoscan_ml.provenance import bytes_sha256, combine_hashes, file_sha256, git_commit, write_run_record


def test_run_record_stores_seed_commit_dataset_and_config(tmp_path: Path) -> None:
    path = write_run_record(
        tmp_path / "run",
        "synth",
        config={"a": 1},
        seed=42,
        dataset_version="ds-test",
        dataset_hash="abc",
    )

    actual = json.loads(path.read_text(encoding="utf-8"))

    assert actual["command"] == "synth"
    assert actual["seed"] == 42
    assert actual["dataset_version"] == "ds-test"
    assert actual["dataset_hash"] == "abc"
    assert actual["config"] == {"a": 1}
    assert actual["git_commit"] == git_commit()
    assert "python" in actual["versions"]


def test_file_and_bytes_hashes_agree(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"hello")

    assert file_sha256(path) == bytes_sha256(b"hello")


def test_combined_hash_does_not_depend_on_the_order_of_files() -> None:
    first = combine_hashes([("a", "1"), ("b", "2")])
    second = combine_hashes([("b", "2"), ("a", "1")])

    assert first == second
    assert first != combine_hashes([("a", "1"), ("b", "3")])
