"""Reproducibility records: seed, commit, dataset version and hash of every run (REQ-B003-02, C-ML-02)."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import shutil
import subprocess
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from tensoscan_ml import __version__
from tensoscan_ml.config import Config, repo_root

RUN_RECORD_NAME = "run.json"
_TRACKED_PACKAGES = ("numpy", "opencv-python-headless", "torch", "onnx", "onnx2tf", "tensorflow", "ai-edge-litert")


def git_commit() -> str:
    """HEAD of the repository, with a `-dirty` suffix when the working tree has uncommitted changes."""
    git = shutil.which("git")
    if git is None:
        return "unknown"
    try:
        head = subprocess.run(  # noqa: S603
            [git, "rev-parse", "HEAD"], cwd=repo_root(), capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        status = subprocess.run(  # noqa: S603
            [git, "status", "--porcelain", "--", "ml"],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"
    return f"{head}-dirty" if status else head


def seed_everything(seed: int) -> None:
    """Seeds Python, NumPy and (when installed) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002  (legacy global seed, for libraries that still read it)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def combine_hashes(entries: Iterable[tuple[str, str]]) -> str:
    """Hash of a dataset: sha256 over the sorted `(name, sha256)` pairs of its files."""
    digest = hashlib.sha256()
    for name, file_hash in sorted(entries):
        digest.update(f"{name}\0{file_hash}\n".encode())
    return digest.hexdigest()


def library_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": platform.python_version(), "tensoscan-ml": __version__}
    for package in _TRACKED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    return versions


def write_run_record(
    out_dir: str | Path,
    command: str,
    *,
    config: Config | None = None,
    seed: int | None = None,
    dataset_version: str | None = None,
    dataset_hash: str | None = None,
    extra: dict[str, Any] | None = None,
    name: str = RUN_RECORD_NAME,
) -> Path:
    """Writes `run.json` next to the outputs of a command and returns its path."""
    record: dict[str, Any] = {
        "command": command,
        "argv": sys.argv[1:],
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "seed": seed,
        "dataset_version": dataset_version,
        "dataset_hash": dataset_hash,
        "config": config or {},
        "versions": library_versions(),
    }
    if extra:
        record.update(extra)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    path = target / name
    path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path
