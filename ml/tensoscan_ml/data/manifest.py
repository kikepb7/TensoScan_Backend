"""JSONL manifests: one line per image, with its sha256 (REQ-B003-10, C-ML-03).

Manifests are what goes to git; the images stay outside. Paths are **relative to the directory of the manifest**.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ManifestEntry:
    path: str
    sha256: str
    device: str
    #: What the display shows. `status` is `COMPLETE` for a reading, `DEVICE_ERROR` for `E1`/`Err`, `MEASURING` for
    #: `---`, `NO_READING` for a blank display, or `None` when the image carries no reading ground truth.
    status: str | None
    sys: int | None = None
    dia: int | None = None
    pul: int | None = None
    condition: str | None = None
    video_id: str | None = None
    frame_idx: int | None = None
    #: YOLO label file (relative to the manifest) when the image has boxes.
    labels: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def reading(self) -> tuple[int | None, int | None, int | None]:
        return (self.sys, self.dia, self.pul)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ManifestEntry:
        known = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in raw.items() if key in known})


def read_manifest(path: str | Path) -> list[ManifestEntry]:
    entries: list[ManifestEntry] = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                entries.append(ManifestEntry.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError) as error:
                raise ValueError(f"{path}:{number}: invalid manifest line ({error})") from error
    return entries


def write_manifest(path: str | Path, entries: Iterable[ManifestEntry]) -> int:
    """Writes the entries sorted by path (stable, diff-friendly in git). Returns how many were written."""
    ordered = sorted(entries, key=lambda entry: entry.path)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for entry in ordered:
            handle.write(entry.to_json() + "\n")
    return len(ordered)


def iter_resolved(entries: Iterable[ManifestEntry], manifest_path: str | Path) -> Iterator[tuple[ManifestEntry, Path]]:
    """Entries with their image path resolved against the manifest's directory."""
    root = Path(manifest_path).parent
    for entry in entries:
        yield entry, root / entry.path
