"""Dataset writer: images, YOLO labels, metadata and manifests (REQ-B003-08, CA-B003-01)."""

from __future__ import annotations

import json
import multiprocessing
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tensoscan_ml.config import Config
from tensoscan_ml.contract import CLASSES
from tensoscan_ml.data.manifest import ManifestEntry, write_manifest
from tensoscan_ml.provenance import bytes_sha256, combine_hashes
from tensoscan_ml.synth.generate import Sample, encode_image, generate_sample, label_text


@dataclass(frozen=True)
class RenderedSample:
    """A sample already encoded, so it can travel between processes cheaply."""

    name: str
    image_bytes: bytes
    labels_text: str
    meta: dict[str, Any]


def render_one(args: tuple[Config, int, str, int]) -> RenderedSample:
    config, seed, split, index = args
    sample: Sample = generate_sample(config, seed, split, index)
    return RenderedSample(sample.name, encode_image(sample, config["image"]["format"]), label_text(sample), sample.meta)


def _render_all(config: Config, seed: int, split: str, count: int, workers: int) -> Iterator[RenderedSample]:
    jobs = [(config, seed, split, index) for index in range(count)]
    if workers <= 1:
        yield from map(render_one, jobs)
        return
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        yield from pool.map(render_one, jobs, chunksize=max(1, count // (workers * 8)))


def write_dataset(
    out_dir: str | Path, config: Config, counts: dict[str, int], *, seed: int, workers: int = 1
) -> dict[str, Any]:
    """Writes `<out>/<split>/{images,labels,meta.jsonl,manifest.jsonl}`, `classes.txt` and `dataset.json`.

    The output is a pure function of `(config, seed, counts)`: no timestamps or absolute paths are written, so two runs
    with the same seed produce identical files (CA-B003-01).
    """
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "classes.txt").write_text("".join(f"{name}\n" for name in CLASSES), encoding="utf-8")
    suffix = config["image"]["format"]
    hashes: list[tuple[str, str]] = []
    summary: dict[str, Any] = {}
    for split, count in counts.items():
        if count <= 0:
            continue
        images_dir, labels_dir = root / split / "images", root / split / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        entries: list[ManifestEntry] = []
        metas: list[str] = []
        stats = {"readable": 0, "parser_matches": 0, "covered": 0}
        for rendered in _render_all(config, seed, split, count, workers):
            image_rel = f"images/{rendered.name}.{suffix}"
            (root / split / image_rel).write_bytes(rendered.image_bytes)
            (labels_dir / f"{rendered.name}.txt").write_text(rendered.labels_text, encoding="utf-8")
            image_hash = bytes_sha256(rendered.image_bytes)
            hashes.append((f"{split}/{image_rel}", image_hash))
            hashes.append((f"{split}/labels/{rendered.name}.txt", bytes_sha256(rendered.labels_text.encode())))
            metas.append(json.dumps(rendered.meta, sort_keys=True))
            entries.append(_manifest_entry(rendered, image_rel, image_hash))
            stats["readable"] += int(rendered.meta["readable"])
            stats["covered"] += int(bool(rendered.meta["covered"]))
            stats["parser_matches"] += int(bool(rendered.meta["parser_matches"]))
        meta_text = "".join(line + "\n" for line in metas)
        (root / split / "meta.jsonl").write_text(meta_text, encoding="utf-8")
        hashes.append((f"{split}/meta.jsonl", bytes_sha256(meta_text.encode())))
        write_manifest(root / split / "manifest.jsonl", entries)
        summary[split] = {"count": count, **stats}

    info = {
        "name": "synthetic-digits",
        "dataset_version": config.get("dataset_version"),
        "seed": seed,
        "counts": {k: v for k, v in counts.items() if v > 0},
        "classes": list(CLASSES),
        "dataset_hash": combine_hashes(hashes),
        "summary": summary,
        "config": config,
    }
    (root / "dataset.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return info


def _manifest_entry(rendered: RenderedSample, image_rel: str, image_hash: str) -> ManifestEntry:
    meta = rendered.meta
    expected = meta["expected"]
    return ManifestEntry(
        path=image_rel,
        sha256=image_hash,
        device="synthetic",
        status=expected["status"] if expected else None,
        sys=expected["sys"] if expected else None,
        dia=expected["dia"] if expected else None,
        pul=expected["pul"] if expected else None,
        condition=meta["palette"],
        video_id=rendered.name,
        frame_idx=0,
        labels=f"labels/{rendered.name}.txt",
        extra={
            "template": meta["template"],
            "covered": meta["covered"],
            "readable": meta["readable"],
            "parser_compatible": meta["parser_compatible"],
        },
    )
