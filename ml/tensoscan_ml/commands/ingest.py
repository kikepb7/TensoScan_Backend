"""`tensoscan-ml ingest`: videos and photos (with a file-name convention or labels.csv) -> frames + manifest."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from tensoscan_ml.commands import add_common_arguments
from tensoscan_ml.config import apply_overrides, data_dir, default_config_path, load_yaml, manifests_dir
from tensoscan_ml.console import echo, fail
from tensoscan_ml.provenance import write_run_record

HELP = "Turn videos and photos (named by convention or listed in labels.csv) into cropped frames and a manifest"


def configure(parser: argparse.ArgumentParser) -> None:
    add_common_arguments(parser)
    parser.add_argument("--src", required=True, help="folder with the videos/photos (and optionally labels.csv)")
    parser.add_argument("--out", help="output folder (default: $TENSOSCAN_ML_DATA_DIR/real/<dataset_version>)")
    parser.add_argument("--labels", help="labels.csv (default: <src>/labels.csv when it exists)")
    parser.add_argument("--dataset-version", help="name of the dataset, e.g. ds-real-0.1.0")
    parser.add_argument("--fps", type=float, help="frames per second taken from each video")
    parser.add_argument("--blur-threshold", type=float, help="drop frames whose Laplacian variance is below this")
    parser.add_argument("--hash-distance", type=int, help="pHash Hamming distance that makes two frames duplicates")
    parser.add_argument("--roi-mode", choices=("center", "none", "manual"), help="how to crop the region of interest")
    parser.add_argument("--max-side", type=int, help="longest side of the stored crops")
    parser.add_argument("--drop-blurry-photos", action="store_true", default=None, help="also drop blurry photos")
    parser.add_argument("--dedupe-photos", action="store_true", default=None, help="de-duplicate photos by pHash too")
    parser.add_argument("--preview", action="store_true", help="only write preview.jpg (framing check) and the report")
    parser.add_argument(
        "--commit-manifest",
        action="store_true",
        help="also copy the manifest to ml/datasets/manifests/<dataset_version>.jsonl (the only part that goes to git)",
    )


def run(args: argparse.Namespace) -> int:
    from tensoscan_ml.data.ingest import IngestOptions, copy_manifest, dataset_hash, ingest

    config = apply_overrides(load_yaml(default_config_path("ingest", "default")), [])
    if args.config:
        config = {**config, **load_yaml(args.config)}
    config = apply_overrides(config, args.overrides)
    for key in ("fps", "blur_threshold", "hash_distance", "roi_mode", "max_side", "dataset_version"):
        value = getattr(args, key.replace("-", "_"), None)
        if value is not None:
            config[key] = value
    for key in ("drop_blurry_photos", "dedupe_photos"):
        if getattr(args, key, None):
            config[key] = True
    options = IngestOptions(
        fps=float(config["fps"]),
        blur_threshold=float(config["blur_threshold"]),
        hash_distance=int(config["hash_distance"]),
        roi_mode=str(config["roi_mode"]),
        max_side=int(config["max_side"]),
        jpeg_quality=int(config["jpeg_quality"]),
        drop_blurry_photos=bool(config["drop_blurry_photos"]),
        dedupe_photos=bool(config["dedupe_photos"]),
        preview=bool(args.preview),
        preview_limit=int(config["preview_limit"]),
    )
    version = str(config["dataset_version"])
    out_dir = Path(args.out) if args.out else data_dir() / "real" / version
    report = ingest(args.src, out_dir, options, labels_csv=args.labels, dataset_version=version)

    summary = report.summary()
    echo(
        f"{summary['files']} files read: {summary['frames_kept']} frames kept, {summary['frames_blurry']} blurry, "
        f"{summary['frames_duplicate']} duplicates, {summary['rejections']} rejections in total."
    )
    for code, count in sorted(Counter(r.code for r in report.rejections).items()):
        echo(f"  rejected {code}: {count}")
    for rejection in [r for r in report.rejections if r.code not in ("BLURRY", "DUPLICATE")][:10]:
        echo(f"    {rejection.source}: {rejection.detail}")
    for warning in report.warnings[:10]:
        echo(f"  warning: {warning}")
    echo(f"Report: {out_dir / 'ingest_report.json'}")
    if options.preview:
        echo(f"Framing preview: {out_dir / 'preview.jpg'} (check it, then run again without --preview)")
        return 0
    if not report.entries:
        fail("no frame was kept; see the report for the reasons")
        return 1
    write_run_record(
        out_dir,
        "ingest",
        config=config,
        dataset_version=version,
        dataset_hash=dataset_hash(report.entries),
    )
    echo(f"Manifest: {out_dir / 'manifest.jsonl'} ({len(report.entries)} entries)")
    if args.commit_manifest:
        copied = copy_manifest(out_dir / "manifest.jsonl", manifests_dir(), version)
        echo(f"Committable manifest: {copied}")
    return 0
