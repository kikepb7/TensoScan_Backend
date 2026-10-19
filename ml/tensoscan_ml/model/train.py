"""Two-stage training pipeline (REQ-B003-15): synthetic pre-training, then fine-tuning with real frames.

Seeds, checkpoints (`last.pt`, `best.pt` per stage), `metrics.csv`, `curves.png` and `run.json` (commit, dataset hashes,
config) are written under `<out_dir>/<name>/`. Stage 2 is skipped, with a warning, while no real data with boxes exists.
"""

from __future__ import annotations

import copy
import csv
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from tensoscan_ml.config import Config
from tensoscan_ml.console import echo, warn
from tensoscan_ml.contract import CLASSES, INPUT_SIZE, STRIDES
from tensoscan_ml.data.manifest import ManifestEntry
from tensoscan_ml.detector.runtime import ModelDetector, Sidecar
from tensoscan_ml.eval.metrics import detection_metrics, make_result
from tensoscan_ml.export.wrapper import OutputContractWrapper
from tensoscan_ml.model.curves import plot_curves
from tensoscan_ml.model.dataset import (
    AugmentConfig,
    DigitDataset,
    collate,
    cosine_with_warmup,
    items_from_manifest,
    items_from_yolo_dir,
    mixed_sampler,
    read_labels,
)
from tensoscan_ml.model.loss import YoloxLoss
from tensoscan_ml.model.runtime import TorchRawModel, save_checkpoint
from tensoscan_ml.model.yolox import build_model, count_parameters
from tensoscan_ml.provenance import file_sha256, seed_everything, write_run_record

METRIC_FIELDS = (
    "stage", "epoch", "train_loss", "train_iou", "train_obj", "train_cls", "val_loss",
    "val_map50_synthetic", "val_map50_real", "lr", "seconds",
)  # fmt: skip


@dataclass
class TrainSummary:
    run_dir: Path
    stages_run: list[str] = field(default_factory=list)
    stages_skipped: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    checkpoints: dict[str, Path] = field(default_factory=dict)
    parameters: int = 0


class ModelEma:
    """Exponential moving average of the weights; evaluation and checkpoints use it."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        self.updates = 0
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        decay = self.decay * (1.0 - math.exp(-self.updates / 500.0))
        source = model.state_dict()
        for name, value in self.module.state_dict().items():
            if value.dtype.is_floating_point:
                value.mul_(decay).add_(source[name].detach(), alpha=1.0 - decay)
            else:
                value.copy_(source[name])


def _expand(value: Any) -> Any:
    return os.path.expandvars(os.path.expanduser(value)) if isinstance(value, str) else value


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def ensure_synthetic(config: Config) -> Path:
    """The synthetic dataset folder; it is generated when missing and `autogenerate` is set."""
    section = config["data"]["synthetic"]
    root = Path(_expand(section["path"]))
    if (root / "train" / "images").is_dir():
        return root
    auto = section.get("autogenerate")
    if not auto:
        raise FileNotFoundError(f"synthetic dataset '{root}' not found; run `tensoscan-ml synth --out {root}`")
    from tensoscan_ml.synth.params import load_synth_config
    from tensoscan_ml.synth.writer import write_dataset

    echo(f"Generating the synthetic dataset in {root} ({auto})")
    synth = load_synth_config(overrides=section.get("synth_overrides"))
    write_dataset(root, synth, {"train": int(auto["train"]), "val": int(auto["val"])}, seed=int(config["seed"]))
    return root


def _real_items(config: Config, split: str) -> list[tuple[Path, Path | None]] | None:
    path = config["data"].get("real", {}).get(split)
    if not path or not Path(_expand(path)).is_file():
        return None
    items = items_from_manifest(_expand(path))
    return items or None


def _loader(
    dataset: Any, batch_size: int, sampler: Any = None, shuffle: bool = False, workers: int = 0
) -> DataLoader[Any]:
    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle and sampler is None, collate_fn=collate,
        num_workers=workers, drop_last=False,
    )  # fmt: skip


@torch.no_grad()
def validation_loss(model: nn.Module, criterion: YoloxLoss, loader: DataLoader[Any], device: torch.device) -> float:
    model.eval()
    total, count = 0.0, 0
    for images, targets in loader:
        levels = model(images.to(device))
        losses = criterion(levels, [t.to(device) for t in targets])
        total += float(losses["loss"]) * images.shape[0]
        count += images.shape[0]
    return total / max(count, 1)


def validation_map50(
    model: nn.Module, items: list[tuple[Path, Path | None]], input_size: int, limit: int, device: torch.device
) -> float | None:
    """mAP@0.5 of `model` (through the contract wrapper and the app's decoder) on up to `limit` images with boxes."""
    import cv2

    wrapper = OutputContractWrapper(model, input_size, STRIDES, layout="nhwc").eval()
    sidecar = Sidecar(CLASSES, wrapper.num_candidates, 0.4, 0.45, 32, input_size)
    detector = ModelDetector(TorchRawModel(wrapper, str(device)), "validation", sidecar)
    results = []
    for image_path, label_path in items[:limit]:
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        truth = [(int(c), _box(cx, cy, w, h)) for c, cx, cy, w, h in read_labels(label_path)]
        entry = ManifestEntry(str(image_path), "", "validation", None)
        results.append(
            make_result(entry, detector.detect(np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)), truth)
        )
    metrics = detection_metrics(results)
    return None if metrics is None else metrics["map50"]


def _box(cx: float, cy: float, w: float, h: float) -> Any:
    from tensoscan_ml.reading import Box

    return Box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def run_training(
    config: Config,
    *,
    out_root: str | Path | None = None,
    require_real: bool = False,
    only_stages: list[str] | None = None,
    init_from: str | Path | None = None,
    max_steps: int | None = None,
) -> TrainSummary:
    """Runs every stage of `config` in order and returns what was produced."""
    seed = int(config["seed"])
    input_size = int(config.get("input_size", INPUT_SIZE))
    device = _device(str(config.get("device", "auto")))
    run_dir = Path(out_root or _expand(config["out_dir"])) / str(config["name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    synthetic_root = ensure_synthetic(config)
    synthetic_train = items_from_yolo_dir(synthetic_root / "train")
    synthetic_val = items_from_yolo_dir(synthetic_root / "val")
    real_train, real_val = _real_items(config, "train"), _real_items(config, "val")

    arch = dict(config["arch"])
    model = build_model(arch).to(device)
    summary = TrainSummary(run_dir, parameters=count_parameters(model))
    echo(f"Architecture {arch['name']}: {summary.parameters / 1e6:.2f} M parameters on {device}")
    if init_from:
        state = torch.load(Path(init_from), map_location=device, weights_only=True)
        model.load_state_dict(state["state_dict"])

    dataset_info = {
        "synthetic": _dataset_hash(synthetic_root / "dataset.json"),
        "real_train": _manifest_hash(config, "train"),
    }
    write_run_record(
        run_dir,
        "train",
        config=config,
        seed=seed,
        dataset_version=str(config["name"]),
        dataset_hash=str(dataset_info),
        extra={"datasets": dataset_info},
    )

    criterion = YoloxLoss(model_classes(model), input_size, STRIDES).to(device)
    for stage_index, stage in enumerate(config["stages"]):
        name = str(stage["name"])
        if only_stages and name not in only_stages:
            continue
        needs_real = bool(stage.get("requires_real"))
        if needs_real and (real_train is None):
            message = f"stage '{name}' needs real frames with boxes (data.real.train) and none exist yet"
            if require_real:
                raise FileNotFoundError(message)
            warn(message + ": skipped (pre-trained synthetic checkpoint is the result)")
            summary.stages_skipped.append(name)
            continue
        seed_everything(seed + stage_index)
        _train_stage(
            model,
            criterion,
            stage,
            config,
            summary,
            device,
            input_size,
            synthetic_train,
            synthetic_val,
            real_train,
            real_val,
            max_steps,
        )
        summary.stages_run.append(name)

    if summary.rows:
        _write_metrics(run_dir / "metrics.csv", summary.rows)
        plot_curves(summary.rows, run_dir / "curves.png")
    return summary


def model_classes(model: nn.Module) -> int:
    count = getattr(model, "num_classes", None)
    return int(count) if count is not None else 14


def _dataset_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    import json

    value = json.loads(path.read_text(encoding="utf-8")).get("dataset_hash")
    return None if value is None else str(value)


def _manifest_hash(config: Config, split: str) -> str | None:
    path = config["data"].get("real", {}).get(split)
    return file_sha256(_expand(path)) if path and Path(_expand(path)).is_file() else None


def _train_stage(
    model: nn.Module,
    criterion: YoloxLoss,
    stage: dict[str, Any],
    config: Config,
    summary: TrainSummary,
    device: torch.device,
    input_size: int,
    synthetic_train: list[tuple[Path, Path | None]],
    synthetic_val: list[tuple[Path, Path | None]],
    real_train: list[tuple[Path, Path | None]] | None,
    real_val: list[tuple[Path, Path | None]] | None,
    max_steps: int | None,
) -> None:
    name, seed = str(stage["name"]), int(config["seed"])
    shares: dict[str, float] = dict(stage["sources"])
    augment = AugmentConfig() if stage.get("augment", True) else AugmentConfig.off()
    train_sets: list[DigitDataset] = []
    weights: list[float] = []
    if shares.get("synthetic"):
        train_sets.append(DigitDataset(synthetic_train, input_size, augment, seed))
        weights.append(float(shares["synthetic"]))
    if shares.get("real"):
        if real_train is None:
            raise FileNotFoundError(f"stage '{name}' lists real data but data.real.train has no boxes")
        train_sets.append(DigitDataset(real_train, input_size, augment, seed + 1))
        weights.append(float(shares["real"]))
    epoch_size = int(stage.get("samples_per_epoch", sum(len(d) for d in train_sets)))
    batch_size = int(stage["batch_size"])
    val_sets = {"synthetic": synthetic_val, **({"real": real_val} if real_val else {})}
    val_loaders = {
        key: _loader(DigitDataset(items[: int(stage.get("val_images", 300))], input_size), batch_size)
        for key, items in val_sets.items()
    }

    opt_cfg = config["optimizer"]
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for parameter_name, parameter in model.named_parameters():
        (no_decay if parameter.ndim <= 1 or parameter_name.endswith(".bias") else decay).append(parameter)
    base_lr = float(stage["lr"])
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": float(opt_cfg["weight_decay"])}, {"params": no_decay, "weight_decay": 0.0}],
        lr=base_lr,
    )
    ema = ModelEma(model, float(opt_cfg.get("ema_decay", 0.9995)))
    steps_per_epoch = max(1, math.ceil(epoch_size / batch_size))
    epochs = int(stage["epochs"])
    total_steps = steps_per_epoch * epochs
    warmup = int(float(stage.get("warmup_epochs", 1)) * steps_per_epoch)
    floor = float(stage.get("min_lr_ratio", 0.05))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    best_score, step = -math.inf, 0
    stage_dir = summary.run_dir / name
    echo(f"Stage {name}: {epochs} epochs x {steps_per_epoch} steps, batch {batch_size}, lr {base_lr}")

    for epoch in range(1, epochs + 1):
        started = time.time()
        for dataset in train_sets:
            dataset.set_epoch(epoch)
        data, sampler = mixed_sampler(
            train_sets, weights, epoch_size, seed * 1000 + epoch, replacement=bool(stage.get("replacement", True))
        )
        loader = _loader(data, batch_size, sampler=sampler)
        model.train()
        sums = {"loss": 0.0, "iou": 0.0, "obj": 0.0, "cls": 0.0}
        batches = 0
        for images, targets in loader:
            for group in optimizer.param_groups:
                group["lr"] = base_lr * cosine_with_warmup(step, total_steps, warmup, floor)
            with torch.autocast(device.type, enabled=use_amp):
                levels = model(images.to(device))
            losses = criterion([level.float() for level in levels], [t.to(device) for t in targets])
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(opt_cfg.get("grad_clip", 10.0)))
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            for key in sums:
                sums[key] += float(losses[key].detach())
            batches += 1
            step += 1
            if max_steps is not None and step >= max_steps:
                break
        row: dict[str, Any] = {"stage": name, "epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
        row.update({f"train_{key}": value / max(batches, 1) for key, value in sums.items()})
        row["val_loss"] = validation_loss(ema.module, criterion, val_loaders["synthetic"], device)
        evaluate_map = epoch % int(stage.get("eval_every", 5)) == 0 or epoch == epochs
        score = -row["val_loss"]
        if evaluate_map:
            for key, items in val_sets.items():
                value = validation_map50(ema.module, items, input_size, int(stage.get("val_images", 300)), device)
                row[f"val_map50_{key}"] = value
            primary = row.get("val_map50_real") if real_val else row.get("val_map50_synthetic")
            if primary is not None:
                score = float(primary)
        row["seconds"] = round(time.time() - started, 2)
        summary.rows.append(row)
        shown = row.get("val_map50_synthetic")
        echo(f"  epoch {epoch}/{epochs} loss {row['train_loss']:.3f} val {row['val_loss']:.3f} mAP50 {shown}")
        extra = {
            "stage": name,
            "epoch": epoch,
            "metrics": {k: v for k, v in row.items() if isinstance(v, (int, float))},
            "seed": seed,
        }
        save_checkpoint(stage_dir / "last.pt", ema.module, dict(config["arch"]), input_size, extra)
        if score > best_score:
            best_score = score
            save_checkpoint(stage_dir / "best.pt", ema.module, dict(config["arch"]), input_size, extra)
        if max_steps is not None and step >= max_steps:
            break

    summary.checkpoints[name] = stage_dir / "best.pt"
    model.load_state_dict(ema.module.state_dict())  # the next stage starts from the averaged weights


def _write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(METRIC_FIELDS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
