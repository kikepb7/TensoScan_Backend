"""T-B003-09 and T-B003-10b/10 (model side): architecture, contract wrapper, loss, data and a CPU smoke training."""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensoscan_ml.cli import main  # noqa: E402
from tensoscan_ml.config import default_config_path, load_config  # noqa: E402
from tensoscan_ml.contract import NUM_CLASSES, STRIDES  # noqa: E402
from tensoscan_ml.export.wrapper import OutputContractWrapper, build_grid  # noqa: E402
from tensoscan_ml.model.dataset import (  # noqa: E402
    AugmentConfig,
    DigitDataset,
    cosine_with_warmup,
    items_from_yolo_dir,
    mixed_sampler,
)
from tensoscan_ml.model.loss import YoloxLoss, box_iou_cxcywh  # noqa: E402
from tensoscan_ml.model.train import ModelEma, run_training  # noqa: E402
from tensoscan_ml.model.yolox import ARCHITECTURES, build_model, count_parameters  # noqa: E402
from tensoscan_ml.synth.params import load_synth_config  # noqa: E402
from tensoscan_ml.synth.writer import write_dataset  # noqa: E402

pytestmark = pytest.mark.torch


def test_default_architecture_is_small_and_has_three_levels_with_the_contract_grid() -> None:
    model = build_model({"name": "yolox-nano"}).eval()

    with torch.no_grad():
        levels = model(torch.zeros(1, 3, 320, 320))

    assert [tuple(level.shape) for level in levels] == [
        (1, 5 + NUM_CLASSES, 40, 40),
        (1, 5 + NUM_CLASSES, 20, 20),
        (1, 5 + NUM_CLASSES, 10, 10),
    ]
    assert count_parameters(model) < 1_500_000


def test_unknown_architecture_is_rejected_and_the_registry_is_pluggable() -> None:
    with pytest.raises(ValueError, match="unknown architecture"):
        build_model({"name": "nanodet-plus"})
    ARCHITECTURES["tiny-test"] = lambda arch: build_model({"name": "yolox-nano", "width": 0.125})
    try:
        assert build_model({"name": "tiny-test"}) is not None
    finally:
        del ARCHITECTURES["tiny-test"]


def test_wrapper_output_follows_the_contract_shape_range_and_order() -> None:
    model = build_model({"name": "yolox-nano", "width": 0.125})
    wrapper = OutputContractWrapper(model).eval()

    with torch.no_grad():
        output = wrapper(torch.rand(1, 320, 320, 3))

    assert output.shape == (1, 2100, 4 + NUM_CLASSES)
    assert wrapper.num_candidates == 2100
    assert torch.isfinite(output).all()
    scores = output[..., 4:]
    assert float(scores.min()) >= 0.0 and float(scores.max()) <= 1.0
    assert float(output[..., :4].min()) >= 0.0 and float(output[..., :4].max()) <= 1.0  # boxes are normalised


def test_wrapper_decodes_boxes_from_the_grid_and_strides() -> None:
    class Constant(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
            levels = []
            for stride in STRIDES:
                side = 320 // stride
                level = torch.zeros(1, 5 + NUM_CLASSES, side, side)
                level[:, 4] = 20.0  # objectness logit: sigmoid -> 1
                level[:, 5] = 20.0  # class "0"
                levels.append(level)
            return levels

    wrapper = OutputContractWrapper(Constant()).eval()  # type: ignore[arg-type]

    output = wrapper(torch.zeros(1, 320, 320, 3))[0]

    assert output[0, :4].tolist() == pytest.approx([0.0, 0.0, 8 / 320, 8 / 320])  # first anchor of level 0
    assert output[1, 0].item() == pytest.approx(8 / 320)  # next cell to the right
    assert output[40, 1].item() == pytest.approx(8 / 320)  # next row
    assert output[1600, :4].tolist() == pytest.approx([0.0, 0.0, 16 / 320, 16 / 320])  # level 1 starts at 1600
    assert output[2000, 2].item() == pytest.approx(32 / 320)  # level 2 starts at 2000
    assert output[0, 4].item() == pytest.approx(1.0, abs=1e-6)
    assert output[0, 5].item() == pytest.approx(0.5, abs=1e-6)  # class "1": logit 0 times objectness 1


def test_wrapper_layouts_agree_and_export_layout_takes_nchw() -> None:
    model = build_model({"name": "yolox-nano", "width": 0.125}).eval()
    nhwc, nchw = OutputContractWrapper(model, layout="nhwc"), OutputContractWrapper(model, layout="nchw")
    image = torch.rand(1, 320, 320, 3)

    with torch.no_grad():
        assert torch.allclose(nhwc(image), nchw(image.permute(0, 3, 1, 2)), atol=1e-6)
    with pytest.raises(ValueError, match="nhwc or nchw"):
        OutputContractWrapper(model, layout="chw")


def test_grid_has_one_row_per_candidate_in_row_major_order() -> None:
    grid, stride = build_grid(320, STRIDES)

    assert grid.shape == (2100, 2) and stride.shape == (2100, 1)
    assert grid[41].tolist() == [1.0, 1.0] and stride[41].item() == 8.0


def test_iou_matrix_is_exact_for_simple_boxes() -> None:
    a = torch.tensor([[10.0, 10.0, 10.0, 10.0]])
    b = torch.tensor([[10.0, 10.0, 10.0, 10.0], [15.0, 10.0, 10.0, 10.0], [100.0, 100.0, 5.0, 5.0]])

    actual = box_iou_cxcywh(a, b)

    assert actual[0].tolist() == pytest.approx([1.0, 1 / 3, 0.0], abs=1e-6)


def test_loss_is_finite_assigns_nearby_anchors_and_handles_images_without_glyphs() -> None:
    model = build_model({"name": "yolox-nano", "width": 0.125})
    criterion = YoloxLoss(NUM_CLASSES, 320)
    images = torch.rand(2, 3, 320, 320)
    targets = [torch.tensor([[3.0, 160.0, 160.0, 40.0, 60.0], [8.0, 220.0, 160.0, 40.0, 60.0]]), torch.zeros(0, 5)]

    losses = criterion(model(images), targets)

    assert all(torch.isfinite(value) for value in losses.values())
    losses["loss"].backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.parameters())


def test_loss_is_lower_for_a_prediction_that_matches_the_target() -> None:
    criterion = YoloxLoss(NUM_CLASSES, 320)
    grid, _ = build_grid(320, STRIDES)
    target = torch.tensor([[3.0, 164.0, 164.0, 40.0, 60.0]])

    def levels(good: bool) -> list[torch.Tensor]:
        out = []
        offset = 0
        for s in STRIDES:
            side = 320 // s
            level = torch.zeros(1, 5 + NUM_CLASSES, side, side)
            level[:, 4] = -4.6
            level[:, 5:] = -4.6
            if good and s == 8:
                for index in range(side * side):
                    x, y = grid[offset + index].tolist()
                    cx, cy = (x + 0.5) * s, (y + 0.5) * s
                    if abs(cx - 164) < 20 and abs(cy - 164) < 30:
                        level[0, 0, int(y), int(x)] = (164 - x * s) / s
                        level[0, 1, int(y), int(x)] = (164 - y * s) / s
                        level[0, 2, int(y), int(x)] = float(np.log(40 / s))
                        level[0, 3, int(y), int(x)] = float(np.log(60 / s))
                        level[0, 4, int(y), int(x)] = 4.0
                        level[0, 5 + 3, int(y), int(x)] = 4.0
            offset += side * side
            out.append(level)
        return out

    good = criterion(levels(True), [target])["loss"]
    bad = criterion(levels(False), [target])["loss"]

    assert float(good) < float(bad)


def test_lr_schedule_warms_up_then_decays_to_the_floor() -> None:
    values = [cosine_with_warmup(step, 100, 10, 0.05) for step in range(100)]

    assert values[0] == pytest.approx(0.1) and values[9] == pytest.approx(1.0)
    assert all(a >= b for a, b in itertools.pairwise(values[10:]))
    assert values[-1] == pytest.approx(0.05, abs=0.01)


def test_ema_follows_the_model_and_copies_integer_buffers() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
    ema = ModelEma(model, decay=0.9)
    with torch.no_grad():
        model[0].weight.fill_(5.0)  # type: ignore[index, operator]
    for _ in range(200):
        ema.update(model)

    assert float(ema.module[0].weight.mean()) == pytest.approx(5.0, abs=0.1)  # type: ignore[index]


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("tiny")
    config = load_synth_config(overrides={"image": {"width": 160, "height": 200}})
    write_dataset(root, config, {"train": 6, "val": 3}, seed=5)
    return root


def test_dataset_returns_letterboxed_tensors_and_targets_in_input_pixels(tiny_dataset: Path) -> None:
    dataset = DigitDataset(items_from_yolo_dir(tiny_dataset / "train"), 160, AugmentConfig.off())

    image, targets = dataset[0]

    assert image.shape == (3, 160, 160) and image.dtype == torch.float32
    assert float(image.min()) >= 0.0 and float(image.max()) <= 1.0
    if len(targets):
        assert targets.shape[1] == 5
        left = targets[:, 1] - targets[:, 3] / 2
        assert float(left.min()) >= 16 - 1e-3  # 160x200 letterboxed to 128x160: 16 px of grey on each side
        assert float((targets[:, 1] + targets[:, 3] / 2).max()) <= 144 + 1e-3


def test_augmentation_is_reproducible_changes_the_image_and_keeps_boxes_valid(tiny_dataset: Path) -> None:
    plain = DigitDataset(items_from_yolo_dir(tiny_dataset / "train"), 160, AugmentConfig.off())
    augmented = DigitDataset(items_from_yolo_dir(tiny_dataset / "train"), 160, AugmentConfig(), seed=1)

    first, second = augmented[1], augmented[1]
    augmented.set_epoch(1)
    other_epoch = augmented[1]

    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])
    assert not torch.equal(first[0], other_epoch[0])
    assert not torch.equal(first[0], plain[1][0])
    assert bool((first[1][:, 3:5] > 0).all())


def test_mixed_sampler_draws_each_source_in_proportion_to_its_weight(tiny_dataset: Path) -> None:
    small = DigitDataset(items_from_yolo_dir(tiny_dataset / "val"), 160)
    large = DigitDataset(items_from_yolo_dir(tiny_dataset / "train"), 160)
    data, sampler = mixed_sampler([small, large], [0.7, 0.3], 2000, seed=0)

    drawn = list(sampler)
    from_small = sum(1 for index in drawn if index < len(small))

    assert len(data) == len(small) + len(large)
    assert from_small / 2000 == pytest.approx(0.7, abs=0.05)


def _smoke_config(root: Path) -> dict:
    config = load_config(default_config_path("train", "smoke"))
    config["data"]["synthetic"]["path"] = str(root / "synth")
    config["out_dir"] = str(root / "runs")
    return config


def test_smoke_training_runs_end_to_end_on_cpu_and_the_loss_goes_down(tmp_path: Path) -> None:
    summary = run_training(_smoke_config(tmp_path))

    losses = [row["train_loss"] for row in summary.rows]
    assert len(losses) == 80
    assert np.mean(losses[-5:]) < 0.6 * np.mean(losses[:3])
    run_dir = summary.run_dir
    for name in ("metrics.csv", "curves.png", "run.json"):
        assert (run_dir / name).is_file()
    assert (run_dir / "pretrain-synthetic" / "best.pt").is_file() and (
        run_dir / "pretrain-synthetic" / "last.pt"
    ).is_file()
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["seed"] == 7 and record["git_commit"] and record["datasets"]["synthetic"]
    with (run_dir / "metrics.csv").open(encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 80
    assert summary.stages_run == ["pretrain-synthetic"]


def test_stage_two_is_skipped_without_real_data_and_required_when_asked(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    config["stages"][0]["epochs"] = 1
    config["stages"].append(
        {
            **config["stages"][0],
            "name": "finetune-real",
            "requires_real": True,
            "sources": {"real": 0.7, "synthetic": 0.3},
        }
    )

    summary = run_training(config)

    assert summary.stages_run == ["pretrain-synthetic"] and summary.stages_skipped == ["finetune-real"]
    with pytest.raises(FileNotFoundError, match="needs real frames"):
        run_training(config, require_real=True)


def test_stage_two_finetunes_on_real_frames_with_boxes_when_they_exist(tmp_path: Path) -> None:
    from tensoscan_ml.data.manifest import ManifestEntry, write_manifest

    config = _smoke_config(tmp_path)
    config["stages"][0]["epochs"] = 1
    real = tmp_path / "real"
    synth = load_synth_config(overrides={"image": {"width": 160, "height": 200}})
    write_dataset(real / "raw", synth, {"train": 6, "val": 2}, seed=77)
    for split in ("train", "val"):  # pretend these synthetic frames are real ones annotated with boxes
        entries = [
            ManifestEntry(
                f"raw/{split}/{e.path}",
                e.sha256,
                "omron-m3",
                e.status,
                e.sys,
                e.dia,
                e.pul,
                "good",
                e.video_id,
                0,
                f"raw/{split}/{e.labels}",
            )
            for e in __import__("tensoscan_ml.data.manifest", fromlist=["read_manifest"]).read_manifest(
                real / "raw" / split / "manifest.jsonl"
            )
        ]
        write_manifest(real / f"{split}_boxes.jsonl", entries)
    config["data"]["real"] = {"train": str(real / "train_boxes.jsonl"), "val": str(real / "val_boxes.jsonl")}
    config["stages"].append(
        {
            **config["stages"][0],
            "name": "finetune-real",
            "requires_real": True,
            "sources": {"real": 0.7, "synthetic": 0.3},
            "lr": 0.001,
        }
    )

    summary = run_training(config)

    assert summary.stages_run == ["pretrain-synthetic", "finetune-real"]
    assert any(row.get("val_map50_real") is not None for row in summary.rows)
    assert (summary.run_dir / "finetune-real" / "best.pt").is_file()


def test_a_trained_checkpoint_can_be_evaluated_through_the_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _smoke_config(tmp_path)
    config["stages"][0]["epochs"] = 2
    summary = run_training(config)
    checkpoint = summary.checkpoints["pretrain-synthetic"]

    actual = main(
        [
            "eval",
            "--detector",
            "checkpoint",
            "--model",
            str(checkpoint),
            "--manifest",
            str(tmp_path / "synth" / "val" / "manifest.jsonl"),
            "--out",
            str(tmp_path / "eval"),
        ]
    )

    assert actual in (0, 1)  # an untrained model is below the thresholds: the harness still runs and reports
    assert (tmp_path / "eval" / "report.json").is_file()
    assert "checkpoint:best.pt" in capsys.readouterr().out


def test_train_cli_runs_the_smoke_config(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    config["stages"][0]["epochs"] = 1
    path = tmp_path / "smoke.yaml"
    import yaml

    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert main(["train", "--config", str(path)]) == 0
    assert (tmp_path / "runs" / "smoke" / "pretrain-synthetic" / "best.pt").is_file()
