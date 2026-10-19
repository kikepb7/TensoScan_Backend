"""T-B003-05: scene, augmentations, deterministic CLI and gallery (REQ-B003-07/08, CA-B003-01/02)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from tensoscan_ml.cli import main
from tensoscan_ml.config import Config
from tensoscan_ml.synth.generate import generate_sample
from tensoscan_ml.synth.params import load_synth_config
from tensoscan_ml.synth.scene import background


@pytest.fixture(scope="module")
def config() -> Config:
    return load_synth_config()


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "run.json"
    }


def test_same_seed_gives_byte_identical_dataset_and_the_worker_count_does_not_matter(tmp_path: Path) -> None:
    """CA-B003-01: seed 42, 100 images, twice."""
    first, second = tmp_path / "a", tmp_path / "b"

    assert main(["synth", "--out", str(first), "--seed", "42", "--train", "100", "--val", "0", "--test", "0"]) == 0
    parallel = ["synth", "--out", str(second), "--seed", "42", "--train", "100", "--val", "0", "--test", "0"]
    assert main([*parallel, "--workers", "2"]) == 0

    hashes = _tree_hashes(first)
    assert len([name for name in hashes if name.endswith(".jpg")]) == 100
    assert hashes == _tree_hashes(second)


def test_a_different_seed_gives_different_images(config: Config) -> None:
    first = generate_sample(config, 1, "train", 0)
    second = generate_sample(config, 2, "train", 0)

    assert not np.array_equal(first.image, second.image)


def test_samples_are_independent_of_each_other(config: Config) -> None:
    alone = generate_sample(config, 5, "train", 17)
    after_others = [generate_sample(config, 5, "train", i) for i in (3, 17)][-1]

    assert np.array_equal(alone.image, after_others.image)
    assert alone.labels == after_others.labels


def test_splits_use_independent_random_streams(config: Config) -> None:
    train = generate_sample(config, 5, "train", 0)
    val = generate_sample(config, 5, "val", 0)

    assert not np.array_equal(train.image, val.image)


def test_every_label_box_hugs_its_rendered_glyph_with_iou_above_090(config: Config) -> None:
    """CA-B003-02 on whole scenes: boxes come from the same homography that draws the glyph."""
    checked = 0
    for index in range(25):
        sample = generate_sample(config, 42, "val", index, with_masks=True)
        height, width = sample.image.shape[:2]
        assert len(sample.masks) == len(sample.labels)
        for label, mask in zip(sample.labels, sample.masks, strict=True):
            ys, xs = np.nonzero(mask > 127)
            if len(xs) == 0 or label.h * height < 40 or label.w * width < 20:
                continue
            tight = (xs.min() / width, ys.min() / height, (xs.max() + 1) / width, (ys.max() + 1) / height)
            box = (label.cx - label.w / 2, label.cy - label.h / 2, label.cx + label.w / 2, label.cy + label.h / 2)
            inter_w = max(0.0, min(box[2], tight[2]) - max(box[0], tight[0]))
            inter_h = max(0.0, min(box[3], tight[3]) - max(box[1], tight[1]))
            inter = inter_w * inter_h
            union = (box[2] - box[0]) * (box[3] - box[1]) + (tight[2] - tight[0]) * (tight[3] - tight[1]) - inter

            assert inter / union >= 0.90, (index, label.symbol)
            checked += 1
    assert checked >= 60


def test_display_occupies_the_configured_share_of_the_crop(config: Config) -> None:
    fractions = [generate_sample(config, 8, "train", i).meta["scene"]["display_fraction"] for i in range(40)]

    assert min(fractions) >= 0.40
    assert max(fractions) <= 0.95


def test_camera_angles_stay_inside_the_configured_ranges(config: Config) -> None:
    scenes = [generate_sample(config, 8, "train", i).meta["scene"] for i in range(40)]

    assert max(abs(s["yaw_deg"]) for s in scenes) <= 25.0
    assert max(abs(s["pitch_deg"]) for s in scenes) <= 25.0
    assert max(abs(s["roll_deg"]) for s in scenes) <= 8.0


def test_augmentations_use_the_ranges_of_the_config(config: Config) -> None:
    used = [generate_sample(config, 12, "train", i).meta["augmentation"] for i in range(60)]

    assert any("glare_alpha" in u for u in used)
    assert any("motion_blur" in u or "gaussian_blur" in u for u in used)
    assert all(0.5 <= u["gamma"] <= 1.8 for u in used)
    assert all(0.2 <= u["glare_alpha"] <= 0.9 for u in used if "glare_alpha" in u)
    assert all(40 <= generate_sample(config, 12, "train", i).jpeg_quality <= 95 for i in range(10))


def test_backgrounds_are_textured_not_flat() -> None:
    texture = background(np.random.default_rng(0), 64, 64)

    assert texture.std() > 3.0


def test_gallery_command_writes_a_small_png(tmp_path: Path) -> None:
    target = tmp_path / "gallery.png"

    assert main(["synth", "--gallery", str(target), "--gallery-count", "12"]) == 0

    assert target.is_file()
    assert target.stat().st_size < 1024 * 1024


def test_synth_cli_rejects_a_config_whose_shares_do_not_add_up(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("layouts:\n  error: 0.9\n", encoding="utf-8")

    actual = main(["synth", "--config", str(bad), "--out", str(tmp_path / "o"), "--train", "1", "--val", "0"])

    assert actual == 2
    assert "add up to 1" in capsys.readouterr().err
