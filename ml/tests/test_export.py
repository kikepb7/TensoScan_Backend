"""T-B003-10: export to the contract (needs the `export` extra; slow)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("onnx2tf")
pytest.importorskip("ai_edge_litert")
pytest.importorskip("tensorflow")

import torch

from tensoscan_ml.cli import main
from tensoscan_ml.detector.runtime import TfliteModel
from tensoscan_ml.export.pipeline import ExportOptions, export_checkpoint
from tensoscan_ml.export.wrapper import OutputContractWrapper
from tensoscan_ml.model.runtime import save_checkpoint
from tensoscan_ml.model.yolox import build_model

pytestmark = pytest.mark.export


def _tiny_checkpoint(tmp_path: Path) -> Path:
    torch.manual_seed(3)
    model = build_model({"name": "yolox-nano", "width": 0.125})
    path = tmp_path / "tiny.pt"
    save_checkpoint(path, model, {"name": "yolox-nano", "width": 0.125}, 320, {"epoch": 1, "stage": "test"})
    return path


def test_float32_export_of_a_checkpoint_matches_the_torch_wrapper(tmp_path: Path) -> None:
    from tensoscan_ml.model.runtime import build_from_checkpoint, load_checkpoint

    checkpoint = _tiny_checkpoint(tmp_path)

    result = export_checkpoint(
        checkpoint, tmp_path / "out", ExportOptions(version="1.2.3", precision="float32", dataset_version="ds-x")
    )

    reference = build_from_checkpoint(load_checkpoint(checkpoint))
    picture = torch.rand(1, 320, 320, 3)
    expected = reference(picture).detach().numpy()
    actual = TfliteModel(result.tflite).run(picture.numpy())
    assert result.tflite.name == "digits-1.2.3-float32.tflite"
    assert np.abs(actual - expected).max() < 1e-4
    sidecar = json.loads(result.sidecar.read_text(encoding="utf-8"))
    assert sidecar["version"] == "1.2.3" and sidecar["training"]["dataset_version"] == "ds-x"


def test_float16_is_the_default_precision_and_stays_close_to_float32(tmp_path: Path) -> None:
    from tensoscan_ml.model.runtime import build_from_checkpoint, load_checkpoint

    checkpoint = _tiny_checkpoint(tmp_path)

    result = export_checkpoint(checkpoint, tmp_path / "out", ExportOptions(version="1.0.0"))

    picture = torch.rand(1, 320, 320, 3)
    expected = build_from_checkpoint(load_checkpoint(checkpoint))(picture).detach().numpy()
    actual = TfliteModel(result.tflite).run(picture.numpy())
    assert result.tflite.name == "digits-1.0.0.tflite"
    assert "DEQUANTIZE" in result.info.operator_set
    assert np.abs(actual - expected).max() < 0.02
    assert result.tflite.stat().st_size < 6 * 1024 * 1024


def test_int8_export_takes_uint8_input_and_keeps_float_output(tmp_path: Path) -> None:
    from tensoscan_ml.synth.params import load_synth_config
    from tensoscan_ml.synth.writer import write_dataset

    write_dataset(
        tmp_path / "synth", load_synth_config(overrides={"image": {"width": 160, "height": 200}}), {"val": 6}, seed=2
    )
    checkpoint = _tiny_checkpoint(tmp_path)

    result = export_checkpoint(
        checkpoint,
        tmp_path / "out",
        ExportOptions(
            version="1.0.0", precision="int8", int8_images=tmp_path / "synth" / "val" / "images", int8_count=6
        ),
    )

    assert result.tflite.name == "digits-1.0.0-int8.tflite"
    assert result.info.inputs[0].dtype == "UINT8" and result.info.outputs[0].dtype == "FLOAT32"
    assert result.info.custom_operators == ()
    sidecar = json.loads(result.sidecar.read_text(encoding="utf-8"))
    assert sidecar["input"]["dtype"] == "uint8" and sidecar["input"]["range"] == [0, 255]
    output = TfliteModel(result.tflite).run(np.random.default_rng(0).random((1, 320, 320, 3), dtype=np.float32))
    assert output.shape == (1, 2100, 18) and np.isfinite(output).all()


def test_int8_without_representative_images_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="int8-images"):
        export_checkpoint(_tiny_checkpoint(tmp_path), tmp_path / "out", ExportOptions(precision="int8"))


def test_export_rejects_a_wrapper_that_is_not_in_the_export_layout(tmp_path: Path) -> None:
    from tensoscan_ml.export.pipeline import export_module

    wrapper = OutputContractWrapper(build_model({"name": "yolox-nano", "width": 0.125}), layout="nhwc")

    with pytest.raises(ValueError, match="layout='nchw'"):
        export_module(wrapper, tmp_path, ExportOptions())


def test_export_cli_writes_the_artifacts_and_the_run_record(tmp_path: Path) -> None:
    checkpoint = _tiny_checkpoint(tmp_path)

    actual = main(
        [
            "export",
            "--checkpoint",
            str(checkpoint),
            "--out",
            str(tmp_path / "dist"),
            "--version",
            "0.3.0",
            "--precision",
            "float32",
        ]
    )

    assert actual == 0
    for name in (
        "digits-0.3.0-float32.tflite",
        "digits-0.3.0-float32.json",
        "digits-0.3.0-float32.model-card.md",
        "run.json",
    ):
        assert (tmp_path / "dist" / name).is_file()
