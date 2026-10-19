"""Checkpoint -> ONNX -> TFLite with the contract sidecar (REQ-B003-18; conversion path: ADR-B003-07).

`torch.onnx.export` (legacy exporter, fixed opset) produces an NCHW graph; `onnx2tf` turns it into a TensorFlow
SavedModel and a float32/float16 `.tflite` with the contract's NHWC input; int8 goes through the TF converter with a
representative dataset. The result is renamed to the contract tensor names and checked for builtin ops only.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from tensoscan_ml.contract import INPUT_SIZE, INPUT_TENSOR_NAME, OUTPUT_TENSOR_NAME
from tensoscan_ml.export.sidecar import build_sidecar, render_model_card, write_sidecar
from tensoscan_ml.export.tflite_inspect import ModelInfo, inspect_bytes, rename_io
from tensoscan_ml.export.wrapper import OutputContractWrapper

PRECISIONS = ("float16", "float32", "int8")
DEFAULT_OPSET = 17
_CALIBRATION_FILE = "calibration_image_sample_data_20x128x128x3_float32.npy"


@dataclass
class ExportOptions:
    version: str = "0.1.0"
    precision: str = "float16"
    opset: int = DEFAULT_OPSET
    input_size: int = INPUT_SIZE
    int8_images: Path | None = None  # folder with representative images (about 300)
    int8_count: int = 300
    dataset_version: str = "unknown"
    metrics: dict[str, float] = field(default_factory=dict)
    keep_work: bool = False
    git_commit: str | None = None  # overrides the commit recorded in the sidecar (dummies are not built from a commit)

    def __post_init__(self) -> None:
        if self.precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {', '.join(PRECISIONS)}")


@dataclass
class ExportResult:
    tflite: Path
    sidecar: Path
    model_card: Path
    export_info: Path
    sha256: str
    info: ModelInfo


@contextmanager
def _workdir(path: Path | None) -> Iterator[Path]:
    """A scratch folder that is also the current directory (onnx2tf looks for a sample file there and would
    otherwise try to download it)."""
    previous = Path.cwd()
    holder = tempfile.TemporaryDirectory() if path is None else None
    root = Path(holder.name) if holder is not None else path
    assert root is not None  # noqa: S101
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / _CALIBRATION_FILE, np.random.default_rng(0).random((20, 128, 128, 3), dtype=np.float32))
    os.chdir(root)
    try:
        yield root
    finally:
        os.chdir(previous)
        if holder is not None:
            holder.cleanup()


def export_onnx(module: nn.Module, path: Path, input_size: int, opset: int) -> Path:
    """Exports the NCHW wrapper to ONNX with the contract tensor names."""
    module.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            module,
            (torch.rand(1, 3, input_size, input_size),),
            str(path),
            input_names=[INPUT_TENSOR_NAME],
            output_names=[OUTPUT_TENSOR_NAME],
            opset_version=opset,
            dynamo=False,
        )
    import onnx

    onnx.checker.check_model(onnx.load(str(path)))
    return path


def convert_onnx(onnx_path: Path, out_dir: Path) -> Path:
    """ONNX -> SavedModel + float32/float16 `.tflite` with onnx2tf. Returns the output folder."""
    import onnx2tf

    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(out_dir),
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True,
    )
    return out_dir


def representative_inputs(images: Path, count: int, input_size: int) -> list[np.ndarray[Any, Any]]:
    """Letterboxed float32 `[1, S, S, 3]` inputs from the images of a folder (the app's preprocessing)."""
    import cv2

    from tensoscan_ml.detector.preprocess import letterbox

    paths = sorted(p for p in images.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not paths:
        raise FileNotFoundError(f"no images found in {images} for the int8 calibration")
    step = max(1, len(paths) // count)
    samples = []
    for path in paths[::step][:count]:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is not None:
            samples.append(letterbox(np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8), input_size)[0])
    return samples


def quantize_int8(saved_model: Path, samples: list[np.ndarray[Any, Any]]) -> bytes:
    """Full-integer post-training quantisation: uint8 input, float32 output, builtin int8 kernels only."""
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: ([sample] for sample in samples)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.float32
    data: bytes = converter.convert()
    return data


def check_contract(info: ModelInfo, input_size: int, num_candidates: int, num_values: int, int8: bool) -> None:
    """Raises when the converted model does not have the contract's I/O or uses non-builtin operators."""
    if info.custom_operators:
        raise ValueError(f"the model uses custom/Flex operators: {', '.join(info.custom_operators)}")
    (model_input,), (model_output,) = info.inputs, info.outputs
    if model_input.shape != (1, input_size, input_size, 3):
        raise ValueError(f"input shape {model_input.shape}, expected (1, {input_size}, {input_size}, 3)")
    if model_input.dtype != ("UINT8" if int8 else "FLOAT32"):
        raise ValueError(f"input dtype {model_input.dtype}")
    if model_output.shape != (1, num_candidates, num_values) or model_output.dtype != "FLOAT32":
        raise ValueError(
            f"output {model_output.shape} {model_output.dtype}, expected (1, {num_candidates}, {num_values})"
        )


def export_module(
    module_nchw: nn.Module, out_dir: str | Path, options: ExportOptions, *, extra_info: dict[str, Any] | None = None
) -> ExportResult:
    """Exports a contract wrapper (NCHW layout) to `digits-<version>[-int8].tflite` + sidecar + model card."""
    if not isinstance(module_nchw, OutputContractWrapper) or module_nchw.layout != "nchw":
        raise ValueError("export_module expects an OutputContractWrapper with layout='nchw'")
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    int8 = options.precision == "int8"
    suffix = "-int8" if int8 else ("-float32" if options.precision == "float32" else "")
    stem = f"digits-{options.version}{suffix}"
    work_root = (target / "work") if options.keep_work else None
    with _workdir(work_root) as work:
        onnx_path = export_onnx(module_nchw, work / "model.onnx", options.input_size, options.opset)
        converted = convert_onnx(onnx_path, work / "tf")
        name = "model_float32.tflite" if options.precision == "float32" else "model_float16.tflite"
        if int8:
            if options.int8_images is None:
                raise ValueError("int8 export needs --int8-images with representative frames")
            data = quantize_int8(
                converted, representative_inputs(options.int8_images, options.int8_count, options.input_size)
            )
        else:
            data = (converted / name).read_bytes()
    data = rename_io(data, INPUT_TENSOR_NAME, OUTPUT_TENSOR_NAME)
    info = inspect_bytes(data)
    check_contract(info, options.input_size, module_nchw.num_candidates, 4 + 14, int8)

    tflite_path = target / f"{stem}.tflite"
    tflite_path.write_bytes(data)
    sidecar = build_sidecar(
        tflite_path,
        version=options.version,
        num_candidates=module_nchw.num_candidates,
        int8=int8,
        dataset_version=options.dataset_version,
        metrics=options.metrics,
        input_size=options.input_size,
    )
    if options.git_commit is not None:
        sidecar["training"]["git_commit"] = options.git_commit
    sidecar_path = write_sidecar(target / f"{stem}.json", sidecar)
    card_path = target / f"{stem}.model-card.md"
    card_path.write_text(
        render_model_card(sidecar, filename=tflite_path.name, size_bytes=len(data), operators=list(info.operator_set)),
        encoding="utf-8",
    )
    export_info = {
        "precision": options.precision,
        "onnx_opset": options.opset,
        "operators": sorted(info.operator_set),
        "input": vars(info.inputs[0]),
        "output": vars(info.outputs[0]),
        "size_bytes": len(data),
        "converter": _tool_versions(),
        **(extra_info or {}),
    }
    info_path = target / f"{stem}.export-info.json"
    info_path.write_text(json.dumps(export_info, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if not options.keep_work:
        shutil.rmtree(target / "work", ignore_errors=True)
    return ExportResult(tflite_path, sidecar_path, card_path, info_path, sidecar["sha256"], info)


def _tool_versions() -> dict[str, str]:
    from importlib import metadata

    versions = {}
    for package in ("torch", "onnx", "onnx2tf", "tensorflow", "ai-edge-litert"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    return versions


def export_checkpoint(checkpoint_path: str | Path, out_dir: str | Path, options: ExportOptions) -> ExportResult:
    """Exports a training checkpoint."""
    from tensoscan_ml.model.runtime import build_from_checkpoint, load_checkpoint

    checkpoint = load_checkpoint(checkpoint_path)
    wrapper = build_from_checkpoint(checkpoint, layout="nchw")
    options.input_size = int(checkpoint["input_size"])
    extra = {"checkpoint": str(checkpoint_path), "epoch": checkpoint.get("epoch"), "stage": checkpoint.get("stage")}
    return export_module(wrapper, out_dir, options, extra_info=extra)
