"""Checkpoint loading and inference: the `checkpoint` source of `eval` and `prelabel`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from tensoscan_ml.contract import CLASSES, MAX_DETECTIONS, NMS_IOU_THRESHOLD, SCORE_THRESHOLD, STRIDES
from tensoscan_ml.detector.runtime import ModelDetector, Sidecar
from tensoscan_ml.export.wrapper import OutputContractWrapper
from tensoscan_ml.model.yolox import build_model


def save_checkpoint(
    path: Path, model: torch.nn.Module, arch: dict[str, Any], input_size: int, extra: dict[str, Any]
) -> None:
    """Weights plus what is needed to rebuild the model; only plain Python types and tensors (loadable safely)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"arch": arch, "input_size": input_size, "classes": list(CLASSES), "state_dict": model.state_dict(), **extra},
        path,
    )


def load_checkpoint(path: str | Path, device: str = "cpu") -> dict[str, Any]:
    loaded: dict[str, Any] = torch.load(Path(path), map_location=device, weights_only=True)
    return loaded


def build_from_checkpoint(checkpoint: dict[str, Any], layout: str = "nhwc") -> OutputContractWrapper:
    """The contract wrapper around the trained weights, in eval mode."""
    model = build_model(checkpoint["arch"])
    model.load_state_dict(checkpoint["state_dict"])
    wrapper = OutputContractWrapper(model, int(checkpoint["input_size"]), STRIDES, layout=layout)
    return wrapper.eval()


class TorchRawModel:
    """Runs the wrapped network on CPU/GPU with NumPy in and out."""

    def __init__(self, wrapper: OutputContractWrapper, device: str = "cpu") -> None:
        self.wrapper = wrapper.to(device)
        self.device = device

    @torch.no_grad()
    def run(self, model_input: NDArray[np.float32]) -> NDArray[np.float32]:
        output = self.wrapper(torch.from_numpy(model_input).to(self.device))
        return np.asarray(output.cpu().numpy(), dtype=np.float32)


def checkpoint_detector(path: str | Path, device: str = "cpu") -> ModelDetector:
    checkpoint = load_checkpoint(path, device)
    wrapper = build_from_checkpoint(checkpoint)
    sidecar = Sidecar(
        classes=tuple(checkpoint.get("classes", CLASSES)),
        num_candidates=wrapper.num_candidates,
        score_threshold=SCORE_THRESHOLD,
        nms_iou_threshold=NMS_IOU_THRESHOLD,
        max_detections=MAX_DETECTIONS,
        input_size=int(checkpoint["input_size"]),
    )
    return ModelDetector(TorchRawModel(wrapper, device), f"checkpoint:{Path(path).name}", sidecar)
