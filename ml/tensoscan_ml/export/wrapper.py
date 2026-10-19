"""`OutputContractWrapper` (ADR-B003-03): puts the contract's output inside the graph.

It takes the per-level raw maps of a detector and returns `[B, N, 4 + C]` with `cx, cy, w, h` normalised to the model
input and sigmoid class scores (multiplied by the objectness), in the order level 0 (stride 8) then 1 then 2, each
row-major over `(y, x)`. There is **no NMS**: the app does it (contract section 6).
"""

from __future__ import annotations

import torch
from torch import nn

from tensoscan_ml.contract import INPUT_SIZE, STRIDES

_MAX_LOG_SIZE = 8.0  # exp() guard: keeps untrained or diverged heads from producing inf


def build_grid(input_size: int, strides: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-anchor `(x, y)` cell indices `[N, 2]` and stride `[N, 1]` for every level."""
    grids: list[torch.Tensor] = []
    stride_columns: list[torch.Tensor] = []
    for stride in strides:
        side = input_size // stride
        ys, xs = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
        grids.append(torch.stack([xs, ys], dim=-1).reshape(-1, 2).float())
        stride_columns.append(torch.full((side * side, 1), float(stride)))
    return torch.cat(grids), torch.cat(stride_columns)


def decode_levels(
    levels: list[torch.Tensor], grid: torch.Tensor, stride: torch.Tensor, input_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Raw maps -> boxes `[B,N,4]` (cx, cy, w, h in px), objectness logits `[B,N,1]` and class logits `[B,N,C]`."""
    flat = torch.cat([level.flatten(2).permute(0, 2, 1) for level in levels], dim=1)
    reg, obj, cls = flat[..., :4], flat[..., 4:5], flat[..., 5:]
    xy = (reg[..., :2] + grid) * stride
    wh = torch.exp(torch.clamp(reg[..., 2:4], max=_MAX_LOG_SIZE)) * stride
    del input_size
    return torch.cat([xy, wh], dim=-1), obj, cls


class OutputContractWrapper(nn.Module):
    """Raw detector -> contract output.

    `layout="nhwc"` accepts the contract's `[B, S, S, 3]` input (what the tflite model takes). `layout="nchw"` accepts
    `[B, 3, S, S]`: it is what the ONNX export uses, because onnx2tf turns an NCHW input into the NHWC tensor.
    """

    grid: torch.Tensor
    stride: torch.Tensor

    def __init__(
        self, model: nn.Module, input_size: int = INPUT_SIZE, strides: tuple[int, ...] = STRIDES, layout: str = "nhwc"
    ) -> None:
        super().__init__()
        if layout not in ("nhwc", "nchw"):
            raise ValueError("layout must be nhwc or nchw")
        self.model = model
        self.input_size = input_size
        self.layout = layout
        grid, stride = build_grid(input_size, strides)
        self.register_buffer("grid", grid)
        self.register_buffer("stride", stride)

    @property
    def num_candidates(self) -> int:
        return int(self.grid.shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layout == "nhwc":
            x = x.permute(0, 3, 1, 2)
        levels: list[torch.Tensor] = self.model(x)
        boxes, obj, cls = decode_levels(levels, self.grid, self.stride, self.input_size)
        scale = 1.0 / self.input_size
        scores = torch.sigmoid(cls) * torch.sigmoid(obj)
        return torch.cat([torch.clamp(boxes * scale, 0.0, 1.0), scores], dim=-1)
