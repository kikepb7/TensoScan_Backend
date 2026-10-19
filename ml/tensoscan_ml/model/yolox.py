"""A compact anchor-free detector in the style of YOLOX-Nano (ADR-B003-02 default; original implementation).

Backbone with strides 8/16/32, a PAN neck and a decoupled head per level. Each level outputs
`[B, 4 + 1 + C, H, W]`: box regression (`dx, dy, log w, log h`), objectness logit and class logits, which is the
interface the loss and the `OutputContractWrapper` rely on. A different architecture plugs in by registering a builder
in `ARCHITECTURES` that returns a module with the same interface (see README, "Changing the architecture").
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from tensoscan_ml.contract import NUM_CLASSES, STRIDES
from tensoscan_ml.model.blocks import ConvBnAct, CspBlock, PoolBottleneck, conv3x3


class DetectionHead(nn.Module):
    """Decoupled head: a class branch and a box+objectness branch."""

    def __init__(self, in_channels: int, hidden: int, num_classes: int, depthwise: bool) -> None:
        super().__init__()
        self.stem = ConvBnAct(in_channels, hidden, 1)
        self.cls_branch = nn.Sequential(*[conv3x3(hidden, hidden, 1, depthwise) for _ in range(2)])
        self.reg_branch = nn.Sequential(*[conv3x3(hidden, hidden, 1, depthwise) for _ in range(2)])
        self.cls_pred = nn.Conv2d(hidden, num_classes, 1)
        self.reg_pred = nn.Conv2d(hidden, 4, 1)
        self.obj_pred = nn.Conv2d(hidden, 1, 1)
        self.set_score_bias(-math.log((1 - 0.01) / 0.01))  # start with 1 % probability: stable early training

    def set_score_bias(self, bias: float) -> None:
        """Sets the initial logit of the class and objectness predictions (0.01 probability by default)."""
        for conv in (self.cls_pred, self.obj_pred):
            assert conv.bias is not None  # noqa: S101
            nn.init.constant_(conv.bias, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        cls_feat, reg_feat = self.cls_branch(x), self.reg_branch(x)
        return torch.cat([self.reg_pred(reg_feat), self.obj_pred(reg_feat), self.cls_pred(cls_feat)], dim=1)


class YoloxNano(nn.Module):
    strides: tuple[int, ...] = STRIDES

    def __init__(
        self, depth: float = 0.33, width: float = 0.25, depthwise: bool = True, num_classes: int = NUM_CLASSES
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        c0, c1, c2, c3, c4 = (max(8, round(base * width)) for base in (64, 128, 256, 512, 1024))
        n = max(round(3 * depth), 1)
        self.stem = ConvBnAct(3, c0, 3, 2)  # stride 2
        self.stage1 = nn.Sequential(conv3x3(c0, c1, 2, depthwise), CspBlock(c1, c1, n, True, depthwise))  # stride 4
        self.stage2 = nn.Sequential(conv3x3(c1, c2, 2, depthwise), CspBlock(c2, c2, 3 * n, True, depthwise))  # stride 8
        self.stage3 = nn.Sequential(
            conv3x3(c2, c3, 2, depthwise), CspBlock(c3, c3, 3 * n, True, depthwise)
        )  # stride 16
        self.stage4 = nn.Sequential(  # stride 32
            conv3x3(c3, c4, 2, depthwise), PoolBottleneck(c4), CspBlock(c4, c4, n, False, depthwise)
        )
        self.lateral5 = ConvBnAct(c4, c3, 1)
        self.fuse4 = CspBlock(2 * c3, c3, n, False, depthwise)
        self.lateral4 = ConvBnAct(c3, c2, 1)
        self.fuse3 = CspBlock(2 * c2, c2, n, False, depthwise)
        self.down3 = conv3x3(c2, c2, 2, depthwise)
        self.pan4 = CspBlock(2 * c2, c2, n, False, depthwise)
        self.down4 = conv3x3(c2, c2, 2, depthwise)
        self.pan5 = CspBlock(c2 + c3, c2, n, False, depthwise)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.heads = nn.ModuleList([DetectionHead(c2, c2, num_classes, depthwise) for _ in self.strides])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stage1(self.stem(x))
        c2 = self.stage2(x)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        top5 = self.lateral5(c4)
        mid4 = self.lateral4(self.fuse4(torch.cat([self.up(top5), c3], dim=1)))
        out3 = self.fuse3(torch.cat([self.up(mid4), c2], dim=1))
        out4 = self.pan4(torch.cat([self.down3(out3), mid4], dim=1))
        out5 = self.pan5(torch.cat([self.down4(out4), top5], dim=1))
        return [head(feature) for head, feature in zip(self.heads, (out3, out4, out5), strict=True)]

    def set_score_bias(self, bias: float) -> None:
        """Sets the initial class/objectness logit of every head."""
        for head in self.heads:
            assert isinstance(head, DetectionHead)  # noqa: S101
            head.set_score_bias(bias)


ArchBuilder = Callable[[dict[str, Any]], nn.Module]


def _build_yolox_nano(arch: dict[str, Any]) -> nn.Module:
    return YoloxNano(
        depth=float(arch.get("depth", 0.33)),
        width=float(arch.get("width", 0.25)),
        depthwise=bool(arch.get("depthwise", True)),
        num_classes=NUM_CLASSES,
    )


#: Architectures selectable from the config (`arch.name`). Register another one here to plug it in.
ARCHITECTURES: dict[str, ArchBuilder] = {"yolox-nano": _build_yolox_nano}


def build_model(arch: dict[str, Any]) -> nn.Module:
    name = str(arch.get("name", "yolox-nano"))
    if name not in ARCHITECTURES:
        raise ValueError(f"unknown architecture '{name}'; available: {', '.join(sorted(ARCHITECTURES))}")
    return ARCHITECTURES[name](arch)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
