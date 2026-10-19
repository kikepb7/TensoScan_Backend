"""Building blocks of the detector (original implementation, project licence).

Everything is a convolution, batch norm, SiLU, nearest upsampling, concatenation or max-pooling: operations with
LiteRT builtins, so the exported graph needs no Flex or custom ops.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBnAct(nn.Module):
    """Conv2d + BatchNorm + SiLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel: int = 1, stride: int = 1, groups: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.act(self.bn(self.conv(x)))
        return out


class SeparableConv(nn.Module):
    """Depthwise 3x3 followed by pointwise 1x1 (what makes the "nano" models small)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = ConvBnAct(in_channels, in_channels, 3, stride, groups=in_channels)
        self.pointwise = ConvBnAct(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.pointwise(self.depthwise(x))
        return out


def conv3x3(in_channels: int, out_channels: int, stride: int, depthwise: bool) -> nn.Module:
    if depthwise:
        return SeparableConv(in_channels, out_channels, stride)
    return ConvBnAct(in_channels, out_channels, 3, stride)


class Bottleneck(nn.Module):
    def __init__(self, channels: int, shortcut: bool, depthwise: bool) -> None:
        super().__init__()
        self.reduce = ConvBnAct(channels, channels, 1)
        self.transform = conv3x3(channels, channels, 1, depthwise)
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y: torch.Tensor = self.transform(self.reduce(x))
        return x + y if self.shortcut else y


class CspBlock(nn.Module):
    """Cross-stage-partial block: two 1x1 branches, `depth` bottlenecks on one of them, concatenated and fused."""

    def __init__(self, in_channels: int, out_channels: int, depth: int, shortcut: bool, depthwise: bool) -> None:
        super().__init__()
        hidden = max(out_channels // 2, 4)
        self.main = ConvBnAct(in_channels, hidden, 1)
        self.side = ConvBnAct(in_channels, hidden, 1)
        self.blocks = nn.Sequential(*[Bottleneck(hidden, shortcut, depthwise) for _ in range(depth)])
        self.fuse = ConvBnAct(2 * hidden, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.fuse(torch.cat([self.blocks(self.main(x)), self.side(x)], dim=1))
        return out


class PoolBottleneck(nn.Module):
    """Spatial pyramid pooling (fast variant): three chained 5x5 max-pools."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 2, 4)
        self.reduce = ConvBnAct(channels, hidden, 1)
        self.pool = nn.MaxPool2d(5, 1, 2)
        self.fuse = ConvBnAct(hidden * 4, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)
        first = self.pool(x)
        second = self.pool(first)
        out: torch.Tensor = self.fuse(torch.cat([x, first, second, self.pool(second)], dim=1))
        return out
