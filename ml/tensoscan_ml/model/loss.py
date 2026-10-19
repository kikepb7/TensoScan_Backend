"""YOLOX-style loss with SimOTA label assignment (original implementation).

Per image, every ground-truth glyph is matched to a dynamic number of anchors chosen by the lowest cost (classification
plus IoU), restricted to anchors near the glyph. The loss is the IoU loss of the matched boxes, the objectness BCE of
all anchors and the IoU-aware classification BCE of the matched anchors.
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn

from tensoscan_ml.contract import INPUT_SIZE, STRIDES
from tensoscan_ml.export.wrapper import build_grid, decode_levels

CENTER_RADIUS = 2.5  # in strides
CANDIDATE_K = 10
EPS = 1e-8


def box_iou_cxcywh(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU between `[G, 4]` and `[A, 4]` boxes given as cx, cy, w, h; returns `[G, A]`."""
    a_min, a_max = a[:, None, :2] - a[:, None, 2:] / 2, a[:, None, :2] + a[:, None, 2:] / 2
    b_min, b_max = b[None, :, :2] - b[None, :, 2:] / 2, b[None, :, :2] + b[None, :, 2:] / 2
    inter_wh = (torch.minimum(a_max, b_max) - torch.maximum(a_min, b_min)).clamp(min=0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]
    union = a[:, None, 2] * a[:, None, 3] + b[None, :, 2] * b[None, :, 3] - inter
    return inter / (union + EPS)


class YoloxLoss(nn.Module):
    grid: torch.Tensor
    stride: torch.Tensor

    def __init__(self, num_classes: int, input_size: int = INPUT_SIZE, strides: tuple[int, ...] = STRIDES) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        grid, stride = build_grid(input_size, strides)
        self.register_buffer("grid", grid)
        self.register_buffer("stride", stride)

    def forward(self, levels: list[torch.Tensor], targets: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """`targets[i]` is `[G_i, 5]`: class, cx, cy, w, h in input pixels."""
        boxes, obj, cls = decode_levels(levels, self.grid, self.stride, self.input_size)
        centers = (self.grid + 0.5) * self.stride  # [A, 2]
        total_iou = boxes.new_zeros(())
        total_cls = boxes.new_zeros(())
        total_obj = boxes.new_zeros(())
        foreground = 0
        for index, target in enumerate(targets):
            obj_target = torch.zeros_like(obj[index])
            if target.shape[0] > 0:
                matched = self._assign(boxes[index].detach(), obj[index].detach(), cls[index].detach(), centers, target)
                if matched is not None:
                    anchor_idx, gt_idx, ious = matched
                    obj_target[anchor_idx] = 1.0
                    gt_boxes = target[gt_idx, 1:5]
                    pred = boxes[index][anchor_idx]
                    iou = box_iou_cxcywh(gt_boxes, pred).diagonal()
                    total_iou = total_iou + (1.0 - iou**2).sum()
                    onehot = functional.one_hot(target[gt_idx, 0].long(), self.num_classes).float()
                    total_cls = total_cls + functional.binary_cross_entropy_with_logits(
                        cls[index][anchor_idx], onehot * ious[:, None], reduction="sum"
                    )
                    foreground += int(anchor_idx.numel())
            total_obj = total_obj + functional.binary_cross_entropy_with_logits(obj[index], obj_target, reduction="sum")
        normaliser = max(foreground, 1)
        loss_iou, loss_obj, loss_cls = total_iou / normaliser, total_obj / normaliser, total_cls / normaliser
        return {"loss": 5.0 * loss_iou + loss_obj + loss_cls, "iou": loss_iou, "obj": loss_obj, "cls": loss_cls}

    def _assign(
        self,
        boxes: torch.Tensor,
        obj: torch.Tensor,
        cls: torch.Tensor,
        centers: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """SimOTA for one image: returns `(anchor indices, ground-truth index of each, IoU of each)`."""
        gt_cls, gt_boxes = target[:, 0].long(), target[:, 1:5]
        gt_min, gt_max = gt_boxes[:, :2] - gt_boxes[:, 2:] / 2, gt_boxes[:, :2] + gt_boxes[:, 2:] / 2
        in_box = ((centers[None] > gt_min[:, None]) & (centers[None] < gt_max[:, None])).all(dim=2)  # [G, A]
        radius = CENTER_RADIUS * self.stride[:, 0]
        in_center = ((centers[None] - gt_boxes[:, None, :2]).abs() < radius[None, :, None]).all(dim=2)
        candidate = in_box | in_center
        fg = candidate.any(dim=0)
        if not bool(fg.any()):
            return None
        fg_idx = fg.nonzero(as_tuple=True)[0]
        both = (in_box & in_center)[:, fg_idx]

        ious = box_iou_cxcywh(gt_boxes, boxes[fg_idx])  # [G, F]
        iou_cost = -torch.log(ious + EPS)
        probabilities = (torch.sigmoid(cls[fg_idx]) * torch.sigmoid(obj[fg_idx])).sqrt()  # [F, C]
        onehot = functional.one_hot(gt_cls, self.num_classes).float()  # [G, C]
        cls_cost = functional.binary_cross_entropy(
            probabilities[None].expand(gt_boxes.shape[0], -1, -1).clamp(EPS, 1 - EPS),
            onehot[:, None, :].expand(-1, fg_idx.numel(), -1),
            reduction="none",
        ).sum(dim=2)
        cost = cls_cost + 3.0 * iou_cost + 1e5 * (~both).float()

        matching = torch.zeros_like(cost, dtype=torch.bool)
        k = min(CANDIDATE_K, fg_idx.numel())
        dynamic_k = torch.topk(ious, k, dim=1).values.sum(dim=1).int().clamp(min=1)
        for gt in range(gt_boxes.shape[0]):
            chosen = torch.topk(cost[gt], int(dynamic_k[gt]), largest=False).indices
            matching[gt, chosen] = True
        multiple = matching.sum(dim=0) > 1
        if bool(multiple.any()):
            best_gt = cost[:, multiple].argmin(dim=0)
            matching[:, multiple] = False
            matching[best_gt, multiple.nonzero(as_tuple=True)[0]] = True
        matched_columns = matching.any(dim=0)
        if not bool(matched_columns.any()):
            return None
        gt_of_anchor = matching[:, matched_columns].float().argmax(dim=0)
        anchor_idx = fg_idx[matched_columns]
        matched_ious = (matching.float() * ious).sum(dim=0)[matched_columns]
        return anchor_idx, gt_of_anchor, matched_ious
