# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from mmdet.models.losses.utils import weight_reduce_loss
from mmdet.structures.bbox import HorizontalBoxes

from mmyolo.registry import MODELS


# TODO: unify all code
def bbox_overlaps(pred: torch.Tensor,
                  target: torch.Tensor,
                  iou_mode: str = 'ciou',
                  bbox_format: str = 'xywh',
                  eps: float = 1e-7) -> torch.Tensor:
    r"""Calculate overlap between two set of bboxes.
    `Implementation of paper `Enhancing Geometric Factors into
    Model Learning and Inference for Object Detection and Instance
    Segmentation <https://arxiv.org/abs/2005.03572>`_.
    In the CIoU implementation of YOLOv5 and MMDetection, there is a slight
    difference in the way the alpha parameter is computed.
    mmdet version:
        alpha = (ious > 0.5).float() * v / (1 - ious + v)
    YOLOv5 version:
        alpha = v / (v - ious + (1 + eps)
    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2)
            or (x, y, w, h),shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        iou_mode (str): Options are "ciou".
            Defaults to "ciou".
        bbox_format (str): Options are "xywh" and "xyxy".
            Defaults to "xywh".
        eps (float): Eps to avoid log(0).
    Returns:
        Tensor: shape (n,).
    """
    assert iou_mode in ('ciou', 'giou', 'siou')
    assert bbox_format in ('xyxy', 'xywh')
    if bbox_format == 'xywh':
        pred = HorizontalBoxes.cxcywh_to_xyxy(pred)
        target = HorizontalBoxes.cxcywh_to_xyxy(target)

    bbox1_x1, bbox1_y1 = pred[:, 0], pred[:, 1]
    bbox1_x2, bbox1_y2 = pred[:, 2], pred[:, 3]
    bbox2_x1, bbox2_y1 = target[:, 0], target[:, 1]
    bbox2_x2, bbox2_y2 = target[:, 2], target[:, 3]

    # Overlap
    overlap = (torch.min(bbox1_x2, bbox2_x2) -
               torch.max(bbox1_x1, bbox2_x1)).clamp(0) * \
              (torch.min(bbox1_y2, bbox2_y2) -
               torch.max(bbox1_y1, bbox2_y1)).clamp(0)

    # Union
    w1, h1 = bbox1_x2 - bbox1_x1, bbox1_y2 - bbox1_y1
    w2, h2 = bbox2_x2 - bbox2_x1, bbox2_y2 - bbox2_y1
    union = w1 * h1 + w2 * h2 - overlap + eps

    h1 += eps
    h2 += eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

    enclose_w = enclose_wh[:, 0]
    enclose_h = enclose_wh[:, 1]

    if iou_mode == 'ciou':
        enclose_area = enclose_w**2 + enclose_h**2 + eps

        # rho2(ρ^2): euclidean distance between bbox2(pred) and bbox1(gt),
        # then ** 2
        left_item = ((bbox2_x1 + bbox2_x2) - (bbox1_x1 + bbox1_x2))**2 / 4
        right_item = ((bbox2_y1 + bbox2_y2) - (bbox1_y1 + bbox1_y2))**2 / 4
        rho2 = left_item + right_item

        # Width and height ratio
        wh_ratio = (4 / math.pi**2) * \
                    torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

        with torch.no_grad():
            alpha = wh_ratio / (wh_ratio - ious + (1 + eps))

        # CIoU
        ious = ious - ((rho2 / enclose_area) + (alpha * wh_ratio))

    elif iou_mode == 'giou':
        # GIoU
        convex_area = enclose_w * enclose_h + eps  # convex area
        ious = ious - (convex_area - union) / convex_area

    elif iou_mode == 'siou':
        # SIoU: https://arxiv.org/pdf/2205.12740.pdf
        # Distance cost
        # b_cx_gt-b_cx
        sigma_cw = (bbox2_x1 + bbox2_x2) / 2 - (bbox1_x1 + bbox1_x2) / 2
        # (b_cy_gt-b_cy)
        sigma_ch = (bbox2_y1 + bbox2_y2) / 2 - (bbox1_y1 + bbox1_y2) / 2
        sigma = torch.pow(sigma_cw**2 + sigma_ch**2, 0.5)

        # try to minimize alpha
        sin_alpha = torch.abs(sigma_cw) / sigma
        sin_beta = torch.abs(sigma_ch) / sigma

        threshold = pow(2, 0.5) / 2
        sin_alpha = torch.where(sin_alpha > threshold, sin_beta, sin_alpha)

        # Angle cost
        # 1 - 2 * sin ^ ( 2 * ( arcsin(x) - pi / 4 ) )
        angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)

        # Distance cost
        rho_x = (sigma_cw / enclose_w)**2
        rho_y = (sigma_ch / enclose_h)**2
        gamma = angle_cost - 2
        distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)

        # Shape cost
        omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
        omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
        shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(
            1 - torch.exp(-1 * omiga_h), 4)

        ious = ious - 0.5 * (distance_cost + shape_cost)

    return ious.clamp(min=-1.0, max=1.0)


@MODELS.register_module()
class IoULoss(nn.Module):
    """IoULoss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    Args:
        iou_mode (str): Options are "ciou".
            Defaults to "ciou".
        bbox_format (str): Options are "xywh" and "xyxy".
            Defaults to "xywh".
        eps (float): Eps to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
        return_iou (bool): If True, return loss and iou.
    """

    def __init__(self,
                 iou_mode: str = 'ciou',
                 bbox_format: str = 'xywh',
                 eps: float = 1e-7,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 return_iou: bool = True):
        super().__init__()
        assert bbox_format in ('xywh', 'xyxy')
        assert iou_mode in ('ciou', 'siou', 'giou')
        self.iou_mode = iou_mode
        self.bbox_format = bbox_format
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.return_iou = return_iou

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        avg_factor: Optional[str] = None,
        reduction_override: Optional[Union[str, bool]] = None
    ) -> Tuple[Union[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2)
                or (x, y, w, h),shape (n, 4).
            target (Tensor): Corresponding gt bboxes, shape (n, 4).
            weight (Tensor, optional): Element-wise weights.
            avg_factor (float, optional): Average factor when computing the
                mean of losses.
            reduction_override (str, bool, optional): Same as built-in losses
                of PyTorch. Defaults to None.
        Returns:
            loss or tuple(loss, iou):
        """
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if weight is not None and weight.dim() > 1:
            weight = weight.mean(-1)

        iou = bbox_overlaps(
            pred,
            target,
            iou_mode=self.iou_mode,
            bbox_format=self.bbox_format,
            eps=self.eps)
        loss = self.loss_weight * weight_reduce_loss(1.0 - iou, weight,
                                                     reduction, avg_factor)

        if self.return_iou:
            return loss, iou
        else:
            return loss
