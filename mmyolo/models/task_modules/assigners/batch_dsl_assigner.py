# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.structures.bbox import BaseBoxes
from mmdet.utils import ConfigType
from torch import Tensor

from mmyolo.registry import TASK_UTILS

INF = 100000000
EPS = 1.0e-7


def find_inside_points(boxes: Tensor,
                       points: Tensor,
                       box_dim: int = 4,
                       eps: float = 0.01) -> Tensor:
    """Find inside box points in batches. Boxes dimension must be 3.

    Args:
        boxes (Tensor): Boxes tensor. Must be batch input.
            Has shape of (batch_size, n_boxes, box_dim).
        points (Tensor): Points coordinates. Has shape of (n_points, 2).
        box_dim (int): The dimension of box. 4 means horizontal box and
            5 means rotated box. Defaults to 4.
        eps (float): Make sure the points are inside not on the boundary.
            Only use in rotated boxes. Defaults to 0.01.

    Returns:
        Tensor: A BoolTensor indicating whether a point is inside
        boxes. The index has shape of (n_points, batch_size, n_boxes).
    """
    if box_dim == 4:
        # Horizontal Boxes
        lt_ = points[:, None, None] - boxes[..., :2]
        rb_ = boxes[..., 2:] - points[:, None, None]

        deltas = torch.cat([lt_, rb_], dim=-1)
        is_in_gts = deltas.min(dim=-1).values > 0

    elif box_dim == 5:
        # Rotated Boxes
        points = points[:, None, None]
        ctrs, wh, t = torch.split(boxes, [2, 2, 1], dim=-1)
        cos_value, sin_value = torch.cos(t), torch.sin(t)
        matrix = torch.cat([cos_value, sin_value, -sin_value, cos_value],
                           dim=-1).reshape(*boxes.shape[:-1], 2, 2)

        offset = points - ctrs
        offset = torch.matmul(matrix, offset[..., None])
        offset = offset.squeeze(-1)
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        w, h = wh[..., 0], wh[..., 1]
        is_in_gts = (offset_x <= w / 2 - eps) & (offset_x >= - w / 2 + eps) & \
                    (offset_y <= h / 2 - eps) & (offset_y >= - h / 2 + eps)
    else:
        raise NotImplementedError(f'Unsupport box_dim:{box_dim}')

    return is_in_gts


def get_box_center(boxes: Tensor, box_dim: int = 4) -> Tensor:
    """Return a tensor representing the centers of boxes.

    Args:
        boxes (Tensor): Boxes tensor. Has shape of (b, n, box_dim)
        box_dim (int): The dimension of box. 4 means horizontal box and
            5 means rotated box. Defaults to 4.

    Returns:
        Tensor: Centers have shape of (b, n, 2)
    """
    if box_dim == 4:
        # Horizontal Boxes, (x1, y1, x2, y2)
        return (boxes[..., :2] + boxes[..., 2:]) / 2.0
    elif box_dim == 5:
        # Rotated Boxes, (x, y, w, h, a)
        return boxes[..., :2]
    else:
        raise NotImplementedError(f'Unsupported box_dim:{box_dim}')


@TASK_UTILS.register_module()
class BatchDynamicSoftLabelAssigner(nn.Module):
    """Computes matching between predictions and ground truth with dynamic soft
    label assignment.

    Args:
        num_classes (int): number of class
        soft_center_radius (float): Radius of the soft center prior.
            Defaults to 3.0.
        topk (int): Select top-k predictions to calculate dynamic k
            best matches for each gt. Defaults to 13.
        iou_weight (float): The scale factor of iou cost. Defaults to 3.0.
        iou_calculator (ConfigType): Config of overlaps Calculator.
            Defaults to dict(type='BboxOverlaps2D').
        batch_iou (bool): Use batch input when calculate IoU.
            If set to False use loop instead. Defaults to True.
    """

    def __init__(
        self,
        num_classes,
        soft_center_radius: float = 3.0,
        topk: int = 13,
        iou_weight: float = 3.0,
        iou_calculator: ConfigType = dict(type='mmdet.BboxOverlaps2D'),
        batch_iou: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.soft_center_radius = soft_center_radius
        self.topk = topk
        self.iou_weight = iou_weight
        self.iou_calculator = TASK_UTILS.build(iou_calculator)
        self.batch_iou = batch_iou

    @torch.no_grad()
    def forward(self, pred_bboxes: Tensor, pred_scores: Tensor, priors: Tensor,
                gt_labels: Tensor, gt_bboxes: Tensor,
                pad_bbox_flag: Tensor) -> dict:
        num_gt = gt_bboxes.size(1)
        decoded_bboxes = pred_bboxes
        batch_size, num_bboxes, box_dim = decoded_bboxes.size()

        if num_gt == 0 or num_bboxes == 0:
            return {
                'assigned_labels':
                gt_labels.new_full(
                    pred_scores[..., 0].shape,
                    self.num_classes,
                    dtype=torch.long),
                'assigned_labels_weights':
                gt_bboxes.new_full(pred_scores[..., 0].shape, 1),
                'assigned_bboxes':
                gt_bboxes.new_full(pred_bboxes.shape, 0),
                'assign_metrics':
                gt_bboxes.new_full(pred_scores[..., 0].shape, 0)
            }

        prior_center = priors[:, :2]
        if isinstance(gt_bboxes, BaseBoxes):
            raise NotImplementedError(
                f'type of {type(gt_bboxes)} are not implemented !')
        else:
            is_in_gts = find_inside_points(gt_bboxes, prior_center, box_dim)

        # (N_points, B, N_boxes)
        is_in_gts = is_in_gts * pad_bbox_flag[..., 0][None]
        # (N_points, B, N_boxes) -> (B, N_points, N_boxes)
        is_in_gts = is_in_gts.permute(1, 0, 2)
        # (B, N_points)
        valid_mask = is_in_gts.sum(dim=-1) > 0

        gt_center = get_box_center(gt_bboxes, box_dim)

        strides = priors[..., 2]
        distance = (priors[None].unsqueeze(2)[..., :2] -
                    gt_center[:, None, :, :]
                    ).pow(2).sum(-1).sqrt() / strides[None, :, None]

        # prevent overflow
        distance = distance * valid_mask.unsqueeze(-1)
        soft_center_prior = torch.pow(10, distance - self.soft_center_radius)

        if self.batch_iou:
            pairwise_ious = self.iou_calculator(decoded_bboxes, gt_bboxes)
        else:
            ious = []
            for box, gt in zip(decoded_bboxes, gt_bboxes):
                iou = self.iou_calculator(box, gt)
                ious.append(iou)
            pairwise_ious = torch.stack(ious, dim=0)

        iou_cost = -torch.log(pairwise_ious + EPS) * self.iou_weight

        # select the predicted scores corresponded to the gt_labels
        pairwise_pred_scores = pred_scores.permute(0, 2, 1)
        idx = torch.zeros([2, batch_size, num_gt], dtype=torch.long)
        idx[0] = torch.arange(end=batch_size).view(-1, 1).repeat(1, num_gt)
        idx[1] = gt_labels.long().squeeze(-1)
        pairwise_pred_scores = pairwise_pred_scores[idx[0],
                                                    idx[1]].permute(0, 2, 1)
        # classification cost
        scale_factor = pairwise_ious - pairwise_pred_scores.sigmoid()
        pairwise_cls_cost = F.binary_cross_entropy_with_logits(
            pairwise_pred_scores, pairwise_ious,
            reduction='none') * scale_factor.abs().pow(2.0)

        cost_matrix = pairwise_cls_cost + iou_cost + soft_center_prior

        max_pad_value = torch.ones_like(cost_matrix) * INF
        cost_matrix = torch.where(valid_mask[..., None].repeat(1, 1, num_gt),
                                  cost_matrix, max_pad_value)

        (matched_pred_ious, matched_gt_inds,
         fg_mask_inboxes) = self.dynamic_k_matching(cost_matrix, pairwise_ious,
                                                    pad_bbox_flag)

        del pairwise_ious, cost_matrix

        batch_index = (fg_mask_inboxes > 0).nonzero(as_tuple=True)[0]

        assigned_labels = gt_labels.new_full(pred_scores[..., 0].shape,
                                             self.num_classes)
        assigned_labels[fg_mask_inboxes] = gt_labels[
            batch_index, matched_gt_inds].squeeze(-1)
        assigned_labels = assigned_labels.long()

        assigned_labels_weights = gt_bboxes.new_full(pred_scores[..., 0].shape,
                                                     1)

        assigned_bboxes = gt_bboxes.new_full(pred_bboxes.shape, 0)
        assigned_bboxes[fg_mask_inboxes] = gt_bboxes[batch_index,
                                                     matched_gt_inds]

        assign_metrics = gt_bboxes.new_full(pred_scores[..., 0].shape, 0)
        assign_metrics[fg_mask_inboxes] = matched_pred_ious

        return dict(
            assigned_labels=assigned_labels,
            assigned_labels_weights=assigned_labels_weights,
            assigned_bboxes=assigned_bboxes,
            assign_metrics=assign_metrics)

    def dynamic_k_matching(
            self, cost_matrix: Tensor, pairwise_ious: Tensor,
            pad_bbox_flag: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Use IoU and matching cost to calculate the dynamic top-k positive
        targets.

        Args:
            cost_matrix (Tensor): Cost matrix.
            pairwise_ious (Tensor): Pairwise iou matrix.
            num_gt (int): Number of gt.
            valid_mask (Tensor): Mask for valid bboxes.
        Returns:
            tuple: matched ious and gt indexes.
        """
        matching_matrix = torch.zeros_like(cost_matrix, dtype=torch.uint8)
        # select candidate topk ious for dynamic-k calculation
        candidate_topk = min(self.topk, pairwise_ious.size(1))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=1)
        # calculate dynamic k for each gt
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)

        num_gts = pad_bbox_flag.sum((1, 2)).int()
        # sorting the batch cost matirx is faster than topk
        _, sorted_indices = torch.sort(cost_matrix, dim=1)
        for b in range(pad_bbox_flag.shape[0]):
            for gt_idx in range(num_gts[b]):
                topk_ids = sorted_indices[b, :dynamic_ks[b, gt_idx], gt_idx]
                matching_matrix[b, :, gt_idx][topk_ids] = 1

        del topk_ious, dynamic_ks

        prior_match_gt_mask = matching_matrix.sum(2) > 1
        if prior_match_gt_mask.sum() > 0:
            cost_min, cost_argmin = torch.min(
                cost_matrix[prior_match_gt_mask, :], dim=1)
            matching_matrix[prior_match_gt_mask, :] *= 0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1

        # get foreground mask inside box and center prior
        fg_mask_inboxes = matching_matrix.sum(2) > 0
        matched_pred_ious = (matching_matrix *
                             pairwise_ious).sum(2)[fg_mask_inboxes]
        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        return matched_pred_ious, matched_gt_inds, fg_mask_inboxes
