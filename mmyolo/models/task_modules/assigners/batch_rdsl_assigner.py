# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn.functional as F
from mmdet.utils import ConfigType
from torch import Tensor

from mmyolo.registry import TASK_UTILS
from .batch_dsl_assigner import BatchDynamicSoftLabelAssigner

INF = 100000000
EPS = 1.0e-7


def find_inside_rbox(boxes, points, eps=0.01):
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
    valid_mask = (offset_x <= w / 2 - eps) & (offset_x >= - w / 2 + eps) & \
        (offset_y <= h / 2 - eps) & (offset_y >= - h / 2 + eps)
    # (N_points, B, N_boxes)->(B, N_points, N_boxes)
    valid_mask = valid_mask.permute(1, 0, 2)
    return valid_mask


@TASK_UTILS.register_module()
class BatchRotatedDSLAssigner(BatchDynamicSoftLabelAssigner):
    """Computes matching between predictions and ground truth with dynamic soft
    label assignment.

    `BatchDynamicSoftLabelAssigner` for Rotated boxes with format (x,y,w,h,a)
    The main changes are:
    1. The `pred_bboxes` and `gt_bboxes` are rotated boxes with last dim 5.
    2. The bbox centre and find inside box points method is different.
    3. RBboxOverlaps2D doesn't support batch input, use loop instead.

    Args:
        num_classes (int): number of class
        soft_center_radius (float): Radius of the soft center prior.
            Defaults to 3.0.
        topk (int): Select top-k predictions to calculate dynamic k
            best matches for each gt. Defaults to 13.
        iou_weight (float): The scale factor of iou cost. Defaults to 3.0.
        iou_calculator (ConfigType): Config of overlaps Calculator.
            Defaults to dict(type='mmrotate.RBboxOverlaps2D').
    """

    def __init__(
        self,
        num_classes,
        soft_center_radius: float = 3.0,
        topk: int = 13,
        iou_weight: float = 3.0,
        iou_calculator: ConfigType = dict(type='mmrotate.RBboxOverlaps2D')
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            soft_center_radius=soft_center_radius,
            topk=topk,
            iou_weight=iou_weight,
            iou_calculator=iou_calculator)

    @torch.no_grad()
    def forward(self, pred_bboxes: Tensor, pred_scores: Tensor, priors: Tensor,
                gt_labels: Tensor, gt_bboxes: Tensor,
                pad_bbox_flag: Tensor) -> dict:
        num_gt = gt_bboxes.size(1)
        decoded_bboxes = pred_bboxes
        num_bboxes = decoded_bboxes.size(1)
        batch_size = decoded_bboxes.size(0)

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
        valid_mask = find_inside_rbox(gt_bboxes, prior_center)

        gt_center = gt_bboxes[..., :2]

        strides = priors[..., 2]
        distance = (priors[None].unsqueeze(2)[..., :2] -
                    gt_center[:, None, :, :]
                    ).pow(2).sum(-1).sqrt() / strides[None, :, None]

        # prevent overflow
        distance = distance * valid_mask
        soft_center_prior = torch.pow(10, distance - self.soft_center_radius)

        # rbox iou doesn't support batch input, replace with loop
        ious = []
        for box, gt in zip(decoded_bboxes, gt_bboxes):
            iou = self.iou_calculator(box, gt)
            ious.append(iou)

        pairwise_ious = torch.stack(ious, dim=0)
        del ious
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
        cost_matrix = torch.where(valid_mask, cost_matrix, max_pad_value)

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
