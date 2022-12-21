# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmyolo.registry import TASK_UTILS
from mmdet.structures.bbox import BaseBoxes
from mmdet.utils import ConfigType

INF = 100000000
EPS = 1.0e-7


@TASK_UTILS.register_module()
class BatchDynamicSoftLabelAssigner(nn.Module):
    """Computes matching between predictions and ground truth with dynamic soft
    label assignment.

    Args:
        soft_center_radius (float): Radius of the soft center prior.
            Defaults to 3.0.
        topk (int): Select top-k predictions to calculate dynamic k
            best matches for each gt. Defaults to 13.
        iou_weight (float): The scale factor of iou cost. Defaults to 3.0.
        iou_calculator (ConfigType): Config of overlaps Calculator.
            Defaults to dict(type='BboxOverlaps2D').
    """

    def __init__(self,
                 num_classes: int = 80,
                 soft_center_radius: float = 3.0,
                 topk: int = 13,
                 iou_weight: float = 3.0,
                 iou_calculator: ConfigType = dict(type='mmdet.BboxOverlaps2D')) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.soft_center_radius = soft_center_radius
        self.topk = topk
        self.iou_weight = iou_weight
        self.iou_calculator = TASK_UTILS.build(iou_calculator)

    @torch.no_grad()
    def forward(self,
                pred_bboxes: Tensor,
                pred_scores: Tensor,
                priors: Tensor,
                gt_labels: Tensor,
                gt_bboxes: Tensor,
                pad_bbox_flag: Tensor) -> dict:
        num_gt = gt_bboxes.size(1)
        decoded_bboxes = pred_bboxes
        num_bboxes = decoded_bboxes.size(1)
        batch_size = decoded_bboxes.size(0)

        if num_gt == 0 or num_bboxes == 0:
            return {
                'assigned_labels':
                    gt_labels.new_full(pred_scores[..., 0].shape,
                                       self.num_classes, dtype=torch.long),
                'assigned_labels_weights':
                    gt_bboxes.new_full(pred_scores[..., 0].shape, 1),
                'assigned_bboxes':
                    gt_bboxes.new_full(pred_bboxes.shape, 0),
                'assign_metrics':
                    gt_bboxes.new_full(pred_scores[..., 0].shape, 0)
            }

        prior_center = priors[:, :2]
        if isinstance(gt_bboxes, BaseBoxes):
            raise NotImplementedError(f'type of {type(gt_bboxes)} are not implemented !')
        else:
            # Tensor boxes will be treated as horizontal boxes by defaults
            lt_ = prior_center[:, None, None] - gt_bboxes[..., :2]
            rb_ = gt_bboxes[..., 2:] - prior_center[:, None, None]

            deltas = torch.cat([lt_, rb_], dim=-1)
            is_in_gts = deltas.min(dim=-1).values > 0
            is_in_gts = is_in_gts * pad_bbox_flag[..., 0][None]
            is_in_gts = is_in_gts.permute(1, 0, 2)
            valid_mask = is_in_gts.sum(dim=-1) > 0

        # Tensor boxes will be treated as horizontal boxes by defaults
        gt_center = (gt_bboxes[..., :2] + gt_bboxes[..., 2:]) / 2.0

        strides = priors[..., 2]
        distance = (priors[None].unsqueeze(2)[..., :2] - gt_center[:, None, :, :]
                    ).pow(2).sum(-1).sqrt() / strides[None, :, None]

        # prevent overflow
        distance = distance * valid_mask.unsqueeze(-1)
        soft_center_prior = torch.pow(10, distance - self.soft_center_radius)

        pairwise_ious = self.iou_calculator(decoded_bboxes, gt_bboxes)
        iou_cost = -torch.log(pairwise_ious + EPS) * self.iou_weight

        gt_onehot_label = (
            F.one_hot(gt_labels[..., 0].to(torch.int64),
                      pred_scores.shape[-1]).float().unsqueeze(1).repeat(
                1, decoded_bboxes.shape[1], 1, 1))
        valid_pred_scores = pred_scores.unsqueeze(2).repeat(1, 1, num_gt, 1)

        soft_label = gt_onehot_label * pairwise_ious[..., None]
        scale_factor = soft_label - valid_pred_scores.sigmoid()

        # always OOM
        # soft_cls_cost = F.binary_cross_entropy_with_logits(
        #     valid_pred_scores, soft_label,
        #     reduction='none') * scale_factor.abs().pow(2.0)

        soft_cls_costs = []
        for b in range(batch_size):
            soft_cls_cost = F.binary_cross_entropy_with_logits(
                valid_pred_scores[b], soft_label[b],
                reduction='none') * scale_factor[b].abs().pow(2.0)
            soft_cls_costs.append(soft_cls_cost)
        soft_cls_cost = torch.stack(soft_cls_costs, dim=0)

        soft_cls_cost = soft_cls_cost.sum(dim=-1)

        cost_matrix = soft_cls_cost + iou_cost + soft_center_prior

        max_pad_value = torch.ones_like(cost_matrix) * INF
        cost_matrix = torch.where(valid_mask[..., None].repeat(1, 1, num_gt), cost_matrix, max_pad_value)

        matched_pred_ious, matched_gt_inds, fg_mask_inboxes = self.dynamic_k_matching(
            cost_matrix, pairwise_ious, pad_bbox_flag)

        del pairwise_ious, cost_matrix

        batch_index = (fg_mask_inboxes > 0).nonzero(as_tuple=True)[0]

        assigned_labels = gt_labels.new_full(pred_scores[..., 0].shape, self.num_classes)
        assigned_labels[fg_mask_inboxes] = gt_labels[batch_index, matched_gt_inds].squeeze(-1)
        assigned_labels = assigned_labels.long()

        assigned_labels_weights = gt_bboxes.new_full(pred_scores[..., 0].shape, 1)

        assigned_bboxes = gt_bboxes.new_full(pred_bboxes.shape, 0)
        assigned_bboxes[fg_mask_inboxes] = gt_bboxes[batch_index, matched_gt_inds]

        assign_metrics = gt_bboxes.new_full(pred_scores[..., 0].shape, 0)
        assign_metrics[fg_mask_inboxes] = matched_pred_ious

        return dict(assigned_labels=assigned_labels,
                    assigned_labels_weights=assigned_labels_weights,
                    assigned_bboxes=assigned_bboxes,
                    assign_metrics=assign_metrics)

    def dynamic_k_matching(self, cost_matrix: Tensor,
                           pairwise_ious: Tensor,
                           pad_bbox_flag: int) -> Tuple[Tensor, Tensor]:
        """Use IoU and matching cost to calculate the dynamic top-k positive
        targets. Same as SimOTA.

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

        # TODO: Parallel Computation
        for b in range(pad_bbox_flag.shape[0]):
            num_gt = pad_bbox_flag[b, :, 0].sum()
            for gt_idx in range(int(num_gt.item())):
                _, pos_idx = torch.topk(
                    cost_matrix[b, :, gt_idx], k=dynamic_ks[b, gt_idx], largest=False)
                matching_matrix[b, :, gt_idx][pos_idx] = 1
        del topk_ious, dynamic_ks

        prior_match_gt_mask = matching_matrix.sum(2) > 1
        if prior_match_gt_mask.sum() > 0:
            # TODO: to check
            cost_min, cost_argmin = torch.min(
                cost_matrix[prior_match_gt_mask, :], dim=1)
            matching_matrix[prior_match_gt_mask, :] *= 0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1

        # get foreground mask inside box and center prior
        fg_mask_inboxes = matching_matrix.sum(2) > 0
        matched_pred_ious = (matching_matrix * pairwise_ious).sum(2)[fg_mask_inboxes]
        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        return matched_pred_ious, matched_gt_inds, fg_mask_inboxes
