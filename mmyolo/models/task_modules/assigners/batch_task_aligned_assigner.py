# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine import Config
from torch import Tensor

from mmyolo.registry import TASK_UTILS
from .batch_atss_assigner import (select_candidates_in_gts,
                                  select_highest_overlaps)


@TASK_UTILS.register_module()
class BatchTaskAlignedAssigner(nn.Module):
    """This code referenced to
    https://github.com/meituan/YOLOv6/blob/main/yolov6/
    assigners/tal_assigner.py."""

    def __init__(self,
                 topk: int = 13,
                 iou_calculator: Config = dict(type='mmdet.BboxOverlaps2D'),
                 num_classes: int = 80,
                 alpha: float = 1.0,
                 beta: float = 6.0,
                 eps: float = 1e-9):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

        self.iou_calculator = TASK_UTILS.build(iou_calculator)

    @torch.no_grad()
    def forward(self, pred_scores: Tensor, pred_bboxes: Tensor,
                priors_points: Tensor, gt_labels: Tensor, gt_bboxes: Tensor,
                pad_bbox_flag: Tensor):
        """Get assigner result.

        Args:
            pred_scores (Tensor): Scores of predict bbox,
                shape(batch_size, num_priors, num_classes)
            pred_bboxes (Tensor): Predict bboxes,
                shape(batch_size, num_priors, 4)
            priors_points (Tensor): Priors cx cy points,
                shape (num_priors, 2).
            gt_labels (Tensor): Ground true labels,
                shape(batch_size, num_gt, 1)
            gt_bboxes (Tensor): Ground true bboxes,
                shape(batch_size, num_gt, 4)
            pad_bbox_flag (Tensor): Ground truth bbox mask,
                1 means bbox, 0 means no bbox,
                shape(batch_size, num_gt, 1)
        Returns:
            assigned_labels (Tensor): Assigned labels,
                shape(batch_size, num_priors)
            assigned_bboxes (Tensor): Assigned boxes,
                shape(batch_size, num_priors, 4)
            assigned_scores (Tensor): Assigned score,
                shape(batch_size, num_priors, num_classes)
            fg_mask_pre_prior (Tensor): shape(batch_size, num_priors)
        """
        batch_size = pred_scores.size(0)
        num_gt = gt_bboxes.size(1)

        if num_gt == 0:
            return (gt_bboxes.new_full(pred_scores[..., 0].shape,
                                       self.num_classes),
                    gt_bboxes.new_full(pred_bboxes.shape, 0),
                    gt_bboxes.new_full(pred_scores.shape, 0),
                    gt_bboxes.new_full(pred_scores[..., 0].shape, 0))

        pos_mask, alignment_metrics, overlaps = self.get_pos_mask(
            pred_scores, pred_bboxes, gt_labels, gt_bboxes, priors_points,
            pad_bbox_flag, batch_size, num_gt)

        (assigned_gt_inds, fg_mask_pre_prior,
         pos_mask) = select_highest_overlaps(pos_mask, overlaps, num_gt)

        # assigned target
        assigned_labels, assigned_bboxes, assigned_scores = self.get_targets(
            gt_labels, gt_bboxes, assigned_gt_inds, fg_mask_pre_prior,
            batch_size, num_gt)

        # normalize
        alignment_metrics *= pos_mask
        pos_align_metrics = alignment_metrics.max(axis=-1, keepdim=True)[0]
        pos_overlaps = (overlaps * pos_mask).max(axis=-1, keepdim=True)[0]
        norm_align_metric = (
            alignment_metrics * pos_overlaps /
            (pos_align_metrics + self.eps)).max(-2)[0].unsqueeze(-1)
        assigned_scores = assigned_scores * norm_align_metric

        return (assigned_labels, assigned_bboxes, assigned_scores,
                fg_mask_pre_prior.bool())

    def get_pos_mask(self, pred_scores: Tensor, pred_bboxes: Tensor,
                     gt_labels: Tensor, gt_bboxes: Tensor,
                     priors_points: Tensor, pad_bbox_flag: Tensor,
                     batch_size: int,
                     num_gt: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Get possible mask.

        Args:
            pred_scores (Tensor): Scores of predict bbox,
                shape(batch_size, num_priors, num_classes)
            pred_bboxes (Tensor): Predict bboxes,
                shape(batch_size, num_priors, 4)
            priors_points (Tensor): Priors cx cy points,
                shape (num_priors, 2).
            gt_labels (Tensor): Ground true labels,
                shape(batch_size, num_gt, 1)
            gt_bboxes (Tensor): Ground true bboxes,
                shape(batch_size, num_gt, 4)
            pad_bbox_flag (Tensor): Ground truth bbox mask,
                1 means bbox, 0 means no bbox,
                shape(batch_size, num_gt, 1)
            batch_size (int): Batch size.
            num_gt (int): Number of ground true.
        Returns:
            pos_mask (Tensor): Possible mask,
                shape(batch_size, num_gt, num_priors)
            alignment_metrics (Tensor): Alignment metrics,
                shape(batch_size, num_gt, num_priors)
            overlaps (Tensor): Overlaps, shape(batch_size, num_gt, num_priors)
        """

        # Compute alignment metric between all bbox and gt
        alignment_metrics, overlaps = self.get_box_metrics(
            pred_scores, pred_bboxes, gt_labels, gt_bboxes, batch_size, num_gt)
        # get in_gts mask
        mask_in_gts = select_candidates_in_gts(priors_points, gt_bboxes)

        # get topk_metric mask
        mask_topk = self.select_topk_candidates(
            alignment_metrics * mask_in_gts,
            topk_mask=pad_bbox_flag.repeat([1, 1, self.topk]).bool())

        # merge all mask to a final mask
        pos_mask = mask_topk * mask_in_gts * pad_bbox_flag

        return pos_mask, alignment_metrics, overlaps

    def get_box_metrics(self, pred_scores: Tensor, pred_bboxes: Tensor,
                        gt_labels: Tensor, gt_bboxes: Tensor, batch_size: int,
                        num_gt: int) -> Tuple[Tensor, Tensor]:
        """Compute alignment metric between all bbox and gt.

        Args:
            pred_scores (Tensor): Scores of predict bbox,
                shape(batch_size, num_priors, num_classes)
            pred_bboxes (Tensor): Predict bboxes,
                shape(batch_size, num_priors, 4)
            gt_labels (Tensor): Ground true labels,
                shape(batch_size, num_gt, 1)
            gt_bboxes (Tensor): Ground true bboxes,
                shape(batch_size, num_gt, 4)
            batch_size (int): Batch size.
            num_gt (int): Number of ground true.
        Returns:
            alignment_metrics (Tensor): Align metric,
                shape(batch_size, num_gt, num_priors)
            overlaps (Tensor): Overlaps, shape(batch_size, num_gt, num_priors)
        """
        pred_scores = pred_scores.permute(0, 2, 1)
        gt_labels = gt_labels.to(torch.long)
        ind = torch.zeros([2, batch_size, num_gt], dtype=torch.long)
        ind[0] = torch.arange(end=batch_size).view(-1, 1).repeat(1, num_gt)
        ind[1] = gt_labels.squeeze(-1)
        bbox_scores = pred_scores[ind[0], ind[1]]

        overlaps = self.iou_calculator(gt_bboxes, pred_bboxes)
        alignment_metrics = bbox_scores.pow(self.alpha) * overlaps.pow(
            self.beta)

        return alignment_metrics, overlaps

    def select_topk_candidates(self,
                               metrics: Tensor,
                               largest: bool = True,
                               topk_mask: Tensor = None) -> Tensor:
        """Compute alignment metric between all bbox and gt.

        Args:
            metrics (Tensor): Alignment metrics, Scores of predict bbox,
                shape(batch_size, num_gt, num_priors)
            largest (bool): Controls whether to return largest or
                smallest elements.
            topk_mask (Tensor): Topk mask, shape(batch_size, num_gt, self.topk)
        Returns:
            Tensor: Topk candidates mask, shape(batch_size, num_gt, num_priors)
        """
        num_anchors = metrics.shape[-1]
        topk_metrics, topk_idxs = torch.topk(
            metrics, self.topk, axis=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(axis=-1, keepdim=True) >
                         self.eps).tile([1, 1, self.topk])
        topk_idxs = torch.where(topk_mask, topk_idxs,
                                torch.zeros_like(topk_idxs))
        is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(axis=-2)
        is_in_topk = torch.where(is_in_topk > 1, torch.zeros_like(is_in_topk),
                                 is_in_topk)
        return is_in_topk.to(metrics.dtype)

    def get_targets(self, gt_labels: Tensor, gt_bboxes: Tensor,
                    assigned_gt_inds: Tensor, fg_mask_pre_prior: Tensor,
                    batch_size: int,
                    num_gt: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Get assigner info.

        Args:
            gt_labels (Tensor): Ground true labels,
                shape(batch_size, num_gt, 1)
            gt_bboxes (Tensor): Ground true bboxes,
                shape(batch_size, num_gt, 4)
            assigned_gt_inds (Tensor): Assigned ground true indexes,
                shape(batch_size, num_priors)
            fg_mask_pre_prior (Tensor): Force ground true matching mask,
                shape(batch_size, num_priors)
            batch_size (int): Batch size.
            num_gt (int): Number of ground true.
        Returns:
            assigned_labels (Tensor): Assigned labels,
                shape(batch_size, num_priors)
            assigned_bboxes (Tensor): Assigned bboxes,
                shape(batch_size, num_priors)
            assigned_scores (Tensor): Assigned scores,
                shape(batch_size, num_priors)
        """
        # assigned target labels
        batch_ind = torch.arange(
            end=batch_size, dtype=torch.int64, device=gt_labels.device)[...,
                                                                        None]
        assigned_gt_inds = assigned_gt_inds + batch_ind * num_gt
        assigned_labels = gt_labels.long().flatten()[assigned_gt_inds]

        # assigned target boxes
        assigned_bboxes = gt_bboxes.reshape([-1, 4])[assigned_gt_inds]

        # assigned target scores
        assigned_labels[assigned_labels < 0] = 0
        assigned_scores = F.one_hot(assigned_labels, self.num_classes)
        force_gt_scores_mask = fg_mask_pre_prior[:, :, None].repeat(
            1, 1, self.num_classes)
        assigned_scores = torch.where(force_gt_scores_mask > 0,
                                      assigned_scores,
                                      torch.full_like(assigned_scores, 0))

        return assigned_labels, assigned_bboxes, assigned_scores
