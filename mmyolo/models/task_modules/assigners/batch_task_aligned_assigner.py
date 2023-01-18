# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmyolo.models.losses import bbox_overlaps
from mmyolo.registry import TASK_UTILS
from .utils import (select_candidates_in_gts, select_highest_overlaps,
                    yolov6_iou_calculator)


@TASK_UTILS.register_module()
class BatchTaskAlignedAssigner(nn.Module):
    """This code referenced to
    https://github.com/meituan/YOLOv6/blob/main/yolov6/
    assigners/tal_assigner.py.
    Batch Task aligned assigner base on the paper:
    `TOOD: Task-aligned One-stage Object Detection.
    <https://arxiv.org/abs/2108.07755>`_.
    Assign a corresponding gt bboxes or background to a batch of
    predicted bboxes. Each bbox will be assigned with `0` or a
    positive integer indicating the ground truth index.
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt
    Args:
        num_classes (int): number of class
        topk (int): number of bbox selected in each level
        alpha (float): Hyper-parameters related to alignment_metrics.
            Defaults to 1.0
        beta (float): Hyper-parameters related to alignment_metrics.
            Defaults to 6.
        eps (float): Eps to avoid log(0). Default set to 1e-9
        use_ciou (bool): Whether to use ciou while calculating iou.
            Defaults to False.
    """

    def __init__(self,
                 num_classes: int,
                 topk: int = 13,
                 alpha: float = 1.0,
                 beta: float = 6.0,
                 eps: float = 1e-7,
                 use_ciou: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.use_ciou = use_ciou

    @torch.no_grad()
    def forward(
        self,
        pred_bboxes: Tensor,
        pred_scores: Tensor,
        priors: Tensor,
        gt_labels: Tensor,
        gt_bboxes: Tensor,
        pad_bbox_flag: Tensor,
    ) -> dict:
        """Assign gt to bboxes.

        The assignment is done in following steps
        1. compute alignment metric between all bbox (bbox of all pyramid
           levels) and gt
        2. select top-k bbox as candidates for each gt
        3. limit the positive sample's center in gt (because the anchor-free
           detector only can predict positive distance)
        Args:
            pred_bboxes (Tensor): Predict bboxes,
                shape(batch_size, num_priors, 4)
            pred_scores (Tensor): Scores of predict bboxes,
                shape(batch_size, num_priors, num_classes)
            priors (Tensor): Model priors,  shape (num_priors, 4)
            gt_labels (Tensor): Ground true labels,
                shape(batch_size, num_gt, 1)
            gt_bboxes (Tensor): Ground true bboxes,
                shape(batch_size, num_gt, 4)
            pad_bbox_flag (Tensor): Ground truth bbox mask,
                1 means bbox, 0 means no bbox,
                shape(batch_size, num_gt, 1)
        Returns:
            assigned_result (dict) Assigned result:
                assigned_labels (Tensor): Assigned labels,
                    shape(batch_size, num_priors)
                assigned_bboxes (Tensor): Assigned boxes,
                    shape(batch_size, num_priors, 4)
                assigned_scores (Tensor): Assigned scores,
                    shape(batch_size, num_priors, num_classes)
                fg_mask_pre_prior (Tensor): Force ground truth matching mask,
                    shape(batch_size, num_priors)
        """
        # (num_priors, 4) -> (num_priors, 2)
        priors = priors[:, :2]

        batch_size = pred_scores.size(0)
        num_gt = gt_bboxes.size(1)

        assigned_result = {
            'assigned_labels':
            gt_bboxes.new_full(pred_scores[..., 0].shape, self.num_classes),
            'assigned_bboxes':
            gt_bboxes.new_full(pred_bboxes.shape, 0),
            'assigned_scores':
            gt_bboxes.new_full(pred_scores.shape, 0),
            'fg_mask_pre_prior':
            gt_bboxes.new_full(pred_scores[..., 0].shape, 0)
        }

        if num_gt == 0:
            return assigned_result

        pos_mask, alignment_metrics, overlaps = self.get_pos_mask(
            pred_bboxes, pred_scores, priors, gt_labels, gt_bboxes,
            pad_bbox_flag, batch_size, num_gt)

        (assigned_gt_idxs, fg_mask_pre_prior,
         pos_mask) = select_highest_overlaps(pos_mask, overlaps, num_gt)

        # assigned target
        assigned_labels, assigned_bboxes, assigned_scores = self.get_targets(
            gt_labels, gt_bboxes, assigned_gt_idxs, fg_mask_pre_prior,
            batch_size, num_gt)

        # normalize
        alignment_metrics *= pos_mask
        pos_align_metrics = alignment_metrics.max(axis=-1, keepdim=True)[0]
        pos_overlaps = (overlaps * pos_mask).max(axis=-1, keepdim=True)[0]
        norm_align_metric = (
            alignment_metrics * pos_overlaps /
            (pos_align_metrics + self.eps)).max(-2)[0].unsqueeze(-1)
        assigned_scores = assigned_scores * norm_align_metric

        assigned_result['assigned_labels'] = assigned_labels
        assigned_result['assigned_bboxes'] = assigned_bboxes
        assigned_result['assigned_scores'] = assigned_scores
        assigned_result['fg_mask_pre_prior'] = fg_mask_pre_prior.bool()
        return assigned_result

    def get_pos_mask(self, pred_bboxes: Tensor, pred_scores: Tensor,
                     priors: Tensor, gt_labels: Tensor, gt_bboxes: Tensor,
                     pad_bbox_flag: Tensor, batch_size: int,
                     num_gt: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Get possible mask.

        Args:
            pred_bboxes (Tensor): Predict bboxes,
                shape(batch_size, num_priors, 4)
            pred_scores (Tensor): Scores of predict bbox,
                shape(batch_size, num_priors, num_classes)
            priors (Tensor): Model priors, shape (num_priors, 2)
            gt_labels (Tensor): Ground true labels,
                shape(batch_size, num_gt, 1)
            gt_bboxes (Tensor): Ground true bboxes,
                shape(batch_size, num_gt, 4)
            pad_bbox_flag (Tensor): Ground truth bbox mask,
                1 means bbox, 0 means no bbox,
                shape(batch_size, num_gt, 1)
            batch_size (int): Batch size.
            num_gt (int): Number of ground truth.
        Returns:
            pos_mask (Tensor): Possible mask,
                shape(batch_size, num_gt, num_priors)
            alignment_metrics (Tensor): Alignment metrics,
                shape(batch_size, num_gt, num_priors)
            overlaps (Tensor): Overlaps of gt_bboxes and pred_bboxes,
                shape(batch_size, num_gt, num_priors)
        """

        # Compute alignment metric between all bbox and gt
        alignment_metrics, overlaps = \
            self.get_box_metrics(pred_bboxes, pred_scores, gt_labels,
                                 gt_bboxes, batch_size, num_gt)

        # get is_in_gts mask
        is_in_gts = select_candidates_in_gts(priors, gt_bboxes)

        # get topk_metric mask
        topk_metric = self.select_topk_candidates(
            alignment_metrics * is_in_gts,
            topk_mask=pad_bbox_flag.repeat([1, 1, self.topk]).bool())

        # merge all mask to a final mask
        pos_mask = topk_metric * is_in_gts * pad_bbox_flag

        return pos_mask, alignment_metrics, overlaps

    def get_box_metrics(self, pred_bboxes: Tensor, pred_scores: Tensor,
                        gt_labels: Tensor, gt_bboxes: Tensor, batch_size: int,
                        num_gt: int) -> Tuple[Tensor, Tensor]:
        """Compute alignment metric between all bbox and gt.

        Args:
            pred_bboxes (Tensor): Predict bboxes,
                shape(batch_size, num_priors, 4)
            pred_scores (Tensor): Scores of predict bbox,
                shape(batch_size, num_priors, num_classes)
            gt_labels (Tensor): Ground true labels,
                shape(batch_size, num_gt, 1)
            gt_bboxes (Tensor): Ground true bboxes,
                shape(batch_size, num_gt, 4)
            batch_size (int): Batch size.
            num_gt (int): Number of ground truth.
        Returns:
            alignment_metrics (Tensor): Align metric,
                shape(batch_size, num_gt, num_priors)
            overlaps (Tensor): Overlaps, shape(batch_size, num_gt, num_priors)
        """
        pred_scores = pred_scores.permute(0, 2, 1)
        gt_labels = gt_labels.to(torch.long)
        idx = torch.zeros([2, batch_size, num_gt], dtype=torch.long)
        idx[0] = torch.arange(end=batch_size).view(-1, 1).repeat(1, num_gt)
        idx[1] = gt_labels.squeeze(-1)
        bbox_scores = pred_scores[idx[0], idx[1]]
        # TODO: need to replace the yolov6_iou_calculator function
        if self.use_ciou:
            overlaps = bbox_overlaps(
                pred_bboxes.unsqueeze(1),
                gt_bboxes.unsqueeze(2),
                iou_mode='ciou',
                bbox_format='xyxy').clamp(0)
        else:
            overlaps = yolov6_iou_calculator(gt_bboxes, pred_bboxes)

        alignment_metrics = bbox_scores.pow(self.alpha) * overlaps.pow(
            self.beta)

        return alignment_metrics, overlaps

    def select_topk_candidates(self,
                               alignment_gt_metrics: Tensor,
                               using_largest_topk: bool = True,
                               topk_mask: Optional[Tensor] = None) -> Tensor:
        """Compute alignment metric between all bbox and gt.

        Args:
            alignment_gt_metrics (Tensor): Alignment metric of gt candidates,
                shape(batch_size, num_gt, num_priors)
            using_largest_topk (bool): Controls whether to using largest or
                smallest elements.
            topk_mask (Tensor): Topk mask,
                shape(batch_size, num_gt, self.topk)
        Returns:
            Tensor: Topk candidates mask,
                shape(batch_size, num_gt, num_priors)
        """
        num_priors = alignment_gt_metrics.shape[-1]
        topk_metrics, topk_idxs = torch.topk(
            alignment_gt_metrics,
            self.topk,
            axis=-1,
            largest=using_largest_topk)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(axis=-1, keepdim=True) >
                         self.eps).tile([1, 1, self.topk])
        topk_idxs = torch.where(topk_mask, topk_idxs,
                                torch.zeros_like(topk_idxs))
        is_in_topk = F.one_hot(topk_idxs, num_priors).sum(axis=-2)
        is_in_topk = torch.where(is_in_topk > 1, torch.zeros_like(is_in_topk),
                                 is_in_topk)
        return is_in_topk.to(alignment_gt_metrics.dtype)

    def get_targets(self, gt_labels: Tensor, gt_bboxes: Tensor,
                    assigned_gt_idxs: Tensor, fg_mask_pre_prior: Tensor,
                    batch_size: int,
                    num_gt: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Get assigner info.

        Args:
            gt_labels (Tensor): Ground true labels,
                shape(batch_size, num_gt, 1)
            gt_bboxes (Tensor): Ground true bboxes,
                shape(batch_size, num_gt, 4)
            assigned_gt_idxs (Tensor): Assigned ground truth indexes,
                shape(batch_size, num_priors)
            fg_mask_pre_prior (Tensor): Force ground truth matching mask,
                shape(batch_size, num_priors)
            batch_size (int): Batch size.
            num_gt (int): Number of ground truth.
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
        assigned_gt_idxs = assigned_gt_idxs + batch_ind * num_gt
        assigned_labels = gt_labels.long().flatten()[assigned_gt_idxs]

        # assigned target boxes
        assigned_bboxes = gt_bboxes.reshape([-1, 4])[assigned_gt_idxs]

        # assigned target scores
        assigned_labels[assigned_labels < 0] = 0
        assigned_scores = F.one_hot(assigned_labels, self.num_classes)
        force_gt_scores_mask = fg_mask_pre_prior[:, :, None].repeat(
            1, 1, self.num_classes)
        assigned_scores = torch.where(force_gt_scores_mask > 0,
                                      assigned_scores,
                                      torch.full_like(assigned_scores, 0))

        return assigned_labels, assigned_bboxes, assigned_scores
