# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.utils import ConfigType
from torch import Tensor

from mmyolo.registry import TASK_UTILS
from .utils import (select_candidates_in_gts, select_highest_overlaps,
                    yolov6_iou_calculator)


def bbox_center_distance(bboxes: Tensor,
                         priors: Tensor) -> Tuple[Tensor, Tensor]:
    """Compute the center distance between bboxes and priors.

    Args:
        bboxes (Tensor): Shape (n, 4) for bbox, "xyxy" format.
        priors (Tensor): Shape (num_priors, 4) for priors, "xyxy" format.

    Returns:
        distances (Tensor): Center distances between bboxes and priors,
            shape (num_priors, n).
        priors_points (Tensor): Priors cx cy points,
            shape (num_priors, 2).
    """
    bbox_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
    bbox_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
    bbox_points = torch.stack((bbox_cx, bbox_cy), dim=1)

    priors_cx = (priors[:, 0] + priors[:, 2]) / 2.0
    priors_cy = (priors[:, 1] + priors[:, 3]) / 2.0
    priors_points = torch.stack((priors_cx, priors_cy), dim=1)

    distances = (bbox_points[:, None, :] -
                 priors_points[None, :, :]).pow(2).sum(-1).sqrt()

    return distances, priors_points


@TASK_UTILS.register_module()
class BatchATSSAssigner(nn.Module):
    """Assign a batch of corresponding gt bboxes or background to each prior.

    This code is based on
    https://github.com/meituan/YOLOv6/blob/main/yolov6/assigners/atss_assigner.py

    Each proposal will be assigned with `0` or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        num_classes (int): number of class
        iou_calculator (:obj:`ConfigDict` or dict): Config dict for iou
            calculator. Defaults to ``dict(type='BboxOverlaps2D')``
        topk (int): number of priors selected in each level
    """

    def __init__(
            self,
            num_classes: int,
            iou_calculator: ConfigType = dict(type='mmdet.BboxOverlaps2D'),
            topk: int = 9):
        super().__init__()
        self.num_classes = num_classes
        self.iou_calculator = TASK_UTILS.build(iou_calculator)
        self.topk = topk

    @torch.no_grad()
    def forward(self, pred_bboxes: Tensor, priors: Tensor,
                num_level_priors: List, gt_labels: Tensor, gt_bboxes: Tensor,
                pad_bbox_flag: Tensor) -> dict:
        """Assign gt to priors.

        The assignment is done in following steps

        1. compute iou between all prior (prior of all pyramid levels) and gt
        2. compute center distance between all prior and gt
        3. on each pyramid level, for each gt, select k prior whose center
           are closest to the gt center, so we total select k*l prior as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as positive
        6. limit the positive sample's center in gt

        Args:
            pred_bboxes (Tensor): Predicted bounding boxes,
                shape(batch_size, num_priors, 4)
            priors (Tensor): Model priors with stride, shape(num_priors, 4)
            num_level_priors (List): Number of bboxes in each level, len(3)
            gt_labels (Tensor): Ground truth label,
                shape(batch_size, num_gt, 1)
            gt_bboxes (Tensor): Ground truth bbox,
                shape(batch_size, num_gt, 4)
            pad_bbox_flag (Tensor): Ground truth bbox mask,
                1 means bbox, 0 means no bbox,
                shape(batch_size, num_gt, 1)
        Returns:
            assigned_result (dict): Assigned result
                'assigned_labels' (Tensor): shape(batch_size, num_gt)
                'assigned_bboxes' (Tensor): shape(batch_size, num_gt, 4)
                'assigned_scores' (Tensor):
                    shape(batch_size, num_gt, number_classes)
                'fg_mask_pre_prior' (Tensor): shape(bs, num_gt)
        """
        # generate priors
        cell_half_size = priors[:, 2:] * 2.5
        priors_gen = torch.zeros_like(priors)
        priors_gen[:, :2] = priors[:, :2] - cell_half_size
        priors_gen[:, 2:] = priors[:, :2] + cell_half_size
        priors = priors_gen

        batch_size = gt_bboxes.size(0)
        num_gt, num_priors = gt_bboxes.size(1), priors.size(0)

        assigned_result = {
            'assigned_labels':
            gt_bboxes.new_full([batch_size, num_priors], self.num_classes),
            'assigned_bboxes':
            gt_bboxes.new_full([batch_size, num_priors, 4], 0),
            'assigned_scores':
            gt_bboxes.new_full([batch_size, num_priors, self.num_classes], 0),
            'fg_mask_pre_prior':
            gt_bboxes.new_full([batch_size, num_priors], 0)
        }

        if num_gt == 0:
            return assigned_result

        # compute iou between all prior (prior of all pyramid levels) and gt
        overlaps = self.iou_calculator(gt_bboxes.reshape([-1, 4]), priors)
        overlaps = overlaps.reshape([batch_size, -1, num_priors])

        # compute center distance between all prior and gt
        distances, priors_points = bbox_center_distance(
            gt_bboxes.reshape([-1, 4]), priors)
        distances = distances.reshape([batch_size, -1, num_priors])

        # Selecting candidates based on the center distance
        is_in_candidate, candidate_idxs = self.select_topk_candidates(
            distances, num_level_priors, pad_bbox_flag)

        # get corresponding iou for the these candidates, and compute the
        # mean and std, set mean + std as the iou threshold
        overlaps_thr_per_gt, iou_candidates = self.threshold_calculator(
            is_in_candidate, candidate_idxs, overlaps, num_priors, batch_size,
            num_gt)

        # select candidates iou >= threshold as positive
        is_pos = torch.where(
            iou_candidates > overlaps_thr_per_gt.repeat([1, 1, num_priors]),
            is_in_candidate, torch.zeros_like(is_in_candidate))

        is_in_gts = select_candidates_in_gts(priors_points, gt_bboxes)
        pos_mask = is_pos * is_in_gts * pad_bbox_flag

        # if an anchor box is assigned to multiple gts,
        # the one with the highest IoU will be selected.
        gt_idx_pre_prior, fg_mask_pre_prior, pos_mask = \
            select_highest_overlaps(pos_mask, overlaps, num_gt)

        # assigned target
        assigned_labels, assigned_bboxes, assigned_scores = self.get_targets(
            gt_labels, gt_bboxes, gt_idx_pre_prior, fg_mask_pre_prior,
            num_priors, batch_size, num_gt)

        # soft label with iou
        if pred_bboxes is not None:
            ious = yolov6_iou_calculator(gt_bboxes, pred_bboxes) * pos_mask
            ious = ious.max(axis=-2)[0].unsqueeze(-1)
            assigned_scores *= ious

        assigned_result['assigned_labels'] = assigned_labels.long()
        assigned_result['assigned_bboxes'] = assigned_bboxes
        assigned_result['assigned_scores'] = assigned_scores
        assigned_result['fg_mask_pre_prior'] = fg_mask_pre_prior.bool()
        return assigned_result

    def select_topk_candidates(self, distances: Tensor,
                               num_level_priors: List[int],
                               pad_bbox_flag: Tensor) -> Tuple[Tensor, Tensor]:
        """Selecting candidates based on the center distance.

        Args:
            distances (Tensor): Distance between all bbox and gt,
                shape(batch_size, num_gt, num_priors)
            num_level_priors (List[int]): Number of bboxes in each level,
                len(3)
            pad_bbox_flag (Tensor): Ground truth bbox mask,
                shape(batch_size, num_gt, 1)

        Return:
            is_in_candidate_list (Tensor): Flag show that each level have
                topk candidates or not,  shape(batch_size, num_gt, num_priors)
            candidate_idxs (Tensor): Candidates index,
                shape(batch_size, num_gt, num_gt)
        """
        is_in_candidate_list = []
        candidate_idxs = []
        start_idx = 0

        distances_dtype = distances.dtype
        distances = torch.split(distances, num_level_priors, dim=-1)
        pad_bbox_flag = pad_bbox_flag.repeat(1, 1, self.topk).bool()

        for distances_per_level, priors_per_level in zip(
                distances, num_level_priors):
            # on each pyramid level, for each gt,
            # select k bbox whose center are closest to the gt center
            end_index = start_idx + priors_per_level
            selected_k = min(self.topk, priors_per_level)

            _, topk_idxs_per_level = distances_per_level.topk(
                selected_k, dim=-1, largest=False)
            candidate_idxs.append(topk_idxs_per_level + start_idx)

            topk_idxs_per_level = torch.where(
                pad_bbox_flag, topk_idxs_per_level,
                torch.zeros_like(topk_idxs_per_level))

            is_in_candidate = F.one_hot(topk_idxs_per_level,
                                        priors_per_level).sum(dim=-2)
            is_in_candidate = torch.where(is_in_candidate > 1,
                                          torch.zeros_like(is_in_candidate),
                                          is_in_candidate)
            is_in_candidate_list.append(is_in_candidate.to(distances_dtype))

            start_idx = end_index

        is_in_candidate_list = torch.cat(is_in_candidate_list, dim=-1)
        candidate_idxs = torch.cat(candidate_idxs, dim=-1)

        return is_in_candidate_list, candidate_idxs

    @staticmethod
    def threshold_calculator(is_in_candidate: List, candidate_idxs: Tensor,
                             overlaps: Tensor, num_priors: int,
                             batch_size: int,
                             num_gt: int) -> Tuple[Tensor, Tensor]:
        """Get corresponding iou for the these candidates, and compute the mean
        and std, set mean + std as the iou threshold.

        Args:
            is_in_candidate (Tensor): Flag show that each level have
                topk candidates or not, shape(batch_size, num_gt, num_priors).
            candidate_idxs (Tensor): Candidates index,
                shape(batch_size, num_gt, num_gt)
            overlaps (Tensor): Overlaps area,
                shape(batch_size, num_gt, num_priors).
            num_priors (int): Number of priors.
            batch_size (int): Batch size.
            num_gt (int): Number of ground truth.

        Return:
            overlaps_thr_per_gt (Tensor): Overlap threshold of
                per ground truth, shape(batch_size, num_gt, 1).
            candidate_overlaps (Tensor): Candidate overlaps,
                shape(batch_size, num_gt, num_priors).
        """

        batch_size_num_gt = batch_size * num_gt
        candidate_overlaps = torch.where(is_in_candidate > 0, overlaps,
                                         torch.zeros_like(overlaps))
        candidate_idxs = candidate_idxs.reshape([batch_size_num_gt, -1])

        assist_indexes = num_priors * torch.arange(
            batch_size_num_gt, device=candidate_idxs.device)
        assist_indexes = assist_indexes[:, None]
        flatten_indexes = candidate_idxs + assist_indexes

        candidate_overlaps_reshape = candidate_overlaps.reshape(
            -1)[flatten_indexes]
        candidate_overlaps_reshape = candidate_overlaps_reshape.reshape(
            [batch_size, num_gt, -1])

        overlaps_mean_per_gt = candidate_overlaps_reshape.mean(
            axis=-1, keepdim=True)
        overlaps_std_per_gt = candidate_overlaps_reshape.std(
            axis=-1, keepdim=True)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        return overlaps_thr_per_gt, candidate_overlaps

    def get_targets(self, gt_labels: Tensor, gt_bboxes: Tensor,
                    assigned_gt_inds: Tensor, fg_mask_pre_prior: Tensor,
                    num_priors: int, batch_size: int,
                    num_gt: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Get target info.

        Args:
            gt_labels (Tensor): Ground true labels,
                shape(batch_size, num_gt, 1)
            gt_bboxes (Tensor): Ground true bboxes,
                shape(batch_size, num_gt, 4)
            assigned_gt_inds (Tensor): Assigned ground truth indexes,
                shape(batch_size, num_priors)
            fg_mask_pre_prior (Tensor): Force ground truth matching mask,
                shape(batch_size, num_priors)
            num_priors (int): Number of priors.
            batch_size (int): Batch size.
            num_gt (int): Number of ground truth.

        Return:
            assigned_labels (Tensor): Assigned labels,
                shape(batch_size, num_priors)
            assigned_bboxes (Tensor): Assigned bboxes,
                shape(batch_size, num_priors)
            assigned_scores (Tensor): Assigned scores,
                shape(batch_size, num_priors)
        """

        # assigned target labels
        batch_index = torch.arange(
            batch_size, dtype=gt_labels.dtype, device=gt_labels.device)
        batch_index = batch_index[..., None]
        assigned_gt_inds = (assigned_gt_inds + batch_index * num_gt).long()
        assigned_labels = gt_labels.flatten()[assigned_gt_inds.flatten()]
        assigned_labels = assigned_labels.reshape([batch_size, num_priors])
        assigned_labels = torch.where(
            fg_mask_pre_prior > 0, assigned_labels,
            torch.full_like(assigned_labels, self.num_classes))

        # assigned target boxes
        assigned_bboxes = gt_bboxes.reshape([-1,
                                             4])[assigned_gt_inds.flatten()]
        assigned_bboxes = assigned_bboxes.reshape([batch_size, num_priors, 4])

        # assigned target scores
        assigned_scores = F.one_hot(assigned_labels.long(),
                                    self.num_classes + 1).float()
        assigned_scores = assigned_scores[:, :, :self.num_classes]

        return assigned_labels, assigned_bboxes, assigned_scores
