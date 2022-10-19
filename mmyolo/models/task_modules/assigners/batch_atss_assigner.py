# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.utils import ConfigType
from torch import Tensor

from mmyolo.registry import TASK_UTILS


def bbox_center_distance(bboxes: Tensor, anchor: Tensor):
    """Compute the center distance between bboxes and priors.

    Args:
        bboxes (Tensor): Shape (n, 4) for , "xyxy" format.
        anchor (Tensor): Shape (n, 4) for priors, "xyxy" format.

    Returns:
        Tensor: Center distances between bboxes and priors.
    """
    bbox_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
    bbox_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
    bbox_points = torch.stack((bbox_cx, bbox_cy), dim=1)

    anchor_cx = (anchor[:, 0] + anchor[:, 2]) / 2.0
    anchor_cy = (anchor[:, 1] + anchor[:, 3]) / 2.0
    anchor_points = torch.stack((anchor_cx, anchor_cy), dim=1)

    distances = (bbox_points[:, None, :] -
                 anchor_points[None, :, :]).pow(2).sum(-1).sqrt()

    return distances, anchor_points


def select_candidates_in_gts(xy_centers: Tensor,
                             gt_bboxes: Tensor,
                             eps: float = 1e-9) -> Tensor:
    """select the positive anchors' center in gt.

    Args:
        xy_centers (Tensor): shape(bs*n_max_boxes, num_total_anchors, 4)
        gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
        eps (float):
    Return:
        (Tensor): shape(bs, n_max_boxes, num_total_anchors)
    """
    anchors_number = xy_centers.size(0)
    batch_size, max_boxes, _ = gt_bboxes.size()
    gt_bboxes = gt_bboxes.reshape([-1, 4])

    xy_centers = xy_centers.unsqueeze(0).repeat(batch_size * max_boxes, 1, 1)
    gt_bboxes_lt = gt_bboxes[:, 0:2].unsqueeze(1).repeat(1, anchors_number, 1)
    gt_bboxes_rb = gt_bboxes[:, 2:4].unsqueeze(1).repeat(1, anchors_number, 1)

    bbox_deltas = torch.cat(
        [xy_centers - gt_bboxes_lt, gt_bboxes_rb - xy_centers], dim=-1)
    bbox_deltas = bbox_deltas.reshape(
        [batch_size, max_boxes, anchors_number, -1])

    return (bbox_deltas.min(axis=-1)[0] > eps).to(gt_bboxes.dtype)


def select_highest_overlaps(mask_pos: Tensor, overlaps: Tensor,
                            max_boxes: int) -> Tuple[Tensor, Tensor, Tensor]:
    """If an anchor box is assigned to multiple gts, the one with the highest
    iou will be selected.

    Args:
        mask_pos (Tensor): shape(bs, n_max_boxes, num_total_anchors)
        overlaps (Tensor): shape(bs, n_max_boxes, num_total_anchors)
        max_boxes (int):
    Return:
        target_gt_index (Tensor): shape(bs, num_total_anchors)
        fg_mask (Tensor): shape(bs, num_total_anchors)
        mask_pos (Tensor): shape(bs, n_max_boxes, num_total_anchors)
    """
    fg_mask = mask_pos.sum(axis=-2)

    if fg_mask.max() > 1:
        mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, max_boxes, 1])
        max_overlaps_index = overlaps.argmax(axis=1)
        is_max_overlaps = F.one_hot(max_overlaps_index, max_boxes)
        is_max_overlaps = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)

        mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)
        fg_mask = mask_pos.sum(axis=-2)

    target_gt_index = mask_pos.argmax(axis=-2)
    return target_gt_index, fg_mask, mask_pos


def iou_calculator(bbox1: Tensor, bbox2: Tensor, eps: float = 1e-9) -> Tensor:
    """Calculate iou for batch.

    Args:
        bbox1 (Tensor): shape(bs, n_max_boxes, 1, 4)
        bbox2 (Tensor): shape(bs, 1, num_total_anchors, 4)
        eps (float):
    Return:
        (Tensor): shape(bs, n_max_boxes, num_total_anchors)
    """
    bbox1 = bbox1.unsqueeze(2)  # [N, M1, 4] -> [N, M1, 1, 4]
    bbox2 = bbox2.unsqueeze(1)  # [N, M2, 4] -> [N, 1, M2, 4]

    # calculate xy info of predict and gt bbox
    pred_x1y1, pred_x2y2 = bbox1[:, :, :, 0:2], bbox1[:, :, :, 2:4]
    gt_x1y1, gt_x2y2 = bbox2[:, :, :, 0:2], bbox2[:, :, :, 2:4]

    # calculate overlap area
    overlap = (torch.maximum(pred_x1y1, gt_x1y1) -
               torch.minimum(pred_x2y2, gt_x2y2)).clip(0).prod(-1)

    # calculate bbox area
    pred_area = (pred_x2y2 - pred_x1y1).clip(0).prod(-1)
    gt_area = (gt_x2y2 - gt_x1y1).clip(0).prod(-1)

    union = pred_area + gt_area - overlap + eps

    return overlap / union


@TASK_UTILS.register_module()
class BatchATSSAssigner(nn.Module):
    """Adaptive Training Sample Selection Assigner."""

    def __init__(
            self,
            topk: int = 9,
            iou2d_calculator: ConfigType = dict(type='mmdet.BboxOverlaps2D'),
            num_classes: int = 80):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_index = num_classes
        self.iou2d_calculator = TASK_UTILS.build(iou2d_calculator)

    @torch.no_grad()
    def forward(
            self, anchor_bboxes: torch.Tensor, n_level_bboxes: List,
            gt_labels: torch.Tensor, gt_bboxes: torch.Tensor,
            mask_gt: torch.Tensor,
            pd_bboxes: torch.Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""This code is based on
            https://github.com/fcjian/TOOD/blob/master/mmdet/core/bbox/assigners/atss_assigner.py

        Args:
            anchor_bboxes (Tensor): shape(num_total_anchors, 4)
            n_level_bboxes (List): len(3)
            gt_labels (Tensor): shape(batch_size, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(batch_size, n_max_boxes, 4)
            mask_gt (Tensor): shape(batch_size, n_max_boxes, 1)
            pd_bboxes (Tensor): shape(batch_size, n_max_boxes, 4)
        Returns:
            target_labels (Tensor): shape(batch_size, num_total_anchors)
            target_bboxes (Tensor): shape(batch_size, num_total_anchors, 4)
            target_scores (Tensor): shape(batch_size, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
        """
        total_anchors = anchor_bboxes.size(0)
        batch_size = gt_bboxes.size(0)
        max_boxes = gt_bboxes.size(1)

        if max_boxes == 0:
            device = gt_bboxes.device
            return (torch.full([batch_size, total_anchors],
                               self.bg_index).to(device),
                    torch.zeros([batch_size, total_anchors, 4]).to(device),
                    torch.zeros([batch_size, total_anchors,
                                 self.num_classes]).to(device),
                    torch.zeros([batch_size, total_anchors]).to(device))

        # calculate overlaps
        overlaps = self.iou2d_calculator(
            gt_bboxes.reshape([-1, 4]), anchor_bboxes)
        overlaps = overlaps.reshape([batch_size, -1, total_anchors])

        # calculate bbox center distance and anchor points
        distances, anchor_points = bbox_center_distance(
            gt_bboxes.reshape([-1, 4]), anchor_bboxes)
        distances = distances.reshape([batch_size, -1, total_anchors])

        # filter
        is_in_candidate, candidate_indexes = self.select_topk_candidates(
            distances, n_level_bboxes, mask_gt)
        overlaps_thr_per_gt, iou_candidates = self.threshold_calculator(
            is_in_candidate, candidate_indexes, overlaps, total_anchors,
            batch_size, max_boxes)

        # select candidates iou >= threshold as positive
        is_pos = torch.where(
            iou_candidates > overlaps_thr_per_gt.repeat([1, 1, total_anchors]),
            is_in_candidate, torch.zeros_like(is_in_candidate))

        is_in_gts = select_candidates_in_gts(anchor_points, gt_bboxes)
        mask_pos = is_pos * is_in_gts * mask_gt

        target_gt_index, fg_mask, mask_pos = select_highest_overlaps(
            mask_pos, overlaps, max_boxes)

        # assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(
            gt_labels, gt_bboxes, target_gt_index, fg_mask, total_anchors,
            batch_size, max_boxes)

        # soft label with iou
        if pd_bboxes is not None:
            ious = iou_calculator(gt_bboxes, pd_bboxes) * mask_pos
            ious = ious.max(axis=-2)[0].unsqueeze(-1)
            target_scores *= ious

        return target_labels.long(
        ), target_bboxes, target_scores, fg_mask.bool()

    def select_topk_candidates(self, distances: Tensor,
                               n_level_bboxes: List[int],
                               mask_gt: Tensor) -> Tuple[Tensor, Tensor]:
        """Select topk candidate bboxes.

        Args:
            distances (Tensor):
            n_level_bboxes (List[int]):
            mask_gt (Tensor):

        Return:
            Tuple[Tensor, Tensor]:
        """
        mask_gt = mask_gt.repeat(1, 1, self.topk).bool()
        level_distances = torch.split(distances, n_level_bboxes, dim=-1)
        is_in_candidate_list = []
        candidate_indexes = []
        start_index = 0
        for per_level_distances, per_level_boxes in zip(
                level_distances, n_level_bboxes):
            end_index = start_index + per_level_boxes
            selected_k = min(self.topk, per_level_boxes)

            _, per_level_topk_indexes = per_level_distances.topk(
                selected_k, dim=-1, largest=False)
            candidate_indexes.append(per_level_topk_indexes + start_index)

            per_level_topk_indexes = torch.where(
                mask_gt, per_level_topk_indexes,
                torch.zeros_like(per_level_topk_indexes))

            is_in_candidate = F.one_hot(per_level_topk_indexes,
                                        per_level_boxes).sum(dim=-2)
            is_in_candidate = torch.where(is_in_candidate > 1,
                                          torch.zeros_like(is_in_candidate),
                                          is_in_candidate)
            is_in_candidate_list.append(is_in_candidate.to(distances.dtype))

            start_index = end_index

        is_in_candidate_list = torch.cat(is_in_candidate_list, dim=-1)
        candidate_indexes = torch.cat(candidate_indexes, dim=-1)

        return is_in_candidate_list, candidate_indexes

    @staticmethod
    def threshold_calculator(is_in_candidate: List, candidate_indexes: Tensor,
                             overlaps: Tensor, total_anchors: int,
                             batch_size: int,
                             max_boxes: int) -> Tuple[Tensor, Tensor]:
        """Calculator bbox threshold.

        Args:
            is_in_candidate (Tensor):
            candidate_indexes (Tensor):
            overlaps (Tensor):
            total_anchors (int):
            batch_size (int):
            max_boxes (int):

        Return:
            Tuple[Tensor, Tensor]:
        """

        n_bs_max_boxes = batch_size * max_boxes
        _candidate_overlaps = torch.where(is_in_candidate > 0, overlaps,
                                          torch.zeros_like(overlaps))
        candidate_indexes = candidate_indexes.reshape([n_bs_max_boxes, -1])

        assist_indexes = total_anchors * torch.arange(
            n_bs_max_boxes, device=candidate_indexes.device)
        assist_indexes = assist_indexes[:, None]
        flatten_indexes = candidate_indexes + assist_indexes

        candidate_overlaps = _candidate_overlaps.reshape(-1)[flatten_indexes]
        candidate_overlaps = candidate_overlaps.reshape(
            [batch_size, max_boxes, -1])

        overlaps_mean_per_gt = candidate_overlaps.mean(axis=-1, keepdim=True)
        overlaps_std_per_gt = candidate_overlaps.std(axis=-1, keepdim=True)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        return overlaps_thr_per_gt, _candidate_overlaps

    def get_targets(self, gt_labels: Tensor, gt_bboxes: Tensor,
                    target_gt_index: Tensor, fg_mask: Tensor,
                    total_anchors: int, batch_size: int,
                    max_boxes: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Get target info.

        Args:
            gt_labels (Tensor):
            gt_bboxes (Tensor):
            target_gt_index (Tensor):
            fg_mask (Tensor):
            total_anchors (int):
            batch_size (int):
            max_boxes (int):

        Return:
            Tuple[Tensor, Tensor, Tensor]:
        """

        # assigned target labels
        batch_index = torch.arange(
            batch_size, dtype=gt_labels.dtype, device=gt_labels.device)
        batch_index = batch_index[..., None]
        target_gt_index = (target_gt_index + batch_index * max_boxes).long()
        target_labels = gt_labels.flatten()[target_gt_index.flatten()]
        target_labels = target_labels.reshape([batch_size, total_anchors])
        target_labels = torch.where(
            fg_mask > 0, target_labels,
            torch.full_like(target_labels, self.bg_index))

        # assigned target boxes
        target_bboxes = gt_bboxes.reshape([-1, 4])[target_gt_index.flatten()]
        target_bboxes = target_bboxes.reshape([batch_size, total_anchors, 4])

        # assigned target scores
        target_scores = F.one_hot(target_labels.long(),
                                  self.num_classes + 1).float()
        target_scores = target_scores[:, :, :self.num_classes]

        return target_labels, target_bboxes, target_scores
