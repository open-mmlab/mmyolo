# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.utils import ConfigType
from torch import Tensor

from mmyolo.registry import TASK_UTILS


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


def select_candidates_in_gts(priors_cxy_points: Tensor,
                             gt_bboxes: Tensor,
                             eps: float = 1e-9) -> Tensor:
    """Select the positive anchors' center in gt.

    Args:
        priors_cxy_points (Tensor): Priors center xy points,
            shape(num_priors, 2)
        gt_bboxes (Tensor): Ground true bboxes,
            shape(batch_size, num_gt, 4)
        eps (float): Default to 1e-9.
    Return:
        (Tensor): shape(batch_size, num_gt, num_priors)
    """
    batch_size, num_gt, _ = gt_bboxes.size()
    gt_bboxes = gt_bboxes.reshape([-1, 4])

    priors_number = priors_cxy_points.size(0)
    priors_cxy_points = priors_cxy_points.unsqueeze(0).repeat(
        batch_size * num_gt, 1, 1)

    # calculate the left, top, right, bottom distance between positive
    # prior center and gt side
    gt_bboxes_lt = gt_bboxes[:, 0:2].unsqueeze(1).repeat(1, priors_number, 1)
    gt_bboxes_rb = gt_bboxes[:, 2:4].unsqueeze(1).repeat(1, priors_number, 1)
    bbox_deltas = torch.cat(
        [priors_cxy_points - gt_bboxes_lt, gt_bboxes_rb - priors_cxy_points],
        dim=-1)
    bbox_deltas = bbox_deltas.reshape([batch_size, num_gt, priors_number, -1])

    return (bbox_deltas.min(axis=-1)[0] > eps).to(gt_bboxes.dtype)


def select_highest_overlaps(pos_mask: Tensor, overlaps: Tensor,
                            num_gt: int) -> Tuple[Tensor, Tensor, Tensor]:
    """If an anchor box is assigned to multiple gts, the one with the highest
    iou will be selected.

    Args:
        pos_mask (Tensor): Possible mask,
            shape(batch_size, num_gt, num_priors)
        overlaps (Tensor): IoU between all bbox and ground true,
            shape(batch_size, num_gt, num_priors)
        num_gt (int): Number of ground true.
    Return:
        gt_idx_pre_prior (Tensor): Target ground true index,
            shape(batch_size, num_priors)
        fg_mask_pre_prior (Tensor): Force matching ground true,
            shape(batch_size, num_priors)
        pos_mask (Tensor): Possible mask,
            shape(batch_size, num_gt, num_priors)
    """
    fg_mask_pre_prior = pos_mask.sum(axis=-2)

    # Make sure the positive sample matches the only one and is the largest IoU
    if fg_mask_pre_prior.max() > 1:
        mask_multi_gts = (fg_mask_pre_prior.unsqueeze(1) > 1).repeat(
            [1, num_gt, 1])
        index = overlaps.argmax(axis=1)
        is_max_overlaps = F.one_hot(index, num_gt)
        is_max_overlaps = \
            is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)

        pos_mask = torch.where(mask_multi_gts, is_max_overlaps,
                                    pos_mask)
        fg_mask_pre_prior = pos_mask.sum(axis=-2)

    gt_idx_pre_prior = pos_mask.argmax(axis=-2)
    return gt_idx_pre_prior, fg_mask_pre_prior, pos_mask


def iou_calculator(bbox1: Tensor, bbox2: Tensor, eps: float = 1e-9) -> Tensor:
    """Calculate iou for batch.

    Args:
        bbox1 (Tensor): shape(batch size, num_gt, 4)
        bbox2 (Tensor): shape(batch size, num_priors, 4)
        eps (float): Default to 1e-9.
    Return:
        (Tensor): IoU, shape(size, num_gt, num_priors)
    """
    bbox1 = bbox1.unsqueeze(2)  # [N, M1, 4] -> [N, M1, 1, 4]
    bbox2 = bbox2.unsqueeze(1)  # [N, M2, 4] -> [N, 1, M2, 4]

    # calculate xy info of predict and gt bbox
    pred_x1y1, pred_x2y2 = bbox1[:, :, :, 0:2], bbox1[:, :, :, 2:4]
    gt_x1y1, gt_x2y2 = bbox2[:, :, :, 0:2], bbox2[:, :, :, 2:4]

    # calculate overlap area
    overlap = (torch.minimum(pred_x2y2, gt_x2y2) -
               torch.maximum(pred_x1y1, gt_x1y1)).clip(0).prod(-1)

    # calculate bbox area
    pred_area = (pred_x2y2 - pred_x1y1).clip(0).prod(-1)
    gt_area = (gt_x2y2 - gt_x1y1).clip(0).prod(-1)

    union = pred_area + gt_area - overlap + eps

    return overlap / union


@TASK_UTILS.register_module()
class BatchATSSAssigner(nn.Module):
    """Adaptive Training Sample Selection Assigner.
        This code is based on
        https://github.com/meituan/YOLOv6/blob/main/yolov6/assigners/atss_assigner.py
    """

    def __init__(
            self,
            topk: int = 9,
            iou2d_calculator: ConfigType = dict(type='mmdet.BboxOverlaps2D'),
            num_classes: int = 80):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.iou2d_calculator = TASK_UTILS.build(iou2d_calculator)

    @torch.no_grad()
    def forward(
            self, priors: torch.Tensor, num_level_priors: List,
            gt_labels: torch.Tensor, gt_bboxes: torch.Tensor,
            pad_bbox_flag: torch.Tensor, pred_bboxes: torch.Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""Get assigner result
        Args:
            priors (Tensor): model predictions, it can be anchors, points,
                or bboxes predicted by the model, shape(num_priors, 4).
            num_level_priors (List): Number of bboxes in each level, len(3)
            gt_labels (Tensor): Ground truth label,
                shape(batch_size, num_gt, 1)
            gt_bboxes (Tensor): Ground truth bbox,
                shape(batch_size, num_gt, 4)
            pad_bbox_flag (Tensor): Ground truth bbox mask,
                1 means bbox, 0 means no bbox,
                shape(batch_size, num_gt, 1),
            pred_bboxes (Tensor): Predicted bounding boxes,
                shape(batch_size, num_priors, 4),

        Returns:
            assigned_labels (Tensor): shape(batch_size, num_gt)
            assigned_bboxes (Tensor): shape(batch_size, num_gt, 4)
            assigned_scores (Tensor):
            fg_mask_pre_prior (Tensor): shape(bs, num_gt)
        """
        batch_size = gt_bboxes.size(0)
        num_gt, num_priors = gt_bboxes.size(1), priors.size(0)

        if num_gt == 0:
            device = gt_bboxes.device
            return (torch.full([batch_size, num_priors],
                               self.num_classes).to(device),
                    torch.zeros([batch_size, num_priors, 4]).to(device),
                    torch.zeros([batch_size, num_priors,
                                 self.num_classes]).to(device),
                    torch.zeros([batch_size, num_priors]).to(device))

        # compute iou between all bbox and gt
        overlaps = self.iou2d_calculator(gt_bboxes.reshape([-1, 4]), priors)
        overlaps = overlaps.reshape([batch_size, -1, num_priors])

        # compute center distance between all bbox and gt
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
            ious = iou_calculator(gt_bboxes, pred_bboxes) * pos_mask
            ious = ious.max(axis=-2)[0].unsqueeze(-1)
            assigned_scores *= ious

        # TODO change it to dict
        return assigned_labels.long(
        ), assigned_bboxes, assigned_scores, fg_mask_pre_prior.bool()

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
            num_gt (int): Number of ground true.

        Return:
            overlaps_thr_per_gt (Tensor): Overlap threshold of per ground true,
                shape(batch_size, num_gt, 1).
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
            assigned_gt_inds (Tensor): Assigned ground true indexes,
                shape(batch_size, num_priors)
            fg_mask_pre_prior (Tensor): Force ground true matching mask,
                shape(batch_size, num_priors)
            num_priors (int): Number of priors.
            batch_size (int): Batch size.
            num_gt (int): Number of ground true.

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
