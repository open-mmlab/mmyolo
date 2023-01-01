# Copyright (c) OpenMMLab. All rights reserved.

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def select_candidates_in_gts(priors_points: Tensor,
                             gt_bboxes: Tensor,
                             eps: float = 1e-9) -> Tensor:
    """Select the positive priors' center in gt.

    Args:
        priors_points (Tensor): Model priors points,
            shape(num_priors, 2)
        gt_bboxes (Tensor): Ground true bboxes,
            shape(batch_size, num_gt, 4)
        eps (float): Default to 1e-9.
    Return:
        (Tensor): shape(batch_size, num_gt, num_priors)
    """
    batch_size, num_gt, _ = gt_bboxes.size()
    gt_bboxes = gt_bboxes.reshape([-1, 4])

    priors_number = priors_points.size(0)
    priors_points = priors_points.unsqueeze(0).repeat(batch_size * num_gt, 1,
                                                      1)

    # calculate the left, top, right, bottom distance between positive
    # prior center and gt side
    gt_bboxes_lt = gt_bboxes[:, 0:2].unsqueeze(1).repeat(1, priors_number, 1)
    gt_bboxes_rb = gt_bboxes[:, 2:4].unsqueeze(1).repeat(1, priors_number, 1)
    bbox_deltas = torch.cat(
        [priors_points - gt_bboxes_lt, gt_bboxes_rb - priors_points], dim=-1)
    bbox_deltas = bbox_deltas.reshape([batch_size, num_gt, priors_number, -1])

    return (bbox_deltas.min(axis=-1)[0] > eps).to(gt_bboxes.dtype)


def select_highest_overlaps(pos_mask: Tensor, overlaps: Tensor,
                            num_gt: int) -> Tuple[Tensor, Tensor, Tensor]:
    """If an anchor box is assigned to multiple gts, the one with the highest
    iou will be selected.

    Args:
        pos_mask (Tensor): The assigned positive sample mask,
            shape(batch_size, num_gt, num_priors)
        overlaps (Tensor): IoU between all bbox and ground truth,
            shape(batch_size, num_gt, num_priors)
        num_gt (int): Number of ground truth.
    Return:
        gt_idx_pre_prior (Tensor): Target ground truth index,
            shape(batch_size, num_priors)
        fg_mask_pre_prior (Tensor): Force matching ground truth,
            shape(batch_size, num_priors)
        pos_mask (Tensor): The assigned positive sample mask,
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

        pos_mask = torch.where(mask_multi_gts, is_max_overlaps, pos_mask)
        fg_mask_pre_prior = pos_mask.sum(axis=-2)

    gt_idx_pre_prior = pos_mask.argmax(axis=-2)
    return gt_idx_pre_prior, fg_mask_pre_prior, pos_mask


# TODO:'mmdet.BboxOverlaps2D' will cause gradient inconsistency,
# which will be found and solved in a later version.
def yolov6_iou_calculator(bbox1: Tensor,
                          bbox2: Tensor,
                          eps: float = 1e-9) -> Tensor:
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
    bbox1_x1y1, bbox1_x2y2 = bbox1[:, :, :, 0:2], bbox1[:, :, :, 2:4]
    bbox2_x1y1, bbox2_x2y2 = bbox2[:, :, :, 0:2], bbox2[:, :, :, 2:4]

    # calculate overlap area
    overlap = (torch.minimum(bbox1_x2y2, bbox2_x2y2) -
               torch.maximum(bbox1_x1y1, bbox2_x1y1)).clip(0).prod(-1)

    # calculate bbox area
    bbox1_area = (bbox1_x2y2 - bbox1_x1y1).clip(0).prod(-1)
    bbox2_area = (bbox2_x2y2 - bbox2_x1y1).clip(0).prod(-1)

    union = bbox1_area + bbox2_area - overlap + eps

    return overlap / union
