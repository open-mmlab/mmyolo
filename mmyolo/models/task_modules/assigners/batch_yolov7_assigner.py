# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps


def _cat_multi_level_tensor_in_place(*multi_level_tensor, place_hold_var):
    """concat multi-level tensor in place."""
    for level_tensor in multi_level_tensor:
        for i, var in enumerate(level_tensor):
            if len(var) > 0:
                level_tensor[i] = torch.cat(var, dim=0)
            else:
                level_tensor[i] = place_hold_var


class BatchYOLOv7Assigner(nn.Module):
    """Batch YOLOv7 Assigner.

    It consists of two assigning steps:

        1. YOLOv5 cross-grid sample assigning
        2. SimOTA assigning

    This code referenced to
    https://github.com/WongKinYiu/yolov7/blob/main/utils/loss.py.

    Args:
        num_classes (int): Number of classes.
        num_base_priors (int): Number of base priors.
        featmap_strides (Sequence[int]): Feature map strides.
        prior_match_thr (float): Threshold to match priors.
            Defaults to 4.0.
        candidate_topk (int): Number of topk candidates to
            assign. Defaults to 10.
        iou_weight (float): IOU weight. Defaults to 3.0.
        cls_weight (float): Class weight. Defaults to 1.0.
    """

    def __init__(self,
                 num_classes: int,
                 num_base_priors: int,
                 featmap_strides: Sequence[int],
                 prior_match_thr: float = 4.0,
                 candidate_topk: int = 10,
                 iou_weight: float = 3.0,
                 cls_weight: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.num_base_priors = num_base_priors
        self.featmap_strides = featmap_strides
        # yolov5 param
        self.prior_match_thr = prior_match_thr
        # simota param
        self.candidate_topk = candidate_topk
        self.iou_weight = iou_weight
        self.cls_weight = cls_weight

    @torch.no_grad()
    def forward(self,
                pred_results,
                batch_targets_normed,
                batch_input_shape,
                priors_base_sizes,
                grid_offset,
                near_neighbor_thr=0.5) -> dict:
        """Forward function."""
        # (num_base_priors, num_batch_gt, 7)
        # 7 is mean (batch_idx, cls_id, x_norm, y_norm,
        # w_norm, h_norm, prior_idx)

        # mlvl is mean multi_level
        if batch_targets_normed.shape[1] == 0:
            # empty gt of batch
            num_levels = len(pred_results)
            return dict(
                mlvl_positive_infos=[pred_results[0].new_empty(
                    (0, 4))] * num_levels,
                mlvl_priors=[] * num_levels,
                mlvl_targets_normed=[] * num_levels)

        # if near_neighbor_thr = 0.5 are mean the nearest
        # 3 neighbors are also considered positive samples.
        # if near_neighbor_thr = 1.0 are mean the nearest
        # 5 neighbors are also considered positive samples.
        mlvl_positive_infos, mlvl_priors = self.yolov5_assigner(
            pred_results,
            batch_targets_normed,
            priors_base_sizes,
            grid_offset,
            near_neighbor_thr=near_neighbor_thr)

        mlvl_positive_infos, mlvl_priors, \
            mlvl_targets_normed = self.simota_assigner(
                pred_results, batch_targets_normed, mlvl_positive_infos,
                mlvl_priors, batch_input_shape)

        place_hold_var = batch_targets_normed.new_empty((0, 4))
        _cat_multi_level_tensor_in_place(
            mlvl_positive_infos,
            mlvl_priors,
            mlvl_targets_normed,
            place_hold_var=place_hold_var)

        return dict(
            mlvl_positive_infos=mlvl_positive_infos,
            mlvl_priors=mlvl_priors,
            mlvl_targets_normed=mlvl_targets_normed)

    def yolov5_assigner(self,
                        pred_results,
                        batch_targets_normed,
                        priors_base_sizes,
                        grid_offset,
                        near_neighbor_thr=0.5):
        """YOLOv5 cross-grid sample assigner."""
        num_batch_gts = batch_targets_normed.shape[1]
        assert num_batch_gts > 0

        mlvl_positive_infos, mlvl_priors = [], []

        scaled_factor = torch.ones(7, device=pred_results[0].device)
        for i in range(len(pred_results)):  # lever
            priors_base_sizes_i = priors_base_sizes[i]
            # (1, 1, feat_shape_w, feat_shape_h, feat_shape_w, feat_shape_h)
            scaled_factor[2:6] = torch.tensor(
                pred_results[i].shape)[[3, 2, 3, 2]]

            # Scale batch_targets from range 0-1 to range 0-features_maps size.
            # (num_base_priors, num_batch_gts, 7)
            batch_targets_scaled = batch_targets_normed * scaled_factor

            # Shape match
            wh_ratio = batch_targets_scaled[...,
                                            4:6] / priors_base_sizes_i[:, None]
            match_inds = torch.max(
                wh_ratio, 1. / wh_ratio).max(2)[0] < self.prior_match_thr
            batch_targets_scaled = batch_targets_scaled[
                match_inds]  # (num_matched_target, 7)

            # no gt bbox matches anchor
            if batch_targets_scaled.shape[0] == 0:
                mlvl_positive_infos.append(
                    batch_targets_scaled.new_empty((0, 4)))
                mlvl_priors.append([])
                continue

            # Positive samples with additional neighbors
            batch_targets_cxcy = batch_targets_scaled[:, 2:4]
            grid_xy = scaled_factor[[2, 3]] - batch_targets_cxcy
            left, up = ((batch_targets_cxcy % 1 < near_neighbor_thr) &
                        (batch_targets_cxcy > 1)).T
            right, bottom = ((grid_xy % 1 < near_neighbor_thr) &
                             (grid_xy > 1)).T
            offset_inds = torch.stack(
                (torch.ones_like(left), left, up, right, bottom))
            batch_targets_scaled = batch_targets_scaled.repeat(
                (5, 1, 1))[offset_inds]  # ()
            retained_offsets = grid_offset.repeat(1, offset_inds.shape[1],
                                                  1)[offset_inds]

            # batch_targets_scaled: (num_matched_target, 7)
            # 7 is mean (batch_idx, cls_id, x_scaled,
            # y_scaled, w_scaled, h_scaled, prior_idx)

            # mlvl_positive_info: (num_matched_target, 4)
            # 4 is mean (batch_idx, prior_idx, x_scaled, y_scaled)
            mlvl_positive_info = batch_targets_scaled[:, [0, 6, 2, 3]]
            retained_offsets = retained_offsets * near_neighbor_thr
            mlvl_positive_info[:,
                               2:] = mlvl_positive_info[:,
                                                        2:] - retained_offsets
            mlvl_positive_info[:, 2].clamp_(0, scaled_factor[2] - 1)
            mlvl_positive_info[:, 3].clamp_(0, scaled_factor[3] - 1)
            mlvl_positive_info = mlvl_positive_info.long()
            priors_inds = mlvl_positive_info[:, 1]

            mlvl_positive_infos.append(mlvl_positive_info)
            mlvl_priors.append(priors_base_sizes_i[priors_inds])

        return mlvl_positive_infos, mlvl_priors

    def simota_assigner(self, pred_results, batch_targets_normed,
                        mlvl_positive_infos, mlvl_priors, batch_input_shape):
        """SimOTA assigner."""
        num_batch_gts = batch_targets_normed.shape[1]
        assert num_batch_gts > 0
        num_levels = len(mlvl_positive_infos)

        mlvl_positive_infos_matched = [[] for _ in range(num_levels)]
        mlvl_priors_matched = [[] for _ in range(num_levels)]
        mlvl_targets_normed_matched = [[] for _ in range(num_levels)]

        for batch_idx in range(pred_results[0].shape[0]):
            # (num_batch_gt, 7)
            # 7 is mean (batch_idx, cls_id, x_norm, y_norm,
            # w_norm, h_norm, prior_idx)
            targets_normed = batch_targets_normed[0]
            # (num_gt, 7)
            targets_normed = targets_normed[targets_normed[:, 0] == batch_idx]
            num_gts = targets_normed.shape[0]

            if num_gts == 0:
                continue

            _mlvl_decoderd_bboxes = []
            _mlvl_obj_cls = []
            _mlvl_priors = []
            _mlvl_positive_infos = []
            _from_which_layer = []

            for i, head_pred in enumerate(pred_results):
                # (num_matched_target, 4)
                #  4 is mean (batch_idx, prior_idx, grid_x, grid_y)
                _mlvl_positive_info = mlvl_positive_infos[i]
                if _mlvl_positive_info.shape[0] == 0:
                    continue

                idx = (_mlvl_positive_info[:, 0] == batch_idx)
                _mlvl_positive_info = _mlvl_positive_info[idx]
                _mlvl_positive_infos.append(_mlvl_positive_info)

                priors = mlvl_priors[i][idx]
                _mlvl_priors.append(priors)

                _from_which_layer.append(
                    _mlvl_positive_info.new_full(
                        size=(_mlvl_positive_info.shape[0], ), fill_value=i))

                # (n,85)
                level_batch_idx, prior_ind, \
                    grid_x, grid_y = _mlvl_positive_info.T
                pred_positive = head_pred[level_batch_idx, prior_ind, grid_y,
                                          grid_x]
                _mlvl_obj_cls.append(pred_positive[:, 4:])

                # decoded
                grid = torch.stack([grid_x, grid_y], dim=1)
                pred_positive_cxcy = (pred_positive[:, :2].sigmoid() * 2. -
                                      0.5 + grid) * self.featmap_strides[i]
                pred_positive_wh = (pred_positive[:, 2:4].sigmoid() * 2) ** 2 \
                    * priors * self.featmap_strides[i]
                pred_positive_xywh = torch.cat(
                    [pred_positive_cxcy, pred_positive_wh], dim=-1)
                _mlvl_decoderd_bboxes.append(pred_positive_xywh)

            if len(_mlvl_decoderd_bboxes) == 0:
                continue

            # 1 calc pair_wise_iou_loss
            _mlvl_decoderd_bboxes = torch.cat(_mlvl_decoderd_bboxes, dim=0)
            num_pred_positive = _mlvl_decoderd_bboxes.shape[0]

            if num_pred_positive == 0:
                continue

            # scaled xywh
            batch_input_shape_wh = pred_results[0].new_tensor(
                batch_input_shape[::-1]).repeat((1, 2))
            targets_scaled_bbox = targets_normed[:, 2:6] * batch_input_shape_wh

            targets_scaled_bbox = bbox_cxcywh_to_xyxy(targets_scaled_bbox)
            _mlvl_decoderd_bboxes = bbox_cxcywh_to_xyxy(_mlvl_decoderd_bboxes)
            pair_wise_iou = bbox_overlaps(targets_scaled_bbox,
                                          _mlvl_decoderd_bboxes)
            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

            # 2 calc pair_wise_cls_loss
            _mlvl_obj_cls = torch.cat(_mlvl_obj_cls, dim=0).float().sigmoid()
            _mlvl_positive_infos = torch.cat(_mlvl_positive_infos, dim=0)
            _from_which_layer = torch.cat(_from_which_layer, dim=0)
            _mlvl_priors = torch.cat(_mlvl_priors, dim=0)

            gt_cls_per_image = (
                F.one_hot(targets_normed[:, 1].to(torch.int64),
                          self.num_classes).float().unsqueeze(1).repeat(
                              1, num_pred_positive, 1))
            # cls_score * obj
            cls_preds_ = _mlvl_obj_cls[:, 1:]\
                .unsqueeze(0)\
                .repeat(num_gts, 1, 1) \
                * _mlvl_obj_cls[:, 0:1]\
                .unsqueeze(0).repeat(num_gts, 1, 1)
            y = cls_preds_.sqrt_()
            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
                torch.log(y / (1 - y)), gt_cls_per_image,
                reduction='none').sum(-1)
            del cls_preds_

            # calc cost
            cost = (
                self.cls_weight * pair_wise_cls_loss +
                self.iou_weight * pair_wise_iou_loss)

            # num_gt, num_match_pred
            matching_matrix = torch.zeros_like(cost)

            top_k, _ = torch.topk(
                pair_wise_iou,
                min(self.candidate_topk, pair_wise_iou.shape[1]),
                dim=1)
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)

            # Select only topk matches per gt
            for gt_idx in range(num_gts):
                _, pos_idx = torch.topk(
                    cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
                matching_matrix[gt_idx][pos_idx] = 1.0
            del top_k, dynamic_ks

            # Each prediction box can match at most one gt box,
            # and if there are more than one,
            # only the least costly one can be taken
            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(
                    cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = matching_matrix.sum(0) > 0.0
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)

            targets_normed = targets_normed[matched_gt_inds]
            _mlvl_positive_infos = _mlvl_positive_infos[fg_mask_inboxes]
            _from_which_layer = _from_which_layer[fg_mask_inboxes]
            _mlvl_priors = _mlvl_priors[fg_mask_inboxes]

            # Rearranged in the order of the prediction layers
            # to facilitate loss
            for i in range(num_levels):
                layer_idx = _from_which_layer == i
                mlvl_positive_infos_matched[i].append(
                    _mlvl_positive_infos[layer_idx])
                mlvl_priors_matched[i].append(_mlvl_priors[layer_idx])
                mlvl_targets_normed_matched[i].append(
                    targets_normed[layer_idx])

        results = mlvl_positive_infos_matched, \
            mlvl_priors_matched, \
            mlvl_targets_normed_matched
        return results
