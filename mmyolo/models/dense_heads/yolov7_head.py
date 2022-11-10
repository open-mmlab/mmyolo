# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.utils import OptInstanceList
from mmengine.dist import get_dist_info
from mmengine.structures import InstanceData
from torch import Tensor

from mmyolo.registry import MODELS
from .yolov5_head import YOLOv5Head, YOLOv5HeadModule


class ImplicitA(nn.Module):

    def __init__(self, channel, mean=0., std=.02):
        super().__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit + x


class ImplicitM(nn.Module):

    def __init__(self, channel, mean=1., std=.02):
        super().__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit * x


def bbox_overlaps(box1, box2, bbox_format: str = 'xywh'):
    # nx4, mx4
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def xywh2xyxy(x):
        y = x.clone()
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    box1 = xywh2xyxy(box1)
    box2 = xywh2xyxy(box2)

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
             torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)


def cat_multi_level_in_place(*multi_level_tensor, place_hold_var):
    for level_tensor in multi_level_tensor:
        for i, var in enumerate(level_tensor):
            if len(var) > 0:
                level_tensor[i] = torch.cat(var, dim=0)
            else:
                level_tensor[i] = place_hold_var


@MODELS.register_module()
class YOLOv7HeadModule(YOLOv5HeadModule):

    def _init_layers(self):
        """initialize conv layers in YOLOv5 head."""
        self.convs_pred = nn.ModuleList()
        for i in range(self.num_levels):
            conv_pred = nn.Sequential(
                ImplicitA(self.in_channels[i]),
                nn.Conv2d(self.in_channels[i],
                          self.num_base_priors * self.num_out_attrib, 1),
                ImplicitM(self.num_base_priors * self.num_out_attrib),
            )
            self.convs_pred.append(conv_pred)

    def init_weights(self):
        """Initialize the bias of YOLOv5 head."""
        super(YOLOv5HeadModule, self).init_weights()
        for mi, s in zip(self.convs_pred, self.featmap_strides):  # from
            mi = mi[1]  # nn.Conv2d

            b = mi.bias.data.view(3, -1)
            # obj (8 objects per 640 image)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:] += math.log(0.6 / (self.num_classes - 0.99))

            mi.bias.data = b.view(-1)


@MODELS.register_module()
class YOLOv7Head(YOLOv5Head):
    """YOLOv7Head head used in `YOLOv7 <https://arxiv.org/abs/2207.02696>`_.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def loss_by_feat(
            self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            objectnesses: Sequence[Tensor],
            batch_gt_instances: Sequence[InstanceData],
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (Sequence[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (Sequence[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            objectnesses (Sequence[Tensor]): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
        Returns:
            dict[str, Tensor]: A dictionary of losses.
        """

        # x = torch.load('a.pth')
        # cls_scores = x['cls_scores']
        # bbox_preds = x['bbox_preds']
        # objectnesses = x['objectnesses']
        # batch_gt_instances = x['batch_gt_instances']

        batch_input_shape = batch_img_metas[0]['batch_input_shape']  # hw
        # (1,4)
        batch_input_shape_wh = cls_scores[0].new_tensor(batch_input_shape[::-1]).repeat((1, 2))

        # TODO: refine
        head_preds = []
        for bbox_pred, objectness, cls_score in zip(bbox_preds, objectnesses, cls_scores):
            b, _, h, w = bbox_pred.shape
            bbox_pred = bbox_pred.reshape(b, 3, -1, h, w)
            objectness = objectness.reshape(b, 3, -1, h, w)
            cls_score = cls_score.reshape(b, 3, -1, h, w)
            head_pred = torch.cat([bbox_pred, objectness, cls_score], dim=2).permute(0, 1, 3, 4, 2).contiguous()
            head_preds.append(head_pred)

        num_batch = cls_scores[0].shape[0]
        device = cls_scores[0].device
        loss_cls = torch.zeros(1, device=device)
        loss_box = torch.zeros(1, device=device)
        loss_obj = torch.zeros(1, device=device)

        # 1. Convert gt to norm xywh format (num_bboxes, 6)
        # 6 is mean (batch_idx, cls_id, x_norm, y_norm, w_norm, h_norm)
        batch_targets_normed = self._convert_gt_to_norm_format(
            batch_gt_instances, batch_img_metas)

        multi_level_indices, multi_level_proirs = [], []

        num_batch_gt = batch_targets_normed.shape[0]
        batch_targets_prior_inds = self.prior_inds.repeat(1, num_batch_gt)[..., None]
        # (num_base_priors, num_batch_gt, 7)
        # 7 is mean (batch_idx, cls_id, x_norm, y_norm, w_norm, h_norm, prior_idx)
        batch_targets_normed = torch.cat((batch_targets_normed.repeat(self.num_base_priors, 1, 1),
                                          batch_targets_prior_inds), 2)

        scaled_factor = torch.ones(7, device=device)
        for i in range(self.num_levels):
            priors_base_sizes_i = self.priors_base_sizes[i]
            # (1, 1, feat_shape_w, feat_shape_h, feat_shape_w, feat_shape_h)
            scaled_factor[2:6] = torch.tensor(bbox_preds[i].shape)[[3, 2, 3, 2]]

            # Scale batch_targets from range 0-1 to range 0-features_maps size.
            # (num_base_priors, num_batch_gt, 7)
            batch_targets_scaled = batch_targets_normed * scaled_factor

            if num_batch_gt > 0:
                # Shape match
                wh_ratio = batch_targets_scaled[..., 4:6] / priors_base_sizes_i[:, None]
                match_inds = torch.max(
                    wh_ratio, 1. / wh_ratio).max(2)[0] < self.prior_match_thr
                batch_targets_scaled = batch_targets_scaled[match_inds]  # (num_matched_target, 7)

                # Positive samples with additional neighbors
                batch_targets_cxcy = batch_targets_scaled[:, 2:4]
                grid_xy = scaled_factor[[2, 3]] - batch_targets_cxcy
                left, up = ((batch_targets_cxcy % 1 < 0.5) &
                            (batch_targets_cxcy > 1)).T
                right, bottom = ((grid_xy % 1 < 0.5) & (grid_xy > 1)).T
                offset_inds = torch.stack(
                    (torch.ones_like(left), left, up, right, bottom))
                batch_targets_scaled = batch_targets_scaled.repeat(
                    (5, 1, 1))[offset_inds]  # ()
                retained_offsets = self.grid_offset.repeat(1, offset_inds.shape[1],
                                                           1)[offset_inds]
            else:
                # TODO
                batch_targets_scaled = batch_targets_scaled[0]
                retained_offsets = 0

            _chunk_targets = batch_targets_scaled.chunk(4, 1)
            img_class_inds, grid_xy, grid_wh, priors_inds = _chunk_targets
            priors_inds, (batch_idx, _) = priors_inds.long().view(
                -1), img_class_inds.long().T
            grid_xy_long = (grid_xy - retained_offsets).long()
            grid_x_inds, grid_y_inds = grid_xy_long.T

            multi_level_indices.append(
                (batch_idx, priors_inds, grid_y_inds.clamp_(0, scaled_factor[3] - 1),
                 grid_x_inds.clamp_(0, scaled_factor[2] - 1)))
            multi_level_proirs.append(priors_base_sizes_i[priors_inds])

        multi_level_positive_info = [[] for _ in range(self.num_levels)]
        multi_level_proirs_matched = [[] for _ in range(self.num_levels)]
        multi_level_targets_normed = [[] for _ in range(self.num_levels)]

        for batch_idx in range(num_batch):
            # (num_batch_gt, 7)
            # 7 is mean (batch_idx, cls_id, x_norm, y_norm, w_norm, h_norm, prior_idx)
            targets_normed = batch_targets_normed[0]
            # (num_gt, 7)
            targets_normed = targets_normed[targets_normed[:, 0] == batch_idx]
            num_gt = targets_normed.shape[0]
            if num_gt == 0:
                continue

            _multi_level_decoderd_bbox = []
            _multi_level_obj_cls = []
            _multi_level_proirs = []
            _multi_level_positive_info = []
            _from_which_layer = []

            for i, head_pred in enumerate(head_preds):
                level_batch_idx, prior_ind, grid_y, grid_x = multi_level_indices[i]
                proirs = multi_level_proirs[i]

                idx = (level_batch_idx == batch_idx)
                level_batch_idx, prior_ind, grid_y, grid_x = level_batch_idx[idx], prior_ind[idx], grid_y[idx], grid_x[
                    idx]
                _multi_level_positive_info.append(
                    torch.stack([level_batch_idx, prior_ind, grid_y, grid_x], dim=0).reshape(4, -1).T)

                _multi_level_proirs.append(proirs[idx])
                _from_which_layer.append(torch.ones(size=(len(level_batch_idx),)) * i)

                # (n,85)
                pred_positive = head_pred[level_batch_idx, prior_ind, grid_y, grid_x]
                _multi_level_obj_cls.append(pred_positive[:, 4:])

                # decoded
                grid = torch.stack([grid_x, grid_y], dim=1)
                pred_positive_cxcy = (pred_positive[:, :2].sigmoid() * 2. - 0.5 + grid) * self.featmap_strides[i]
                pred_positive_wh = (pred_positive[:, 2:4].sigmoid() * 2) ** 2 * multi_level_proirs[i][idx] * \
                    self.featmap_strides[i]
                pred_positive_xywh = torch.cat([pred_positive_cxcy, pred_positive_wh], dim=-1)
                _multi_level_decoderd_bbox.append(pred_positive_xywh)

            # 1 calc pair_wise_iou_loss
            _multi_level_decoderd_bbox = torch.cat(_multi_level_decoderd_bbox, dim=0)
            num_pred_positive = _multi_level_decoderd_bbox.shape[0]
            if num_pred_positive == 0:
                continue

            # scaled xywh
            targets_scaled_bbox = targets_normed[:, 2:6] * batch_input_shape_wh
            pair_wise_iou = bbox_overlaps(targets_scaled_bbox, _multi_level_decoderd_bbox)
            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

            # 2 calc pair_wise_cls_loss
            _multi_level_obj_cls = torch.cat(_multi_level_obj_cls, dim=0).float().sigmoid()
            _multi_level_positive_info = torch.cat(_multi_level_positive_info, dim=0)
            _from_which_layer = torch.cat(_from_which_layer, dim=0)
            _multi_level_proirs = torch.cat(_multi_level_proirs, dim=0)

            gt_cls_per_image = (
                F.one_hot(targets_normed[:, 1].to(torch.int64),
                          self.num_classes).float().unsqueeze(1).repeat(
                    1, num_pred_positive, 1))
            # cls_score * obj
            cls_preds_ = _multi_level_obj_cls[:, 1:].unsqueeze(0).repeat(num_gt, 1, 1) \
                         * _multi_level_obj_cls[:, 0:1].unsqueeze(0).repeat(num_gt, 1, 1)
            y = cls_preds_.sqrt_()
            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
                torch.log(y / (1 - y)), gt_cls_per_image,
                reduction='none').sum(-1)
            del cls_preds_

            # calc cost
            cost = (pair_wise_cls_loss + 3.0 * pair_wise_iou_loss)

            # num_gt, num_match_pred
            matching_matrix = torch.zeros_like(cost)

            top_k, _ = torch.topk(
                pair_wise_iou, min(10, pair_wise_iou.shape[1]), dim=1)
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)

            # Select only topk matches per gt
            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(
                    cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
                matching_matrix[gt_idx][pos_idx] = 1.0
            del top_k, dynamic_ks

            # Each prediction box can match at most one gt box,
            # and if there are more than one, only the least costly one can be taken
            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(
                    cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = matching_matrix.sum(0) > 0.0
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)

            targets_normed = targets_normed[matched_gt_inds]

            _multi_level_positive_info = _multi_level_positive_info[fg_mask_inboxes]
            _from_which_layer = _from_which_layer[fg_mask_inboxes]
            _multi_level_proirs = _multi_level_proirs[fg_mask_inboxes]

            # Rearranged in the order of the prediction layers
            # to facilitate loss
            for i in range(self.num_levels):
                layer_idx = _from_which_layer == i
                multi_level_positive_info[i].append(_multi_level_positive_info[layer_idx])
                multi_level_targets_normed[i].append(targets_normed[layer_idx])
                multi_level_proirs_matched[i].append(_multi_level_proirs[layer_idx])

        place_hold_var = torch.tensor([], device=device, dtype=torch.int64)
        cat_multi_level_in_place(multi_level_positive_info,
                                 multi_level_proirs_matched,
                                 multi_level_targets_normed,
                                 place_hold_var=place_hold_var)
        scaled_factors = [
            torch.tensor(head_pred.shape, device=device)[[3, 2, 3, 2]] for head_pred in head_preds
        ]

        # calc losses
        for i, head_pred in enumerate(head_preds):  # layer index, layer predictions
            batch_inds, proir_idx, grid_y, grid_x = multi_level_positive_info[i].T
            proirs = multi_level_proirs_matched[i]
            targets_normed = multi_level_targets_normed[i]

            target_obj = torch.zeros_like(head_pred[..., 0])

            num_pred_positive = batch_inds.shape[0]
            if num_pred_positive > 0:
                head_pred_positive = head_pred[batch_inds, proir_idx, grid_y, grid_x]

                # decoded
                pred_bboxes = head_pred_positive[:, :4].sigmoid()
                pred_bbox_cxy = pred_bboxes[:, :2] * 2. - 0.5
                pred_bbox_wh = (pred_bboxes[:, 2:4] * 2) ** 2 * proirs
                decoded_pred_bbox = torch.cat((pred_bbox_cxy, pred_bbox_wh), 1)  # predicted box

                target_bbox_scaled = targets_normed[:, 2:6] * scaled_factors[i]
                grid = torch.stack([grid_x, grid_y], dim=1)
                target_bbox_scaled[:, :2] -= grid

                loss_box_i, iou = self.loss_bbox(decoded_pred_bbox, target_bbox_scaled)
                loss_box += loss_box_i

                target_obj[batch_inds, proir_idx, grid_y, grid_x] = iou.detach().clamp(0).type(target_obj.dtype)

                if self.num_classes > 1:
                    pred_cls_scores = targets_normed[:, 1].long()
                    # cls loss (only if multiple classes)
                    target_class = torch.full_like(head_pred_positive[:, 5:], 0., device=device)
                    target_class[range(num_pred_positive), pred_cls_scores] = 1.
                    loss_cls += self.loss_cls(head_pred_positive[:, 5:], target_class)

            loss_obj += self.loss_obj(head_pred[..., 4], target_obj) * self.obj_level_weights[i]

        _, world_size = get_dist_info()
        return dict(
            loss_cls=loss_cls * num_batch * world_size,
            loss_conf=loss_obj * num_batch * world_size,
            loss_bbox=loss_box * num_batch * world_size)

    def _convert_gt_to_norm_format(self,
                                   batch_gt_instances: Sequence[InstanceData],
                                   batch_img_metas: Sequence[dict]) -> Tensor:
        if isinstance(batch_gt_instances, torch.Tensor):
            # fast version
            img_shape = batch_img_metas[0]['batch_input_shape']
            gt_bboxes_xyxy = batch_gt_instances[:, 2:]
            xy1, xy2 = gt_bboxes_xyxy.split((2, 2), dim=-1)
            gt_bboxes_xywh = torch.cat([(xy2 + xy1) / 2, (xy2 - xy1)], dim=-1)
            gt_bboxes_xywh[:, 1::2] /= img_shape[0]
            gt_bboxes_xywh[:, 0::2] /= img_shape[1]
            batch_gt_instances[:, 2:] = gt_bboxes_xywh  # (num_bboxes, 6)
        else:
            # TODO
            batch_target_list = []
            # Convert xyxy bbox to yolo format.
            for i, gt_instances in enumerate(batch_gt_instances):
                img_shape = batch_img_metas[i]['batch_input_shape']
                bboxes = gt_instances.bboxes
                labels = gt_instances.labels

                xy1, xy2 = bboxes.split((2, 2), dim=-1)
                bboxes = torch.cat([(xy2 + xy1) / 2, (xy2 - xy1)], dim=-1)
                # normalized to 0-1
                bboxes[:, 1::2] /= img_shape[0]
                bboxes[:, 0::2] /= img_shape[1]

                index = bboxes.new_full((len(bboxes), 1), i)
                # (batch_idx, label, normed_bbox)
                target = torch.cat((index, labels[:, None].float(), bboxes),
                                   dim=1)
                batch_target_list.append(target)

            # (num_bboxes, 6)
            batch_gt_instances = torch.cat(
                batch_target_list, dim=0)

        return batch_gt_instances
