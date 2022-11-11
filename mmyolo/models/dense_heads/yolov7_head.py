# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Sequence

import torch
import torch.nn as nn
from mmdet.utils import OptInstanceList
from mmengine.dist import get_dist_info
from mmengine.structures import InstanceData
from torch import Tensor

from mmyolo.registry import MODELS
from ..task_modules.assigners.batch_yolov7_assigner import BatchYOLOv7Assigner
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
            b.data[:, 4] += math.log(8 / (640 / s)**2)
            b.data[:, 5:] += math.log(0.6 / (self.num_classes - 0.99))

            mi.bias.data = b.view(-1)


@MODELS.register_module()
class YOLOv7Head(YOLOv5Head):
    """YOLOv7Head head used in `YOLOv7 <https://arxiv.org/abs/2207.02696>`_."""

    def __init__(self,
                 *args,
                 simota_candidate_topk: int = 10,
                 simota_iou_weight: float = 3.0,
                 simota_cls_weight: float = 1.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.assigner = BatchYOLOv7Assigner(
            num_classes=self.num_classes,
            num_base_priors=self.num_base_priors,
            featmap_strides=self.featmap_strides,
            prior_match_thr=self.prior_match_thr,
            candidate_topk=simota_candidate_topk,
            iou_weight=simota_iou_weight,
            cls_weight=simota_cls_weight)

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
        batch_size = cls_scores[0].shape[0]
        device = cls_scores[0].device
        loss_cls = torch.zeros(1, device=device)
        loss_box = torch.zeros(1, device=device)
        loss_obj = torch.zeros(1, device=device)

        head_preds = self._merge_predict_results(bbox_preds, objectnesses,
                                                 cls_scores)
        scaled_factors = [
            torch.tensor(head_pred.shape, device=device)[[3, 2, 3, 2]]
            for head_pred in head_preds
        ]

        # 1. Convert gt to norm xywh format (num_bboxes, 6)
        # 6 is mean (batch_idx, cls_id, x_norm, y_norm, w_norm, h_norm)
        batch_targets_normed = self._convert_gt_to_norm_format(
            batch_gt_instances, batch_img_metas)

        assigner_results = self.assigner(
            head_preds, batch_targets_normed,
            batch_img_metas[0]['batch_input_shape'], self.priors_base_sizes,
            self.prior_inds, self.grid_offset)
        # mlvl is mean multi_level
        mlvl_positive_infos = assigner_results['mlvl_positive_infos']
        mlvl_priors = assigner_results['mlvl_priors']
        mlvl_targets_normed = assigner_results['mlvl_targets_normed']

        # calc losses
        for i, head_pred in enumerate(head_preds):
            batch_inds, proir_idx, grid_x, grid_y = mlvl_positive_infos[i].T
            priors = mlvl_priors[i]
            targets_normed = mlvl_targets_normed[i]
            num_pred_positive = batch_inds.shape[0]

            target_obj = torch.zeros_like(head_pred[..., 0])

            # empty positive sampler
            if num_pred_positive == 0:
                loss_box += head_pred[..., :4].sum() * 0
                loss_cls += head_pred[..., 5:].sum() * 0
                loss_obj += self.loss_obj(
                    head_pred[..., 4], target_obj) * self.obj_level_weights[i]
                continue

            head_pred_positive = head_pred[batch_inds, proir_idx, grid_y,
                                           grid_x]

            # calc bbox loss
            grid_xy = torch.stack([grid_x, grid_y], dim=1)
            decoded_pred_bbox = self._decode_bbox_to_xywh(
                head_pred_positive[:, :4], priors, grid_xy)
            target_bbox_scaled = targets_normed[:, 2:6] * scaled_factors[i]

            loss_box_i, iou = self.loss_bbox(decoded_pred_bbox,
                                             target_bbox_scaled)
            loss_box += loss_box_i

            # calc obj loss
            target_obj[batch_inds, proir_idx, grid_y,
                       grid_x] = iou.detach().clamp(0).type(target_obj.dtype)
            loss_obj += self.loss_obj(head_pred[..., 4],
                                      target_obj) * self.obj_level_weights[i]

            # calc cls loss
            if self.num_classes > 1:
                pred_cls_scores = targets_normed[:, 1].long()
                target_class = torch.full_like(
                    head_pred_positive[:, 5:], 0., device=device)
                target_class[range(num_pred_positive), pred_cls_scores] = 1.
                loss_cls += self.loss_cls(head_pred_positive[:, 5:],
                                          target_class)
            else:
                loss_cls += head_pred_positive[:, 5:].sum() * 0

        _, world_size = get_dist_info()
        return dict(
            loss_cls=loss_cls * batch_size * world_size,
            loss_conf=loss_obj * batch_size * world_size,
            loss_bbox=loss_box * batch_size * world_size)

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
            batch_gt_instances = torch.cat(batch_target_list, dim=0)

        return batch_gt_instances

    def _merge_predict_results(self, bbox_preds, objectnesses, cls_scores):
        head_preds = []
        for bbox_pred, objectness, cls_score in zip(bbox_preds, objectnesses,
                                                    cls_scores):
            b, _, h, w = bbox_pred.shape
            bbox_pred = bbox_pred.reshape(b, self.num_base_priors, -1, h, w)
            objectness = objectness.reshape(b, self.num_base_priors, -1, h, w)
            cls_score = cls_score.reshape(b, self.num_base_priors, -1, h, w)
            head_pred = torch.cat([bbox_pred, objectness, cls_score],
                                  dim=2).permute(0, 1, 3, 4, 2).contiguous()
            head_preds.append(head_pred)
        return head_preds

    def _decode_bbox_to_xywh(self, bbox_pred, priors_base_sizes,
                             grid_xy) -> Tensor:
        bbox_pred = bbox_pred.sigmoid()
        pred_xy = bbox_pred[:, :2] * 2 - 0.5 + grid_xy
        pred_wh = (bbox_pred[:, 2:] * 2)**2 * priors_base_sizes
        decoded_bbox_pred = torch.cat((pred_xy, pred_wh), dim=-1)
        return decoded_bbox_pred
