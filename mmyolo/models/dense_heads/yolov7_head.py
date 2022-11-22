# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.models.utils import multi_apply
from mmdet.utils import ConfigType, OptInstanceList
from mmengine.dist import get_dist_info
from mmengine.structures import InstanceData
from torch import Tensor

from mmyolo.registry import MODELS
from ..layers import ImplicitA, ImplicitM
from ..task_modules.assigners.batch_yolov7_assigner import BatchYOLOv7Assigner
from .yolov5_head import YOLOv5Head, YOLOv5HeadModule


@MODELS.register_module()
class YOLOv7HeadModule(YOLOv5HeadModule):
    """YOLOv7Head head module used in YOLOv7."""

    def _init_layers(self):
        """initialize conv layers in YOLOv7 head."""
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
        """Initialize the bias of YOLOv7 head."""
        super(YOLOv5HeadModule, self).init_weights()
        for mi, s in zip(self.convs_pred, self.featmap_strides):  # from
            mi = mi[1]  # nn.Conv2d

            b = mi.bias.data.view(3, -1)
            # obj (8 objects per 640 image)
            b.data[:, 4] += math.log(8 / (640 / s)**2)
            b.data[:, 5:] += math.log(0.6 / (self.num_classes - 0.99))

            mi.bias.data = b.view(-1)


@MODELS.register_module()
class YOLOv7p6HeadModule(YOLOv5HeadModule):
    """YOLOv7Head head module used in YOLOv7."""

    def __init__(self,
                 *args,
                 main_out_channels: Sequence[int] = [256, 512, 768, 1024],
                 aux_out_channels: Sequence[int] = [320, 640, 960, 1280],
                 use_aux: bool = True,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 **kwargs):
        self.main_out_channels = main_out_channels
        self.aux_out_channels = aux_out_channels
        self.use_aux = use_aux
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        super().__init__(*args, **kwargs)

    def _init_layers(self):
        """initialize conv layers in YOLOv7 head."""
        self.main_convs_pred = nn.ModuleList()
        for i in range(self.num_levels):
            conv_pred = nn.Sequential(
                ConvModule(
                    self.in_channels[i],
                    self.main_out_channels[i],
                    3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg),
                ImplicitA(self.main_out_channels[i]),
                nn.Conv2d(self.main_out_channels[i],
                          self.num_base_priors * self.num_out_attrib, 1),
                ImplicitM(self.num_base_priors * self.num_out_attrib),
            )
            self.main_convs_pred.append(conv_pred)

        if self.use_aux:
            self.aux_convs_pred = nn.ModuleList()
            for i in range(self.num_levels):
                aux_pred = nn.Sequential(
                    ConvModule(
                        self.in_channels[i],
                        self.aux_out_channels[i],
                        3,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg),
                    nn.Conv2d(self.aux_out_channels[i],
                              self.num_base_priors * self.num_out_attrib, 1))
                self.aux_convs_pred.append(aux_pred)
        else:
            self.aux_convs_pred = [None] * len(self.main_convs_pred)

    def init_weights(self):
        """Initialize the bias of YOLOv5 head."""
        super(YOLOv5HeadModule, self).init_weights()
        for mi, aux, s in zip(self.main_convs_pred, self.aux_convs_pred,
                              self.featmap_strides):  # from
            mi = mi[2]  # nn.Conv2d
            b = mi.bias.data.view(3, -1)
            # obj (8 objects per 640 image)
            b.data[:, 4] += math.log(8 / (640 / s)**2)
            b.data[:, 5:] += math.log(0.6 / (self.num_classes - 0.99))
            mi.bias.data = b.view(-1)

            if self.use_aux:
                aux = aux[1]  # nn.Conv2d
                b = aux.bias.data.view(3, -1)
                # obj (8 objects per 640 image)
                b.data[:, 4] += math.log(8 / (640 / s)**2)
                b.data[:, 5:] += math.log(0.6 / (self.num_classes - 0.99))
                mi.bias.data = b.view(-1)

    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions, and objectnesses.
        """
        assert len(x) == self.num_levels
        return multi_apply(self.forward_single, x, self.main_convs_pred,
                           self.aux_convs_pred)

    def forward_single(self, x: Tensor, convs: nn.Module,
                       aux_convs: Optional[nn.Module]) \
            -> Tuple[Union[Tensor, List], Union[Tensor, List],
                     Union[Tensor, List]]:
        """Forward feature of a single scale level."""

        pred_map = convs(x)
        bs, _, ny, nx = pred_map.shape
        pred_map = pred_map.view(bs, self.num_base_priors, self.num_out_attrib,
                                 ny, nx)

        cls_score = pred_map[:, :, 5:, ...].reshape(bs, -1, ny, nx)
        bbox_pred = pred_map[:, :, :4, ...].reshape(bs, -1, ny, nx)
        objectness = pred_map[:, :, 4:5, ...].reshape(bs, -1, ny, nx)

        if not self.training or not self.use_aux:
            return cls_score, bbox_pred, objectness
        else:
            aux_pred_map = aux_convs(x)
            aux_pred_map = aux_pred_map.view(bs, self.num_base_priors,
                                             self.num_out_attrib, ny, nx)
            aux_cls_score = aux_pred_map[:, :, 5:, ...].reshape(bs, -1, ny, nx)
            aux_bbox_pred = aux_pred_map[:, :, :4, ...].reshape(bs, -1, ny, nx)
            aux_objectness = aux_pred_map[:, :, 4:5,
                                          ...].reshape(bs, -1, ny, nx)

            return [cls_score,
                    aux_cls_score], [bbox_pred, aux_bbox_pred
                                     ], [objectness, aux_objectness]


@MODELS.register_module()
class YOLOv7Head(YOLOv5Head):
    """YOLOv7Head head used in `YOLOv7 <https://arxiv.org/abs/2207.02696>`_.

    Args:
        simota_candidate_topk (int): The candidate top-k which used to
            get top-k ious to calculate dynamic-k in BatchYOLOv7Assigner.
            Defaults to 10.
        simota_iou_weight (float): The scale factor for regression
            iou cost in BatchYOLOv7Assigner. Defaults to 3.0.
        simota_cls_weight (float): The scale factor for classification
            cost in BatchYOLOv7Assigner. Defaults to 1.0.
    """

    def __init__(self,
                 *args,
                 simota_candidate_topk: int = 20,
                 simota_iou_weight: float = 3.0,
                 simota_cls_weight: float = 1.0,
                 aux_loss_weights: float = 0.25,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_loss_weights = aux_loss_weights
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
            cls_scores: Sequence[Union[Tensor, List]],
            bbox_preds: Sequence[Union[Tensor, List]],
            objectnesses: Sequence[Union[Tensor, List]],
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

        if isinstance(cls_scores[0], Sequence):
            with_aux = True
            batch_size = cls_scores[0][0].shape[0]
            device = cls_scores[0][0].device

            bbox_preds_main, bbox_preds_aux = zip(*bbox_preds)
            objectnesses_main, objectnesses_aux = zip(*objectnesses)
            cls_scores_main, cls_scores_aux = zip(*cls_scores)

            head_preds = self._merge_predict_results(bbox_preds_main,
                                                     objectnesses_main,
                                                     cls_scores_main)
            head_preds_aux = self._merge_predict_results(
                bbox_preds_aux, objectnesses_aux, cls_scores_aux)
        else:
            with_aux = False
            batch_size = cls_scores[0].shape[0]
            device = cls_scores[0].device

            head_preds = self._merge_predict_results(bbox_preds, objectnesses,
                                                     cls_scores)

        # Convert gt to norm xywh format
        # (num_base_priors, num_batch_gt, 7)
        # 7 is mean (batch_idx, cls_id, x_norm, y_norm,
        # w_norm, h_norm, prior_idx)
        batch_targets_normed = self._convert_gt_to_norm_format(
            batch_gt_instances, batch_img_metas)

        scaled_factors = [
            torch.tensor(head_pred.shape, device=device)[[3, 2, 3, 2]]
            for head_pred in head_preds
        ]

        loss_cls, loss_obj, loss_box = self._calc_loss(
            head_preds=head_preds,
            head_preds_aux=None,
            batch_targets_normed=batch_targets_normed,
            near_neighbor_thr=self.near_neighbor_thr,
            scaled_factors=scaled_factors,
            batch_img_metas=batch_img_metas,
            device=device)

        if with_aux:
            loss_cls_aux, loss_obj_aux, loss_box_aux = self._calc_loss(
                head_preds=head_preds,
                head_preds_aux=head_preds_aux,
                batch_targets_normed=batch_targets_normed,
                near_neighbor_thr=self.near_neighbor_thr * 2,
                scaled_factors=scaled_factors,
                batch_img_metas=batch_img_metas,
                device=device)
            loss_cls += self.aux_loss_weights * loss_cls_aux
            loss_obj += self.aux_loss_weights * loss_obj_aux
            loss_box += self.aux_loss_weights * loss_box_aux

        _, world_size = get_dist_info()
        return dict(
            loss_cls=loss_cls * batch_size * world_size,
            loss_obj=loss_obj * batch_size * world_size,
            loss_bbox=loss_box * batch_size * world_size)

    def _calc_loss(self, head_preds, head_preds_aux, batch_targets_normed,
                   near_neighbor_thr, scaled_factors, batch_img_metas, device):
        loss_cls = torch.zeros(1, device=device)
        loss_box = torch.zeros(1, device=device)
        loss_obj = torch.zeros(1, device=device)

        assigner_results = self.assigner(
            head_preds,
            batch_targets_normed,
            batch_img_metas[0]['batch_input_shape'],
            self.priors_base_sizes,
            self.grid_offset,
            near_neighbor_thr=near_neighbor_thr)
        # mlvl is mean multi_level
        mlvl_positive_infos = assigner_results['mlvl_positive_infos']
        mlvl_priors = assigner_results['mlvl_priors']
        mlvl_targets_normed = assigner_results['mlvl_targets_normed']

        if head_preds_aux is not None:
            # This is mean calc aux branch loss
            head_preds = head_preds_aux

        for i, head_pred in enumerate(head_preds):
            batch_inds, proir_idx, grid_x, grid_y = mlvl_positive_infos[i].T
            num_pred_positive = batch_inds.shape[0]
            target_obj = torch.zeros_like(head_pred[..., 0])
            # empty positive sampler
            if num_pred_positive == 0:
                loss_box += head_pred[..., :4].sum() * 0
                loss_cls += head_pred[..., 5:].sum() * 0
                loss_obj += self.loss_obj(
                    head_pred[..., 4], target_obj) * self.obj_level_weights[i]
                continue

            priors = mlvl_priors[i]
            targets_normed = mlvl_targets_normed[i]

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
        return loss_cls, loss_obj, loss_box

    def _merge_predict_results(self, bbox_preds: Sequence[Tensor],
                               objectnesses: Sequence[Tensor],
                               cls_scores: Sequence[Tensor]) -> List[Tensor]:
        """Merge predict output from 3 heads.

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

        Returns:
              List[Tensor]: Merged output.
        """
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
