# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from mmdet.models.utils import filter_scores_and_topk, multi_apply
from mmdet.structures.bbox import HorizontalBoxes
from mmdet.utils import (ConfigType, OptConfigType, OptInstanceList,
                         OptMultiConfig)
from mmengine.config import ConfigDict
from mmengine.dist import get_dist_info
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor

from mmyolo.registry import MODELS, TASK_UTILS
from ..utils import make_divisible
from .yolov5_head import YOLOv5Head


@MODELS.register_module()
class RotatedYOLOv5HeadModule(BaseModule):
    """YOLOv5Head head module used in `YOLOv5`.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (Union[int, Sequence]): Number of channels in the input
            feature map.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        num_base_priors (int): The number of priors (points) at a point
            on the feature grid.
        featmap_strides (Sequence[int]): Downsample factor of each feature map.
             Defaults to (8, 16, 32).
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: Union[int, Sequence],
                 widen_factor: float = 1.0,
                 num_base_priors: int = 3,
                 featmap_strides: Sequence[int] = (8, 16, 32),
                 angle_out_attrib: int = 1,
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.widen_factor = widen_factor

        self.featmap_strides = featmap_strides
        self.angle_out_attrib = angle_out_attrib
        self.num_out_attrib = 5 + angle_out_attrib + self.num_classes
        self.num_levels = len(self.featmap_strides)
        self.num_base_priors = num_base_priors

        if isinstance(in_channels, int):
            self.in_channels = [make_divisible(in_channels, widen_factor)
                                ] * self.num_levels
        else:
            self.in_channels = [
                make_divisible(i, widen_factor) for i in in_channels
            ]

        self._init_layers()

    def _init_layers(self):
        """initialize conv layers in YOLOv5 head."""
        self.convs_pred = nn.ModuleList()
        for i in range(self.num_levels):
            conv_pred = nn.Conv2d(self.in_channels[i],
                                  self.num_base_priors * self.num_out_attrib,
                                  1)

            self.convs_pred.append(conv_pred)

    def init_weights(self):
        """Initialize the bias of YOLOv5 head."""
        super().init_weights()
        for mi, s in zip(self.convs_pred, self.featmap_strides):  # from
            b = mi.bias.data.view(self.num_base_priors, -1)
            # obj (8 objects per 640 image)
            b.data[:, 4 + self.angle_out_attrib] += math.log(8 / (640 / s)**2)
            b.data[:, 5 + self.angle_out_attrib:] += math.log(
                0.6 / (self.num_classes - 0.999999))

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
        return multi_apply(self.forward_single, x, self.convs_pred)

    def forward_single(self, x: Tensor,
                       convs: nn.Module) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward feature of a single scale level."""

        pred_map = convs(x)
        bs, _, ny, nx = pred_map.shape
        pred_map = pred_map.view(bs, self.num_base_priors, self.num_out_attrib,
                                 ny, nx)

        angle_start_dim = 4 + self.angle_out_attrib
        cls_start_dim = 5 + self.angle_out_attrib
        cls_score = pred_map[:, :, cls_start_dim:, ...].reshape(bs, -1, ny, nx)
        bbox_pred = pred_map[:, :, :4, ...].reshape(bs, -1, ny, nx)
        angle_pred = pred_map[:, :, 4:angle_start_dim,
                              ...].reshape(bs, -1, ny, nx)
        objectness = pred_map[:, :, angle_start_dim:cls_start_dim,
                              ...].reshape(bs, -1, ny, nx)

        return cls_score, bbox_pred, angle_pred, objectness


@MODELS.register_module()
class RotatedYOLOv5Head(YOLOv5Head):
    """YOLOv5Head head used in `YOLOv5`.

    Args:
        head_module(ConfigType): Base module used for YOLOv5Head
        prior_generator(dict): Points generator feature maps in
            2D points-based detectors.
        bbox_coder (:obj:`ConfigDict` or dict): Config of bbox coder.
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
        loss_bbox (:obj:`ConfigDict` or dict): Config of localization loss.
        loss_obj (:obj:`ConfigDict` or dict): Config of objectness loss.
        prior_match_thr (float): Defaults to 4.0.
        ignore_iof_thr (float): Defaults to -1.0.
        obj_level_weights (List[float]): Defaults to [4.0, 1.0, 0.4].
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config of
            anchor head. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config of
            anchor head. Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(
            self,
            head_module: ConfigType,
            prior_generator: ConfigType = dict(
                type='mmdet.YOLOAnchorGenerator',
                base_sizes=[[(10, 13), (16, 30), (33, 23)],
                            [(30, 61), (62, 45), (59, 119)],
                            [(116, 90), (156, 198), (373, 326)]],
                strides=[8, 16, 32]),
            bbox_coder: ConfigType = dict(type='YOLOv5BBoxCoder'),
            loss_cls: ConfigType = dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True,
                reduction='mean',
                loss_weight=0.5),
            loss_bbox: ConfigType = dict(
                type='IoULoss',
                iou_mode='ciou',
                bbox_format='xywh',
                eps=1e-7,
                reduction='mean',
                loss_weight=0.05,
                return_iou=True),
            loss_obj: ConfigType = dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True,
                reduction='mean',
                loss_weight=1.0),
            loss_angle: OptConfigType = None,
            use_hbbox_loss: bool = False,
            angle_coder: ConfigType = dict(type='mmrotate.PseudoAngleCoder'),
            prior_match_thr: float = 4.0,
            near_neighbor_thr: float = 0.5,
            ignore_iof_thr: float = -1.0,
            obj_level_weights: List[float] = [4.0, 1.0, 0.4],
            train_cfg: OptConfigType = None,
            test_cfg: OptConfigType = None,
            init_cfg: OptMultiConfig = None):

        # self.angle_version = angle_version
        self.use_hbbox_loss = use_hbbox_loss
        self.angle_coder = TASK_UTILS.build(angle_coder)
        head_module.angle_out_attrib = self.angle_coder.encode_size
        super().__init__(
            head_module=head_module,
            prior_generator=prior_generator,
            bbox_coder=bbox_coder,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_obj=loss_obj,
            prior_match_thr=prior_match_thr,
            near_neighbor_thr=near_neighbor_thr,
            ignore_iof_thr=ignore_iof_thr,
            obj_level_weights=obj_level_weights,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)

        self.loss_angle: nn.Module = None
        if loss_angle is not None:
            self.loss_angle = MODELS.build(loss_angle)

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        angle_preds: List[Tensor],
                        objectnesses: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = True,
                        with_nms: bool = True) -> List[InstanceData]:
        """Transform a batch of output features extracted by the head into
        bbox results.
        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_scores) == len(bbox_preds)
        if objectnesses is None:
            with_objectnesses = False
        else:
            with_objectnesses = True
            assert len(cls_scores) == len(objectnesses)

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)

        multi_label = cfg.multi_label
        multi_label &= self.num_classes > 1
        cfg.multi_label = multi_label

        num_imgs = len(batch_img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

        # If the shape does not change, use the previous mlvl_priors
        if featmap_sizes != self.featmap_sizes:
            self.mlvl_priors = self.prior_generator.grid_priors(
                featmap_sizes,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device)
            self.featmap_sizes = featmap_sizes
        flatten_priors = torch.cat(self.mlvl_priors)

        mlvl_strides = [
            flatten_priors.new_full(
                (featmap_size.numel() * self.num_base_priors, ), stride) for
            featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
        ]
        flatten_stride = torch.cat(mlvl_strides)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.num_classes)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_angle_preds = [
            angle_pred.permute(0, 2, 3,
                               1).reshape(num_imgs, -1,
                                          self.angle_coder.encode_size)
            for angle_pred in angle_preds
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_angle_preds = torch.cat(flatten_angle_preds, dim=1)
        flatten_decoded_bboxes = self.bbox_coder.decode(
            flatten_priors[None], flatten_bbox_preds, flatten_stride)

        if with_objectnesses:
            flatten_objectness = [
                objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
                for objectness in objectnesses
            ]
            flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        else:
            flatten_objectness = [None for _ in range(num_imgs)]

        results_list = []
        for (bboxes, angles, scores, objectness,
             img_meta) in zip(flatten_decoded_bboxes, flatten_angle_preds,
                              flatten_cls_scores, flatten_objectness,
                              batch_img_metas):
            # ori_shape = img_meta['ori_shape']
            scale_factor = img_meta['scale_factor']
            if 'pad_param' in img_meta:
                pad_param = img_meta['pad_param']
            else:
                pad_param = None

            score_thr = cfg.get('score_thr', -1)
            # yolox_style does not require the following operations
            if objectness is not None and score_thr > 0 and not cfg.get(
                    'yolox_style', False):
                conf_inds = objectness > score_thr
                bboxes = bboxes[conf_inds, :]
                angles = angles[conf_inds, :]
                scores = scores[conf_inds, :]
                objectness = objectness[conf_inds]

            if objectness is not None:
                # conf = obj_conf * cls_conf
                scores *= objectness[:, None]

            if scores.shape[0] == 0:
                empty_results = InstanceData()
                empty_results.bboxes = bboxes.new_zeros([bboxes.size(0), 5],
                                                        dtype=float)
                empty_results.scores = scores[:, 0]
                empty_results.labels = scores[:, 0].int()
                results_list.append(empty_results)
                continue

            nms_pre = cfg.get('nms_pre', 100000)
            if cfg.multi_label is False:
                scores, labels = scores.max(1, keepdim=True)
                scores, _, keep_idxs, results = filter_scores_and_topk(
                    scores,
                    score_thr,
                    nms_pre,
                    results=dict(labels=labels[:, 0]))
                labels = results['labels']
            else:
                scores, labels, keep_idxs, _ = filter_scores_and_topk(
                    scores, score_thr, nms_pre)

            decoded_angle = self.angle_coder.decode(
                angles[keep_idxs], keepdim=True)
            bboxes = HorizontalBoxes.xyxy_to_cxcywh(bboxes[keep_idxs])
            rbboxes = torch.cat([bboxes, decoded_angle], dim=-1)

            results = InstanceData(
                scores=scores, labels=labels, bboxes=rbboxes)

            if rescale:
                if pad_param is not None:
                    results.bboxes -= results.bboxes.new_tensor(
                        [pad_param[2], pad_param[0], 0, 0, 0])

                scale_x, scale_y = scale_factor
                ctrs, w, h, t = torch.split(
                    results.bboxes, [2, 1, 1, 1], dim=-1)
                cos_value, sin_value = torch.cos(t), torch.sin(t)

                # Refer to https://github.com/facebookresearch/detectron2/blob/main/detectron2/structures/rotated_boxes.py # noqa
                # rescale centers
                ctrs = ctrs * ctrs.new_tensor([scale_x, scale_y])
                # rescale width and height
                w = w * torch.sqrt((scale_x * cos_value)**2 +
                                   (scale_y * sin_value)**2)
                h = h * torch.sqrt((scale_x * sin_value)**2 +
                                   (scale_y * cos_value)**2)
                # recalculate theta
                t = torch.atan2(scale_x * sin_value, scale_y * cos_value)
                results.bboxes = torch.cat([ctrs, w, h, t], dim=-1)

            if cfg.get('yolox_style', False):
                # do not need max_per_img
                cfg.max_per_img = len(results)

            results = self._bbox_post_process(
                results=results,
                cfg=cfg,
                rescale=False,
                with_nms=with_nms,
                img_meta=img_meta)
            # results.bboxes[:, 0::2].clamp_(0, ori_shape[1])
            # results.bboxes[:, 1::2].clamp_(0, ori_shape[0])

            results_list.append(results)
        return results_list

    def loss(self, x: Tuple[Tensor], batch_data_samples: Union[list,
                                                               dict]) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`], dict): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """

        if isinstance(batch_data_samples, list):
            losses = super().loss(x, batch_data_samples)
        else:
            outs = self(x)
            # Fast version
            loss_inputs = outs + (batch_data_samples['bboxes_labels'],
                                  batch_data_samples['img_metas'])
            losses = self.loss_by_feat(*loss_inputs)

        return losses

    def loss_by_feat(
            self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            angle_preds: Sequence[Tensor],
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
            batch_gt_instances (Sequence[InstanceData]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (Sequence[dict]): Meta information of each image,
                e.g., image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
        Returns:
            dict[str, Tensor]: A dictionary of losses.
        """
        if self.ignore_iof_thr != -1:
            # TODO: Support fast version
            # convert ignore gt
            batch_target_ignore_list = []
            for i, gt_instances_ignore in enumerate(batch_gt_instances_ignore):
                bboxes = gt_instances_ignore.bboxes
                labels = gt_instances_ignore.labels
                index = bboxes.new_full((len(bboxes), 1), i)
                # (batch_idx, label, bboxes)
                target = torch.cat((index, labels[:, None].float(), bboxes),
                                   dim=1)
                batch_target_ignore_list.append(target)

            # (num_bboxes, 6)
            batch_gt_targets_ignore = torch.cat(
                batch_target_ignore_list, dim=0)
            if batch_gt_targets_ignore.shape[0] != 0:
                # Consider regions with ignore in annotations
                return self._loss_by_feat_with_ignore(
                    cls_scores,
                    bbox_preds,
                    objectnesses,
                    batch_gt_instances=batch_gt_instances,
                    batch_img_metas=batch_img_metas,
                    batch_gt_instances_ignore=batch_gt_targets_ignore)

        # 1. Convert gt to norm format
        batch_targets_normed = self._convert_gt_to_norm_format(
            batch_gt_instances, batch_img_metas)

        device = cls_scores[0].device
        loss_cls = torch.zeros(1, device=device)
        loss_box = torch.zeros(1, device=device)
        loss_obj = torch.zeros(1, device=device)
        loss_ang = torch.zeros(1, device=device)
        scaled_factor = torch.ones(8, device=device)

        for i in range(self.num_levels):
            batch_size, _, h, w = bbox_preds[i].shape
            target_obj = torch.zeros_like(objectnesses[i])

            # empty gt bboxes
            if batch_targets_normed.shape[1] == 0:
                loss_box += bbox_preds[i].sum() * 0
                loss_cls += cls_scores[i].sum() * 0
                loss_obj += self.loss_obj(
                    objectnesses[i], target_obj) * self.obj_level_weights[i]
                if self.loss_angle is not None:
                    loss_ang += angle_preds[i].sum() * 0
                continue

            priors_base_sizes_i = self.priors_base_sizes[i]
            # feature map scale whwh
            scaled_factor[2:6] = torch.tensor(
                bbox_preds[i].shape)[[3, 2, 3, 2]]
            # Scale batch_targets from range 0-1 to range 0-features_maps size.
            # (num_base_priors, num_bboxes, 7)
            batch_targets_scaled = batch_targets_normed * scaled_factor

            # 2. Shape match
            wh_ratio = batch_targets_scaled[...,
                                            4:6] / priors_base_sizes_i[:, None]
            match_inds = torch.max(
                wh_ratio, 1 / wh_ratio).max(2)[0] < self.prior_match_thr
            batch_targets_scaled = batch_targets_scaled[match_inds]

            # no gt bbox matches anchor
            if batch_targets_scaled.shape[0] == 0:
                loss_box += bbox_preds[i].sum() * 0
                loss_cls += cls_scores[i].sum() * 0
                loss_obj += self.loss_obj(
                    objectnesses[i], target_obj) * self.obj_level_weights[i]
                continue

            # 3. Positive samples with additional neighbors

            # check the left, up, right, bottom sides of the
            # targets grid, and determine whether assigned
            # them as positive samples as well.
            batch_targets_cxcy = batch_targets_scaled[:, 2:4]
            grid_xy = scaled_factor[[2, 3]] - batch_targets_cxcy
            left, up = ((batch_targets_cxcy % 1 < self.near_neighbor_thr) &
                        (batch_targets_cxcy > 1)).T
            right, bottom = ((grid_xy % 1 < self.near_neighbor_thr) &
                             (grid_xy > 1)).T
            offset_inds = torch.stack(
                (torch.ones_like(left), left, up, right, bottom))

            batch_targets_scaled = batch_targets_scaled.repeat(
                (5, 1, 1))[offset_inds]
            retained_offsets = self.grid_offset.repeat(1, offset_inds.shape[1],
                                                       1)[offset_inds]

            # prepare pred results and positive sample indexes to
            # calculate class loss and bbox lo
            _chunk_targets = batch_targets_scaled.split([2, 2, 2, 1, 1], dim=1)
            (img_class_inds, grid_xy, grid_wh, grid_ang,
             priors_inds) = _chunk_targets
            priors_inds, (img_inds, class_inds) = priors_inds.long().view(
                -1), img_class_inds.long().T

            grid_xy_long = (grid_xy -
                            retained_offsets * self.near_neighbor_thr).long()
            grid_x_inds, grid_y_inds = grid_xy_long.T
            if self.use_hbbox_loss:
                bboxes_targets = torch.cat((grid_xy - grid_xy_long, grid_wh),
                                           1)
            else:
                bboxes_targets = torch.cat(
                    (grid_xy - grid_xy_long, grid_wh, grid_ang), 1)
            angle_targets = self.angle_coder.encode(grid_ang)

            # 4. Calculate loss
            # bbox loss
            retained_bbox_pred = bbox_preds[i].reshape(
                batch_size, self.num_base_priors, -1, h,
                w)[img_inds, priors_inds, :, grid_y_inds, grid_x_inds]
            retained_angle_pred = angle_preds[i].reshape(
                batch_size, self.num_base_priors, -1, h,
                w)[img_inds, priors_inds, :, grid_y_inds, grid_x_inds]

            priors_base_sizes_i = priors_base_sizes_i[priors_inds]
            decoded_bbox_pred = self._decode_bbox_to_xywh(
                retained_bbox_pred, priors_base_sizes_i)

            if self.use_hbbox_loss:
                loss_box_i, iou = self.loss_bbox(decoded_bbox_pred,
                                                 bboxes_targets)
            else:
                decoded_angle = self.angle_coder.decode(
                    retained_angle_pred, keepdim=True)
                decoded_rbbox_pred = torch.cat(
                    [decoded_bbox_pred, decoded_angle], dim=-1)
                loss_box_i, iou = self.loss_bbox(decoded_rbbox_pred,
                                                 bboxes_targets)
            loss_box += loss_box_i

            # ang loss
            if self.loss_angle is not None:
                loss_ang += self.loss_angle(retained_angle_pred, angle_targets)

            # obj loss
            iou = iou.detach().clamp(0)
            target_obj[img_inds, priors_inds, grid_y_inds,
                       grid_x_inds] = iou.type(target_obj.dtype)
            loss_obj += self.loss_obj(objectnesses[i],
                                      target_obj) * self.obj_level_weights[i]

            # cls loss
            if self.num_classes > 1:
                pred_cls_scores = cls_scores[i].reshape(
                    batch_size, self.num_base_priors, -1, h,
                    w)[img_inds, priors_inds, :, grid_y_inds, grid_x_inds]

                target_class = torch.full_like(pred_cls_scores, 0.)
                target_class[range(batch_targets_scaled.shape[0]),
                             class_inds] = 1.
                loss_cls += self.loss_cls(pred_cls_scores, target_class)
            else:
                loss_cls += cls_scores[i].sum() * 0

        _, world_size = get_dist_info()
        losses = dict()
        losses['loss_cls'] = loss_cls * batch_size * world_size
        losses['loss_obj'] = loss_obj * batch_size * world_size
        losses['loss_bbox'] = loss_box * batch_size * world_size
        if self.loss_angle is not None:
            losses['loss_angle'] = loss_ang * batch_size * world_size

        return losses

    def _convert_gt_to_norm_format(self,
                                   batch_gt_instances: Sequence[InstanceData],
                                   batch_img_metas: Sequence[dict]) -> Tensor:
        if isinstance(batch_gt_instances, torch.Tensor):
            # fast version
            img_shape = batch_img_metas[0]['batch_input_shape']
            gt_bboxes_xywht = batch_gt_instances[:, 2:]
            gt_bboxes_xywht[:, 1::2] /= img_shape[0]
            gt_bboxes_xywht[:, 0] /= img_shape[1]
            gt_bboxes_xywht[:, 2] /= img_shape[1]
            batch_gt_instances[:, 2:] = gt_bboxes_xywht

            # (num_base_priors, num_bboxes, 7)
            batch_targets_normed = batch_gt_instances.repeat(
                self.num_base_priors, 1, 1)
        else:
            batch_target_list = []
            # Convert xyxy bbox to yolo format.
            for i, gt_instances in enumerate(batch_gt_instances):
                img_shape = batch_img_metas[i]['batch_input_shape']
                bboxes = gt_instances.bboxes
                labels = gt_instances.labels

                # xy1, xy2 = bboxes.split((2, 2), dim=-1)
                # bboxes = torch.cat([(xy2 + xy1) / 2, (xy2 - xy1)], dim=-1)
                # normalized to 0-1
                bboxes[:, 1::2] /= img_shape[0]
                bboxes[:, 0] /= img_shape[1]
                bboxes[:, 2] /= img_shape[1]

                index = bboxes.new_full((len(bboxes), 1), i)
                # (batch_idx, label, normed_bbox)
                target = torch.cat((index, labels[:, None].float(), bboxes),
                                   dim=1)
                batch_target_list.append(target)

            # (num_base_priors, num_bboxes, 6)
            batch_targets_normed = torch.cat(
                batch_target_list, dim=0).repeat(self.num_base_priors, 1, 1)

        # (num_base_priors, num_bboxes, 1)
        batch_targets_prior_inds = self.prior_inds.repeat(
            1, batch_targets_normed.shape[1])[..., None]
        # (num_base_priors, num_bboxes, 8)
        # (img_ind, labels, bbox_cx, bbox_cy,
        #  bbox_w, bbox_h, bbox_angle, prior_ind)
        batch_targets_normed = torch.cat(
            (batch_targets_normed, batch_targets_prior_inds), 2)
        return batch_targets_normed

    def _decode_bbox_to_xywh(self, bbox_pred, priors_base_sizes) -> Tensor:
        bbox_pred = bbox_pred.sigmoid()
        pred_xy = bbox_pred[:, :2] * 2 - 0.5
        pred_wh = (bbox_pred[:, 2:] * 2)**2 * priors_base_sizes
        decoded_bbox_pred = torch.cat((pred_xy, pred_wh), dim=-1)
        return decoded_bbox_pred

    def _loss_by_feat_with_ignore(
            self, cls_scores: Sequence[Tensor], bbox_preds: Sequence[Tensor],
            angle_preds: Sequence[Tensor], objectnesses: Sequence[Tensor],
            batch_gt_instances: Sequence[InstanceData],
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore: Sequence[Tensor]) -> dict:
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
            batch_gt_instances (Sequence[InstanceData]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (Sequence[dict]): Meta information of each image,
                e.g., image size, scaling factor, etc.
            batch_gt_instances_ignore (Sequence[Tensor]): Ignore boxes with
                batch_ids and labels, each is a 2D-tensor, the channel number
                is 6, means that (batch_id, label, xmin, ymin, xmax, ymax).
        Returns:
            dict[str, Tensor]: A dictionary of losses.
        """
        # TODO
        raise NotImplementedError
