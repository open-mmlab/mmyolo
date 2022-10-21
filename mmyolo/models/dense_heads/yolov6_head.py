# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.models.utils import multi_apply
from mmdet.utils import (ConfigType, OptConfigType, OptInstanceList,
                         OptMultiConfig)
from mmengine import MessageHub
from mmengine.dist import get_dist_info
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor

from mmyolo.registry import MODELS, TASK_UTILS
from ..utils import make_divisible
from .yolov5_head import YOLOv5Head


@MODELS.register_module()
class YOLOv6HeadModule(BaseModule):
    """YOLOv6Head head module used in `YOLOv6.

    <https://arxiv.org/pdf/2209.02976>`_.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (Union[int, Sequence]): Number of channels in the input
            feature map.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Default: 1.0.
        num_base_priors:int: The number of priors (points) at a point
            on the feature grid.
        featmap_strides (Sequence[int]): Downsample factor of each feature map.
             Defaults to [8, 16, 32].
            None, otherwise False. Defaults to "auto".
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: Union[int, Sequence],
                 widen_factor: float = 1.0,
                 num_base_priors: int = 1,
                 featmap_strides: Sequence[int] = (8, 16, 32),
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)

        self.num_classes = num_classes
        self.featmap_strides = featmap_strides
        self.num_levels = len(self.featmap_strides)
        self.num_base_priors = num_base_priors
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        if isinstance(in_channels, int):
            self.in_channels = [make_divisible(in_channels, widen_factor)
                                ] * self.num_levels
        else:
            self.in_channels = [
                make_divisible(i, widen_factor) for i in in_channels
            ]

        self._init_layers()

    def _init_layers(self):
        """initialize conv layers in YOLOv6 head."""
        # Init decouple head
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        for i in range(self.num_levels):
            self.stems.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.in_channels[i],
                    kernel_size=1,
                    stride=1,
                    padding=1 // 2,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
            self.cls_convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.in_channels[i],
                    kernel_size=3,
                    stride=1,
                    padding=3 // 2,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
            self.reg_convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.in_channels[i],
                    kernel_size=3,
                    stride=1,
                    padding=3 // 2,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=self.in_channels[i],
                    out_channels=self.num_base_priors * self.num_classes,
                    kernel_size=1))
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=self.in_channels[i],
                    out_channels=self.num_base_priors * 4,
                    kernel_size=1))

    def init_weights(self):
        super().init_weights()
        for conv in self.cls_preds:
            b = conv.bias.data.view(-1, )
            b.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            conv.bias.data = b.view(-1)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight.data = w

        for conv in self.reg_preds:
            b = conv.bias.data.view(-1, )
            b.fill_(1.0)
            conv.bias.data = b.view(-1)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight.data = w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions.
        """
        assert len(x) == self.num_levels
        return multi_apply(self.forward_single, x, self.stems, self.cls_convs,
                           self.cls_preds, self.reg_convs, self.reg_preds)

    def forward_single(self, x: torch.Tensor, stem: nn.ModuleList,
                       cls_conv: nn.ModuleList, cls_pred: nn.ModuleList,
                       reg_conv: nn.ModuleList,
                       reg_pred: nn.ModuleList) -> torch.Tensor:
        """Forward feature of a single scale level."""
        y = stem(x)
        cls_x = y
        reg_x = y
        cls_feat = cls_conv(cls_x)
        reg_feat = reg_conv(reg_x)

        cls_score = cls_pred(cls_feat)
        bbox_pred = reg_pred(reg_feat)

        return cls_score, bbox_pred


# Training mode is currently not supported
@MODELS.register_module()
class YOLOv6Head(YOLOv5Head):
    """YOLOv6Head head used in `YOLOv6 <https://arxiv.org/pdf/2209.02976>`_.

    Args:
        head_module(nn.Module): Base module used for YOLOv6Head
        prior_generator(dict): Points generator feature maps
            in 2D points-based detectors.
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
        loss_bbox (:obj:`ConfigDict` or dict): Config of localization loss.
        loss_obj (:obj:`ConfigDict` or dict): Config of objectness loss.
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config of
            anchor head. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config of
            anchor head. Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 head_module: nn.Module,
                 prior_generator: ConfigType = dict(
                     type='mmdet.MlvlPointGenerator',
                     offset=0.5,
                     strides=[8, 16, 32]),
                 bbox_coder: ConfigType = dict(type='DistancePointBBoxCoder'),
                 loss_cls=dict(
                     type='mmdet.VarifocalLoss',
                     use_sigmoid=True,
                     alpha=0.75,
                     gamma=2.0,
                     iou_weighted=True,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='IoULoss',
                     iou_mode='giou',
                     bbox_format='xyxy',
                     eps=1e-10,
                     reduction='mean',
                     loss_weight=2.5,
                     return_iou=False),
                 loss_obj: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='sum',
                     loss_weight=1.0),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            head_module=head_module,
            prior_generator=prior_generator,
            bbox_coder=bbox_coder,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_obj=loss_obj,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)

        self.loss_bbox: nn.Module = MODELS.build(loss_bbox)

    def special_init(self):
        """Since YOLO series algorithms will inherit from YOLOv5Head, but
        different algorithms have special initialization process.

        The special_init function is designed to deal with this situation.
        """
        if self.train_cfg:
            self.initial_epoch = self.train_cfg['initial_epoch']
            self.initial_assigner = TASK_UTILS.build(
                self.train_cfg.initial_assigner)
            self.assigner = TASK_UTILS.build(self.train_cfg.assigner)

    def loss_by_feat(
            self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
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

        num_imgs = len(batch_img_metas)
        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)

        n_anchors_list = [len(n) for n in mlvl_priors]

        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.num_classes)
            for cls_pred in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]

        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_priors = torch.cat(mlvl_priors, dim=0)
        flatten_bboxes = self.bbox_coder.decode(flatten_priors[..., :2],
                                                flatten_bbox_preds,
                                                flatten_priors[..., 2])
        # generate v6 anchor
        cell_half_size = flatten_priors[:, 2:] * 2.5
        flatten_anchors = torch.zeros_like(flatten_priors)
        flatten_anchors[:, :2] = flatten_priors[:, :2] - cell_half_size
        flatten_anchors[:, 2:] = flatten_priors[:, :2] + cell_half_size

        batch_size = flatten_cls_preds.shape[0]

        # targets
        targets = self.preprocess(batch_gt_instances, batch_size)

        gt_labels = targets[:, :, :1]
        gt_bboxes = targets[:, :, 1:]  # xyxy
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        # get epoch information from message hub
        message_hub = MessageHub.get_current_instance()
        self.epoch = message_hub.get_info('epoch')

        # pred_scores = flatten_cls_preds.detach().sigmoid()
        pred_scores = torch.sigmoid(flatten_cls_preds)

        if self.epoch < self.initial_epoch:
            target_labels, target_bboxes, target_scores, fg_mask_pre_prior = \
                self.initial_assigner(
                    flatten_anchors,
                    n_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    pad_bbox_flag,
                    flatten_bboxes.detach()
                )
        else:
            target_labels, target_bboxes, target_scores, fg_mask_pre_prior = \
                self.assigner(
                    flatten_priors[:, :2],
                    gt_labels,
                    gt_bboxes,
                    pad_bbox_flag,
                    pred_scores.detach(),
                    flatten_bboxes.detach()
                )

        # cls loss
        target_labels = torch.where(
            fg_mask_pre_prior > 0, target_labels,
            torch.full_like(target_labels, self.num_classes))
        one_hot_label = F.one_hot(target_labels.long(),
                                  self.num_classes + 1)[..., :-1]
        loss_cls = self.varifocal_loss(pred_scores, target_scores,
                                       one_hot_label)

        # rescale bbox
        stride_tensor = flatten_priors[..., [2]]
        target_bboxes /= stride_tensor
        flatten_bboxes /= stride_tensor

        target_scores_sum = target_scores.sum()
        loss_cls /= target_scores_sum

        # select positive samples mask
        num_pos = fg_mask_pre_prior.sum()
        if num_pos > 0:
            # iou loss
            bbox_mask = fg_mask_pre_prior.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(flatten_bboxes,
                                                  bbox_mask).reshape([-1, 4])
            target_bboxes_pos = torch.masked_select(target_bboxes,
                                                    bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(target_scores.sum(-1),
                                              fg_mask_pre_prior).unsqueeze(-1)
            loss_iou = self.loss_bbox(
                pred_bboxes_pos,
                target_bboxes_pos,
                weight=bbox_weight,
                avg_factor=target_scores_sum)

        else:
            loss_iou = torch.tensor(0.).to(flatten_bbox_preds.device)

        _, world_size = get_dist_info()
        return dict(
            loss_cls=loss_cls * world_size, loss_bbox=loss_iou * world_size)

    def preprocess(self, batch_gt_instances, batch_size):
        targets_list = np.zeros((batch_size, 1, 5)).tolist()
        for i, item in enumerate(batch_gt_instances.cpu().numpy().tolist()):
            targets_list[int(item[0])].append(item[1:])
        max_len = max(len(target) for target in targets_list)
        targets = torch.from_numpy(
            np.array(
                list(
                    map(lambda l: l + [[-1, 0, 0, 0, 0]] * (max_len - len(l)),
                        targets_list)))[:,
                                        1:, :]).to(batch_gt_instances.device)
        return targets

    def varifocal_loss(self,
                       pred_score,
                       gt_score,
                       label,
                       alpha=0.75,
                       gamma=2.0):
        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy(
                pred_score.float(), gt_score.float(), reduction='none') *
                    weight).sum()
        return loss
