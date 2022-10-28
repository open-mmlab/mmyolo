# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.utils import multi_apply
from mmdet.utils import (ConfigType, OptConfigType, OptInstanceList,
                         OptMultiConfig)
from mmengine.model import BaseModule, bias_init_with_prob
from mmengine.structures import InstanceData
from torch import Tensor

from mmyolo.registry import MODELS
from ..layers.yolo_bricks import PPYOLOESELayer
from .yolov5_head import YOLOv5Head


@MODELS.register_module()
class PPYOLOEHeadModule(BaseModule):
    """PPYOLOEHead head module used in `PPYOLOE`

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Default: 1.0.
        num_base_priors:int: The number of priors (points) at a point
            on the feature grid.
        featmap_strides (Sequence[int]): Downsample factor of each feature map.
             Defaults to (8, 16, 32).
        reg_max (int): TOOD reg_max param.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: Union[int, Sequence],
                 widen_factor: float = 1.0,
                 num_base_priors: int = 1,
                 featmap_strides: Sequence[int] = (8, 16, 32),
                 reg_max: int = 16,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.1, eps=1e-5),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)

        self.num_classes = num_classes
        self.featmap_strides = featmap_strides
        self.num_levels = len(self.featmap_strides)
        self.num_base_priors = num_base_priors
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.reg_max = reg_max

        if isinstance(in_channels, int):
            self.in_channels = [int(in_channels * widen_factor)
                                ] * self.num_levels
        else:
            self.in_channels = [int(i * widen_factor) for i in in_channels]

        self._init_layers()

    def init_weights(self, prior_prob=0.01):
        """Initialize the weight and bias of PPYOLOE head."""
        super().init_weights()
        for conv in self.cls_preds:
            conv.bias.data.fill_(bias_init_with_prob(prior_prob))
            conv.weight.data.fill_(0.)

        for conv in self.reg_preds:
            conv.bias.data.fill_(1.0)
            conv.weight.data.fill_(0.)

    def _init_layers(self):
        """initialize conv layers in PPYOLOE head."""
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.cls_stems = nn.ModuleList()
        self.reg_stems = nn.ModuleList()

        for in_channel in self.in_channels:
            self.cls_stems.append(
                PPYOLOESELayer(
                    in_channel, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))
            self.reg_stems.append(
                PPYOLOESELayer(
                    in_channel, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))

        for in_channel in self.in_channels:
            self.cls_preds.append(
                nn.Conv2d(in_channel, self.num_classes, 3, padding=1))
            self.reg_preds.append(
                nn.Conv2d(in_channel, 4 * (self.reg_max + 1), 3, padding=1))

        self.proj_conv = nn.Conv2d(self.reg_max + 1, 1, 1, bias=False)
        self.proj = nn.Parameter(
            torch.linspace(0, self.reg_max, self.reg_max + 1),
            requires_grad=False)
        self.proj_conv.weight = nn.Parameter(
            self.proj.view([1, self.reg_max + 1, 1, 1]).clone().detach(),
            requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions.
        """
        assert len(x) == self.num_levels

        return multi_apply(self.forward_single, x, self.cls_stems,
                           self.cls_preds, self.reg_stems, self.reg_preds)

    def forward_single(self, x: Tensor, cls_stem: nn.ModuleList,
                       cls_pred: nn.ModuleList, reg_stem: nn.ModuleList,
                       reg_pred: nn.ModuleList) -> Tensor:
        """Forward feature of a single scale level."""
        b, _, h, w = x.shape
        hw = h * w
        avg_feat = F.adaptive_avg_pool2d(x, (1, 1))
        cls_logit = cls_pred(cls_stem(x, avg_feat) + x)
        reg_dist = reg_pred(reg_stem(x, avg_feat))
        reg_dist = reg_dist.reshape([-1, 4, self.reg_max + 1,
                                     hw]).permute(0, 2, 3, 1)
        reg_dist = self.proj_conv(F.softmax(reg_dist, dim=1))

        return cls_logit, reg_dist


@MODELS.register_module()
class PPYOLOEHead(YOLOv5Head):
    """PPYOLOEHead head used in `PPYOLOE`.

    Args:
        head_module(nn.Module): Base module used for YOLOv5Head
        prior_generator(dict): Points generator feature maps in
            2D points-based detectors.
        bbox_coder (:obj:`ConfigDict` or dict): Config of bbox coder.
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
                 loss_cls: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='sum',
                     loss_weight=1.0),
                 loss_bbox: ConfigType = dict(
                     type='mmdet.GIoULoss', reduction='sum', loss_weight=5.0),
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

    def special_init(self):
        """Not Implenented."""
        pass

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
        raise NotImplementedError('Not implemented yet！')
