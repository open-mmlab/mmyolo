# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.models.utils import multi_apply
from mmdet.utils import (ConfigType, OptConfigType, OptInstanceList,
                         OptMultiConfig)
from mmengine.model import BaseModule, bias_init_with_prob
from mmengine.structures import InstanceData
from torch import Tensor

from mmyolo.registry import MODELS
from .yolov5_head import YOLOv5Head
from ..utils import make_divisible


@MODELS.register_module()
class YOLOv8HeadModule(BaseModule):
    """YOLOv8HeadModule head module used in `YOLOv8`.

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
                 reg_max: int = 16,
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
        self.in_channels = in_channels
        self.reg_max = reg_max  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)

        in_channels = []
        for channel in self.in_channels:
            channel = make_divisible(channel, widen_factor)
            in_channels.append(channel)
        self.in_channels = in_channels

        self._init_layers()

    def init_weights(self):
        """Initialize weights of the head."""
        # Use prior in model initialization to improve stability
        super().init_weights()

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.reset_parameters()

        bias_init = bias_init_with_prob(0.01)
        for conv_cls in self.cls_preds:
            conv_cls.bias.data.fill_(bias_init)

    def _init_layers(self):
        """initialize conv layers in YOLOv8 head."""
        # Init decouple head
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()

        cls_out_channels = max((16, self.in_channels[0] // 4, self.reg_max * 4))
        reg_out_channels = max(self.in_channels[0], self.num_classes)

        for i in range(self.num_levels):
            self.cls_preds.append(nn.Sequential(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=cls_out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg),
                ConvModule(
                    in_channels=cls_out_channels,
                    out_channels=cls_out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg),
                nn.Conv2d(
                    in_channels=cls_out_channels,
                    out_channels=4 * self.reg_max,
                    kernel_size=1)
            ))
            self.reg_preds.append(nn.Sequential(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=reg_out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg),
                ConvModule(
                    in_channels=reg_out_channels,
                    out_channels=reg_out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg),
                nn.Conv2d(
                    in_channels=reg_out_channels,
                    out_channels=self.num_classes,
                    kernel_size=1)
            ))

        self.dfl = YOLOv8DistributionFocalLoss(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x) == self.num_levels
        return multi_apply(self.forward_single, x, self.cls_preds, self.reg_preds)

    def forward_single(self, x: torch.Tensor, cls_pred: nn.ModuleList, reg_pred: nn.ModuleList) -> torch.Tensor:
        """Forward feature of a single scale level."""
        y = stem(x)
        cls_x = y
        reg_x = y
        cls_feat = cls_conv(cls_x)
        reg_feat = reg_conv(reg_x)

        cls_score = cls_pred(cls_feat)
        bbox_pred = reg_pred(reg_feat)

        return cls_score, bbox_pred


# TODO Training mode is currently not supported
@MODELS.register_module()
class YOLOv8Head(YOLOv5Head):
    """YOLOv8Head head used in `YOLOv8.

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
                 loss_cls: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='sum',
                     loss_weight=1.0),
                 loss_bbox: ConfigType = dict(
                     type='mmdet.GIoULoss', reduction='sum', loss_weight=5.0),
                 loss_dfl=dict(
                     type='mmdet.DistributionFocalLoss',
                     reduction='mean',
                     loss_weight=0.5 / 4),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            head_module=head_module,
            prior_generator=prior_generator,
            bbox_coder=bbox_coder,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)

    def special_init(self):
        """Since YOLO series algorithms will inherit from YOLOv5Head, but
        different algorithms have special initialization process.

        The special_init function is designed to deal with this situation.
        """
        pass

    def loss_by_feat(
            self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            objectnesses: Sequence[Tensor],
            batch_gt_instances: Sequence[InstanceData],
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        raise NotImplementedError('Not implemented yet !')


class YOLOv8DistributionFocalLoss(nn.Module):
    def __init__(self, in_channel=16):
        super().__init__()
        self.in_channel = in_channel

        self.conv = nn.Conv2d(self.in_channel, 1, 1, bias=False).requires_grad_(False)
        weight_data = torch.arange(self.in_channel, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(weight_data.view(1, self.in_channel, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, anchors = x.shape
        return self.conv(
            x.view(batch, 4, self.in_channel, anchors).transpose(2, 1).softmax(1)
        ).view(batch, 4, anchors)
