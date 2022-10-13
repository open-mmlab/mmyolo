# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.models.utils import multi_apply
from mmdet.utils import (ConfigType, OptConfigType, OptInstanceList,
                         OptMultiConfig)
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor

from mmyolo.registry import MODELS
from ..utils import make_divisible
from .yolov5_head import YOLOv5Head


class ESEAttn(nn.Module):

    def __init__(self,
                 feat_channels,
                 norm_cfg=dict(type='BN', momentum=0.1, eps=1e-5),
                 act_cfg=dict(type='Swish')):
        super().__init__()
        self.fc = nn.Conv2d(feat_channels, feat_channels, 1)
        self.sig = nn.Sigmoid()
        self.conv = ConvModule(
            feat_channels,
            feat_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.fc.weight, mean=0, std=0.001)

    def forward(self, feat, avg_feat):
        weight = self.sig(self.fc(avg_feat))
        return self.conv(feat * weight)


@MODELS.register_module()
class PPYOLOEHeadModule(BaseModule):

    def __init__(self,
                 num_classes: int,
                 in_channels: Union[int, Sequence],
                 widen_factor: float = 1.0,
                 num_base_priors: int = 1,
                 featmap_strides: Sequence[int] = (8, 16, 32),
                 reg_max: int = 16,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.1, eps=1e-5),
                 act_cfg: ConfigType = dict(type='Swish'),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)

        self.num_classes = num_classes
        self.featmap_strides = featmap_strides
        self.num_levels = len(self.featmap_strides)
        self.num_base_priors = num_base_priors
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_channels = in_channels
        self.reg_max = reg_max

        in_channels = []
        for channel in self.in_channels:
            channel = make_divisible(channel, widen_factor)
            in_channels.append(channel)
        self.in_channels = in_channels

        self._init_layers()

    def init_weights(self, prior_prob=0.01):
        for conv in self.cls_preds:
            b = conv.bias.view(-1, )
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)

        for conv in self.reg_preds:
            b = conv.bias.view(-1, )
            b.data.fill_(1.0)
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)

    def _init_layers(self):
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.cls_stems = nn.ModuleList()
        self.reg_stems = nn.ModuleList()

        for in_c in self.in_channels:
            self.cls_stems.append(
                ESEAttn(in_c, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))
            self.reg_stems.append(
                ESEAttn(in_c, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))

        for in_c in self.in_channels:
            self.cls_preds.append(
                nn.Conv2d(in_c, self.num_classes, 3, padding=1))
            self.reg_preds.append(
                nn.Conv2d(in_c, 4 * (self.reg_max + 1), 3, padding=1))

        # TODO: use ConvModule
        self.proj_conv = nn.Conv2d(self.reg_max + 1, 1, 1, bias=False)
        self.proj = nn.Parameter(
            torch.linspace(0, self.reg_max, self.reg_max + 1),
            requires_grad=False)
        self.proj_conv.weight = torch.nn.Parameter(
            self.proj.view([1, self.reg_max + 1, 1, 1]).clone().detach(),
            requires_grad=False)

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

        return multi_apply(self.forward_single, x, self.cls_stems,
                           self.cls_preds, self.reg_stems, self.reg_preds)

    def forward_single(self, x: torch.Tensor, cls_stem: nn.ModuleList,
                       cls_pred: nn.ModuleList, reg_stem: nn.ModuleList,
                       reg_pred: nn.ModuleList) -> torch.Tensor:
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
        """"""
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
