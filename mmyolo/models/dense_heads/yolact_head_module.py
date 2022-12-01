# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple, Union
from mmcv.cnn import ConvModule
import torch.nn as nn
from mmdet.utils import (ConfigType, OptConfigType, OptInstanceList,
                         OptMultiConfig)
from mmengine.model import BaseModule

from torch import Tensor

from mmyolo.registry import MODELS


class Proto(BaseModule):
    def __init__(self,
                 in_channels,
                 proto_channels=256,
                 num_protos=32,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg=None):
        super().__init__(init_cfg)
        self.cv1 = ConvModule(in_channels, proto_channels, kernel_size=3, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = ConvModule(proto_channels, proto_channels, kernel_size=3, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.cv3 = ConvModule(proto_channels, num_protos, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


@MODELS.register_module()
class YOLACTHeadModule(BaseModule):
    def __init__(self,
                 bbox_head_module,
                 num_protos=32,
                 proto_channels=512,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg=None):
        super().__init__(init_cfg)
        self.bbox_head_module = MODELS.build(bbox_head_module)
        in_channels = self.bbox_head_module.in_channels[0]
        self.proto = Proto(in_channels, proto_channels, num_protos, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        assert len(x) == self.bbox_head_module.num_levels
        cls_scores, bbox_preds, objectnesses = self.bbox_head_module(x)
        segm_preds = self.proto(x[0])
        return cls_scores, bbox_preds, objectnesses, segm_preds

