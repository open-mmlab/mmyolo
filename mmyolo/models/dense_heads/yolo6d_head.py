from abc import ABCMeta
from typing import Union, List

import torch
import torch.nn as nn

from mmdet.utils import ConfigType, OptMultiConfig
from mmyolo.registry import MODELS
from mmengine.model import BaseModule
from mmcv.cnn import ConvModule

@MODELS.register_module()
class YOLO6DHead(nn.Module, metaclass=ABCMeta):
    """YOLO6DHead used in YOLO6D"""
    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[int, List[int]],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 freeze_all: bool = False,
                 norm_cfg: ConfigType = None,
                 act_cfg: ConfigType = None,
                 init_cfg: OptMultiConfig = None,
    ):
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deepen_factor = deepen_factor
        self.widen_factor = widen_factor
        self.freeze_all = freeze_all
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        
    def build_layer(self):
        conv1 = ConvModule(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        
        
    
    def forward(self, inputs: List[torch.Tensor]) -> tuple:
        """Forward function"""
        
        