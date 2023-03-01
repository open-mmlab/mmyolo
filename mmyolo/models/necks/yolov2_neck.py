from abc import ABCMeta
from typing import Union, List

import torch
import torch.nn as nn

from mmdet.utils import ConfigType, OptMultiConfig
from mmyolo.registry import MODELS
from mmengine.model import BaseModule
from mmcv.cnn import ConvModule

class Reorg(nn.Module):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride
    def forward(self, x):
        stride = self.stride
        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert(H % stride == 0)
        assert(W % stride == 0)
        ws = stride
        hs = stride
        x = x.view(B, C, H//hs, hs, W//ws, ws).transpose(3,4).contiguous()
        x = x.view(B, C, H//hs*W//ws, hs*ws).transpose(2,3).contiguous()
        x = x.view(B, C, hs*ws, H//hs, W//ws).transpose(1,2).contiguous()
        x = x.view(B, hs*ws*C, H//hs, W//ws)
        return x


@MODELS.register_module()
class YOLOv2Neck(BaseModule, metaclass=ABCMeta):
    """YOLOv2Neck used in YOLO6D"""
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
        self.pass_through_layer = self.build_passthrough_layer()
        self.out_layer = self.build_out_layer()
    
    def build_out_layer(self):
        return ConvModule(in_channels=1280, 
                          out_channels=1024,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          norm_cfg=self.norm_cfg,
                          act_cfg=self.act_cfg)

    def build_passthrough_layer(self):
        """build through layer"""
        return nn.Sequential(ConvModule(in_channels=512, 
                                        out_channels=64,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        norm_cfg=self.norm_cfg,
                                        act_cfg=self.act_cfg),
                             Reorg())

    def forward(self, inputs: List[torch.Tensor]) -> tuple:
        """Forward function"""
        assert len(inputs) == len(self.in_channels)
        
        # Passthrough layer
        pass_through_out = self.pass_through_layer(input[0])
        result = torch.cat(pass_through_out, 
                                     input[1])
        
        # Out layer
        result = self.out_layer(result)

        return result
    
    def train(self, mode=True):
        """Convert the model into training mode while keep the normalization
        layer freezed."""
        super().train(mode)
        if self.freeze_all:
            self._freeze_all()
