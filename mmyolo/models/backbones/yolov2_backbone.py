from typing import Union, List, Tuple

import torch
from torch import Tensor
import torch.nn as nn
from mmcv.cnn import ConvModule, MaxPool2d
from mmyolo.registry import MODELS
from mmdet.utils import ConfigType, OptMultiConfig, OptConfigType
from mmyolo.models import BaseBackbone
from mmyolo.utils import make_divisible, make_round
from mmengine.model import BaseModule, Sequential


@MODELS.register_module()
class YOLOv2Darknet(BaseBackbone):
    arch_settings = {
        'P5': [[64, 128, 3, ], [128, 256, 3, ],
               [256, 512, 3, ], [512, 1024, 5, ]]
    }
    
    def __init__(self,
                 arch: str = 'P5',
                 last_stage_out_channels: int = 1024,
                 plugins: Union[dict, List[dict]] = None,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 input_channels: int = 3,
                 out_indices: Tuple[int] = (4),
                 frozen_stages: int = -1,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 norm_eval: bool = False,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            self.arch_setting[arch],
            deepen_factor,
            widen_factor,
            input_channels=input_channels,
            out_indices=out_indices,
            plugins=plugins,
            frozen_stages=frozen_stages,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            norm_eval=norm_eval,
            init_cfg=init_cfg)
    
    def build_stem_layer(self) -> nn.Module:
        """Build a stem layer"""
        stage = []
        conv_layer1 = ConvModule(in_channels=self.input_channels,
                                 out_channels=32,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 norm_cfg=self.norm_cfg,
                                 act_cfg=self.act_cfg)
        stage.append(conv_layer1)
        maxpool1 = MaxPool2d(kernel_size=2, stride=2)
        stage.append(maxpool1)
        
        conv_layer2 = ConvModule(in_channels=32,
                                 out_channels=64,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 norm_cfg=self.norm_cfg,
                                 act_cfg=self.act_cfg)
        stage.append(conv_layer2)
        maxpool2 = MaxPool2d(kernel_size=2, stride=2)
        stage.append(maxpool2)
        
        return stage
    
    def build_stage_layer(self, stage_idx: int, setting: list) -> list:
        """Build a stage layer
        Args:
            stage_idx (int): The index of a stage layer.
            setting (last): The architecture setting of a stage layer.
        """
        in_channels, out_channels, num_blocks, add_identity, final = setting
        
        in_channels = make_divisible(in_channels, self.widen_factor)
        out_channels = make_divisible(out_channels, self.widen_factor)
        num_blocks = make_round(num_blocks, self.deepen_factor)
        
        stage = []
        if final:
            conv_layer = YOLOv2DarkNetLayer(in_channels,
                                            out_channels,
                                            norm_cfg=self.norm_cfg,
                                            act_cfg=self.act_cfg)
            stage.append(conv_layer)
            
            maxpool = MaxPool2d(kernel_size=2, stride=2)
            stage.append(maxpool)
        else:
            conv_layer = YOLOv2DarkNetLayer(in_channels,
                                            out_channels,
                                            norm_cfg=self.norm_cfg,
                                            act_cfg=self.act_cfg)
            stage.append(conv_layer)
            conv_layer2 = Sequential([ConvModule(out_channels,
                                                 in_channels,
                                                 norm_cfg=self.norm_cfg,
                                                 act_cfg=self.act_cfg),
                                      ConvModule(in_channels,
                                                 out_channels,
                                                 norm_cfg=self.norm_cfg,
                                                 act_cfg=self.act_cfg)])
            stage.append(conv_layer2)
            
        return stage

class YOLOv2DarkNetLayer(BaseModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        # expand_ratio: float = 0.5,
        # num_blocks: int = 1,
        # add_identity: bool = True,
        # use_depthwise: bool = False,
        # channel_attention: bool = False,
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(
            type='BN', momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type='Swish'),
        init_cfg: OptMultiConfig = None) -> None:
        
        super().__init__(init_cfg=init_cfg)
        
        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = ConvModule(
            in_channels=out_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv3 = ConvModule(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward function"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        return x
    
    def init_weights(self):
        """Initialize the parameters."""
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d):
                    # In order to be consistent with the source code,
                    # reset the Conv2d initialization parameters
                    m.reset_parameters()
        else:
            return super().init_weights()