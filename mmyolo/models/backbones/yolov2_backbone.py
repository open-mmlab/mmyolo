from typing import Union, List, Tuple

import torch.nn as nn
from mmcv.cnn import ConvModule, MaxPool2d
from mmyolo.registry import MODELS
from mmdet.utils import ConfigType, OptMultiConfig
from mmyolo.models import BaseBackbone
from mmyolo.utils import make_divisible, make_round


@MODELS.register_module()
class YOLOv2Darknet(BaseBackbone):
    arch_settings = {
        'P6': [[32, 64, 1], [64, 128, 3], 
               [128, 256, 3], [256, 512, 5],
               [512, 1024, 7]]}
    
    def __init__(self,
                 arch: str = 'P6',
                 plugins: Union[dict, List[dict]] = None,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 input_channels: int = 3,
                 out_indices: Tuple[int] = (4,5),
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
        conv_layer = ConvModule(in_channels=self.input_channels,
                                out_channels=32,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                norm_cfg=self.norm_cfg,
                                act_cfg=self.act_cfg)
        stage.append(conv_layer)
        
        return stage
    
    def build_stage_layer(self, stage_idx: int, setting: list) -> list:
        """Build a stage layer
        Args:
            stage_idx (int): The index of a stage layer.
            setting (last): The architecture setting of a stage layer.
        """
        in_channels, out_channels, num_blocks = setting
        
        in_channels = make_divisible(in_channels, self.widen_factor)
        out_channels = make_divisible(out_channels, self.widen_factor)
        num_blocks = make_round(num_blocks, self.deepen_factor)
        
        stage = []
        stage.append(MaxPool2d(kernel_size=2, stride=2))
        
        if num_blocks == 1:
            stage.append(ConvModule(in_channels,
                                    out_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg))
        elif num_blocks == 3:
            stage.append(ConvModule(in_channels,
                                    out_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg),
                         ConvModule(out_channels,
                                    in_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=1,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg),
                         ConvModule(in_channels,
                                    out_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg))
        elif num_blocks == 5:
            stage.append(ConvModule(in_channels,
                                    out_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg),
                         ConvModule(out_channels,
                                    in_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=1,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg),
                         ConvModule(in_channels,
                                    out_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg),
                         ConvModule(out_channels,
                                    in_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=1,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg),
                         ConvModule(in_channels,
                                    out_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg))
        elif num_blocks == 7:
            stage.append(ConvModule(in_channels,
                                    out_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg),
                         ConvModule(out_channels,
                                    in_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=1,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg),
                         ConvModule(in_channels,
                                    out_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg),
                         ConvModule(out_channels,
                                    in_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=1,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg),
                         ConvModule(in_channels,
                                    out_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg),
                         ConvModule(out_channels,
                                    out_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg),
                         ConvModule(out_channels,
                                    out_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg))
        return stage
