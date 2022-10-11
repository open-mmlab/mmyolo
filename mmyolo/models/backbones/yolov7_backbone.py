# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.registry import MODELS
from ..layers import SPPCSPBlock, ELANBlock, MaxPoolBlock
from .base_backbone import BaseBackbone


@MODELS.register_module()
class YOLOv7Backbone(BaseBackbone):
    # From left to right:
    # in_channels, out_channels, num_blocks, type
    arch_settings = {
        'P5': [[64, 128, 2, 'type1'], [256, 512, 2, 'type1'],
               [512, 1024, 2, 'type1'], [1024, 1024, 2, 'type2']]
    }

    def __init__(self,
                 arch: str = 'P5',
                 plugins: Union[dict, List[dict]] = None,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 input_channels: int = 3,
                 out_indices: Tuple[int] = (2, 3, 4),
                 frozen_stages: int = -1,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 norm_eval: bool = False,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            self.arch_settings[arch],
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
        """Build a stem layer."""
        """Build a stem layer."""
        stem = nn.Sequential(
            ConvModule(
                3,
                int(self.arch_setting[0][0] * self.widen_factor // 2),
                3,
                padding=1,
                stride=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                int(self.arch_setting[0][0] * self.widen_factor // 2),
                int(self.arch_setting[0][0] * self.widen_factor),
                3,
                padding=1,
                stride=2,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                int(self.arch_setting[0][0] * self.widen_factor),
                int(self.arch_setting[0][0] * self.widen_factor),
                3,
                padding=1,
                stride=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        return stem

    def build_stage_layer(self, stage_idx: int, setting: list) -> list:
        """Build a stage layer.

        Args:
            stage_idx (int): The index of a stage layer.
            setting (list): The architecture setting of a stage layer.
        """
        in_channels, out_channels, num_blocks, mode_type = setting

        in_channels = int(in_channels * self.widen_factor)
        out_channels = int(out_channels * self.widen_factor)

        stage = []
        if stage_idx == 0:
            pre_layer = ConvModule(
                in_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            elan_layer = ELANBlock(
                out_channels,
                mode_type,
                num_blocks=num_blocks,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            stage.extend([pre_layer, elan_layer])
        else:
            pre_layer = MaxPoolBlock(in_channels,
                                     norm_cfg=self.norm_cfg,
                                     act_cfg=self.act_cfg)
            elan_layer = ELANBlock(
                in_channels,
                mode_type,
                num_blocks=num_blocks,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            stage.extend([pre_layer, elan_layer])
        return stage
