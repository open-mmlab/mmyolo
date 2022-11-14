# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.registry import MODELS
from ..layers import MaxPoolAndStrideConvBlock
from .base_backbone import BaseBackbone


@MODELS.register_module()
class YOLOv7Backbone(BaseBackbone):
    """Backbone used in YOLOv7.

    Args:
        arch (str): Architecture of YOLOv7, from {P5, P6}.
            Defaults to P5.
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        out_indices (Sequence[int]): Output from which stages.
            Defaults to (2, 3, 4).
        frozen_stages (int): Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters. Defaults to -1.
        plugins (list[dict]): List of plugins for stages, each dict contains:

            - cfg (dict, required): Cfg dict to build plugin.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        norm_cfg (:obj:`ConfigDict` or dict): Dictionary to construct and
            config norm layer. Defaults to dict(type='BN', requires_grad=True).
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`]): Initialization config dict.
    """
    _tiny_stage1_cfg = dict(
        type='TinyDownsampleConv',
        with_maxpool=False
    )
    _tiny_stage2_4_cfg = dict(
        type='TinyDownsampleConv',
        with_maxpool=True
    )

    _l_expand_channel_2x = dict(
        type='ELANBlock',
        mid_ratio=0.5,
        block_ratio=0.5,
        out_ratio=2.0,
        num_blocks=2,
        num_convs_in_block=2)
    _l_no_change_channel = dict(
        type='ELANBlock',
        mid_ratio=0.25,
        block_ratio=0.25,
        out_ratio=1.0,
        num_blocks=2,
        num_convs_in_block=2)

    _x_expand_channel_2x = dict(
        type='ELANBlock',
        mid_ratio=0.4,
        block_ratio=0.4,
        out_ratio=2.0,
        num_blocks=3,
        num_convs_in_block=2)
    _x_no_change_channel = dict(
        type='ELANBlock',
        mid_ratio=0.2,
        block_ratio=0.2,
        out_ratio=1.0,
        num_blocks=3,
        num_convs_in_block=2)

    # From left to right:
    # in_channels, out_channels, Block_params
    arch_settings = {
        'Tiny': [[64, 64, _tiny_stage1_cfg],
                 [64, 128, _tiny_stage2_4_cfg],
                 [128, 256, _tiny_stage2_4_cfg],
                 [256, 512, _tiny_stage2_4_cfg]],
        'L': [[64, 128, _l_expand_channel_2x],
              [256, 512, _l_expand_channel_2x],
              [512, 1024, _l_expand_channel_2x],
              [1024, 1024, _l_no_change_channel]],
        'X': [[80, 160, _x_expand_channel_2x],
              [320, 640, _x_expand_channel_2x],
              [640, 1280, _x_expand_channel_2x],
              [1280, 1280, _x_no_change_channel]]
    }

    def __init__(self,
                 arch: str = 'L',
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
        self.arch = arch
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
        if self.arch in ['L', 'X']:
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
        elif self.arch == 'Tiny':
            stem = nn.Sequential(
                ConvModule(
                    3,
                    int(self.arch_setting[0][0] * self.widen_factor // 2),
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
                    stride=2,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        return stem

    def build_stage_layer(self, stage_idx: int, setting: list) -> list:
        """Build a stage layer.

        Args:
            stage_idx (int): The index of a stage layer.
            setting (list): The architecture setting of a stage layer.
        """
        in_channels, out_channels, stage_block_cfg = setting
        in_channels = int(in_channels * self.widen_factor)
        out_channels = int(out_channels * self.widen_factor)

        stage_block_cfg = stage_block_cfg.copy()
        stage_block_cfg.setdefault('norm_cfg', self.norm_cfg)
        stage_block_cfg.setdefault('act_cfg', self.act_cfg)

        stage = []
        if self.arch == 'Tiny':
            stage_block_cfg['in_channels'] = in_channels
            stage_block_cfg['out_channels'] = out_channels
            stage.append(MODELS.build(stage_block_cfg))
        else:
            if stage_idx == 0:
                pre_layer = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)
                stage_block_cfg['in_channels'] = out_channels
                stage.extend([pre_layer, MODELS.build(stage_block_cfg)])
            else:
                pre_layer = MaxPoolAndStrideConvBlock(
                    in_channels,
                    mode='reduce_channel_2x',
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)
                stage_block_cfg['in_channels'] = in_channels
                stage.extend([pre_layer, MODELS.build(stage_block_cfg)])
        return stage
