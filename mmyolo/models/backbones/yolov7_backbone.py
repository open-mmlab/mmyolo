# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.models.backbones.csp_darknet import Focus
from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.registry import MODELS
from ..layers import MaxPoolAndStrideConvBlock
from .base_backbone import BaseBackbone


@MODELS.register_module()
class YOLOv7Backbone(BaseBackbone):
    """Backbone used in YOLOv7.

    Args:
        arch (str): Architecture of YOLOv7Defaults to L.
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
    _tiny_stage1_cfg = dict(type='TinyDownSampleBlock', middle_ratio=0.5)
    _tiny_stage2_4_cfg = dict(type='TinyDownSampleBlock', middle_ratio=1.0)
    _l_expand_channel_2x = dict(
        type='ELANBlock',
        middle_ratio=0.5,
        block_ratio=0.5,
        num_blocks=2,
        num_convs_in_block=2)
    _l_no_change_channel = dict(
        type='ELANBlock',
        middle_ratio=0.25,
        block_ratio=0.25,
        num_blocks=2,
        num_convs_in_block=2)
    _x_expand_channel_2x = dict(
        type='ELANBlock',
        middle_ratio=0.4,
        block_ratio=0.4,
        num_blocks=3,
        num_convs_in_block=2)
    _x_no_change_channel = dict(
        type='ELANBlock',
        middle_ratio=0.2,
        block_ratio=0.2,
        num_blocks=3,
        num_convs_in_block=2)
    _w_no_change_channel = dict(
        type='ELANBlock',
        middle_ratio=0.5,
        block_ratio=0.5,
        num_blocks=2,
        num_convs_in_block=2)
    _e_no_change_channel = dict(
        type='ELANBlock',
        middle_ratio=0.4,
        block_ratio=0.4,
        num_blocks=3,
        num_convs_in_block=2)
    _d_no_change_channel = dict(
        type='ELANBlock',
        middle_ratio=1 / 3,
        block_ratio=1 / 3,
        num_blocks=4,
        num_convs_in_block=2)
    _e2e_no_change_channel = dict(
        type='EELANBlock',
        num_elan_block=2,
        middle_ratio=0.4,
        block_ratio=0.4,
        num_blocks=3,
        num_convs_in_block=2)

    # From left to right:
    # in_channels, out_channels, Block_params
    arch_settings = {
        'Tiny': [[64, 64, _tiny_stage1_cfg], [64, 128, _tiny_stage2_4_cfg],
                 [128, 256, _tiny_stage2_4_cfg],
                 [256, 512, _tiny_stage2_4_cfg]],
        'L': [[64, 256, _l_expand_channel_2x],
              [256, 512, _l_expand_channel_2x],
              [512, 1024, _l_expand_channel_2x],
              [1024, 1024, _l_no_change_channel]],
        'X': [[80, 320, _x_expand_channel_2x],
              [320, 640, _x_expand_channel_2x],
              [640, 1280, _x_expand_channel_2x],
              [1280, 1280, _x_no_change_channel]],
        'W':
        [[64, 128, _w_no_change_channel], [128, 256, _w_no_change_channel],
         [256, 512, _w_no_change_channel], [512, 768, _w_no_change_channel],
         [768, 1024, _w_no_change_channel]],
        'E':
        [[80, 160, _e_no_change_channel], [160, 320, _e_no_change_channel],
         [320, 640, _e_no_change_channel], [640, 960, _e_no_change_channel],
         [960, 1280, _e_no_change_channel]],
        'D': [[96, 192,
               _d_no_change_channel], [192, 384, _d_no_change_channel],
              [384, 768, _d_no_change_channel],
              [768, 1152, _d_no_change_channel],
              [1152, 1536, _d_no_change_channel]],
        'E2E': [[80, 160, _e2e_no_change_channel],
                [160, 320, _e2e_no_change_channel],
                [320, 640, _e2e_no_change_channel],
                [640, 960, _e2e_no_change_channel],
                [960, 1280, _e2e_no_change_channel]],
    }

    def __init__(self,
                 arch: str = 'L',
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 input_channels: int = 3,
                 out_indices: Tuple[int] = (2, 3, 4),
                 frozen_stages: int = -1,
                 plugins: Union[dict, List[dict]] = None,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 norm_eval: bool = False,
                 init_cfg: OptMultiConfig = None):
        assert arch in self.arch_settings.keys()
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
                    int(self.arch_setting[0][0] * self.widen_factor // 2),
                    int(self.arch_setting[0][0] * self.widen_factor),
                    3,
                    padding=1,
                    stride=2,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        elif self.arch in ['W', 'E', 'D', 'E2E']:
            stem = Focus(
                3,
                int(self.arch_setting[0][0] * self.widen_factor),
                kernel_size=3,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
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

        stage_block_cfg['in_channels'] = in_channels
        stage_block_cfg['out_channels'] = out_channels

        stage = []
        if self.arch in ['W', 'E', 'D', 'E2E']:
            stage_block_cfg['in_channels'] = out_channels
        elif self.arch in ['L', 'X']:
            if stage_idx == 0:
                stage_block_cfg['in_channels'] = out_channels // 2

        downsample_layer = self._build_downsample_layer(
            stage_idx, in_channels, out_channels)
        stage.append(MODELS.build(stage_block_cfg))
        if downsample_layer is not None:
            stage.insert(0, downsample_layer)
        return stage

    def _build_downsample_layer(self, stage_idx: int, in_channels: int,
                                out_channels: int) -> Optional[nn.Module]:
        """Build a downsample layer pre stage."""
        if self.arch in ['E', 'D', 'E2E']:
            downsample_layer = MaxPoolAndStrideConvBlock(
                in_channels,
                out_channels,
                use_in_channels_of_middle=True,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        elif self.arch == 'W':
            downsample_layer = ConvModule(
                in_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        elif self.arch == 'Tiny':
            if stage_idx != 0:
                downsample_layer = nn.MaxPool2d(2, 2)
            else:
                downsample_layer = None
        elif self.arch in ['L', 'X']:
            if stage_idx == 0:
                downsample_layer = ConvModule(
                    in_channels,
                    out_channels // 2,
                    3,
                    stride=2,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)
            else:
                downsample_layer = MaxPoolAndStrideConvBlock(
                    in_channels,
                    in_channels,
                    use_in_channels_of_middle=False,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)
        return downsample_layer
