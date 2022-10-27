# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.models.backbones import BaseBackbone
from mmyolo.models.layers.yolo_bricks import CSPResLayer
from mmyolo.registry import MODELS


@MODELS.register_module()
class PPYOLOECSPResNet(BaseBackbone):
    """CSP-ResNet backbone used in PPYOLOE.

    Args:
        arch (str): Architecture of CSPNeXt, from {P5, P6}.
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
        arch_ovewrite (list): Overwrite default arch settings.
            Defaults to None.
        block_cfg (dict): Config dict for block. Defaults to
            dict(type='PPYOLOEBasicBlock', shortcut=True, use_alpha=True)
        norm_cfg (:obj:`ConfigDict` or dict): Dictionary to construct and
            config norm layer. Defaults to dict(type='BN', momentum=0.1,
            eps=1e-5).
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        attention_cfg (dict): Config dict for `EffectiveSELayer`.
            Defaults to dict(type='EffectiveSELayer',
            act_cfg=dict(type='HSigmoid')).
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`]): Initialization config dict.
        use_large_stem (bool): Whether to use large stem layer.
            Defaults to False.
    """
    # From left to right:
    # in_channels, out_channels, num_blocks
    arch_settings = {
        'P5': [[64, 128, 3], [128, 256, 6], [256, 512, 6], [512, 1024, 3]]
    }

    def __init__(self,
                 arch: str = 'P5',
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 input_channels: int = 3,
                 out_indices: Tuple[int] = (2, 3, 4),
                 frozen_stages: int = -1,
                 plugins: Union[dict, List[dict]] = None,
                 arch_ovewrite: dict = None,
                 block_cfg: ConfigType = dict(
                     type='PPYOLOEBasicBlock', shortcut=True, use_alpha=True),
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.1, eps=1e-5),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 attention_cfg: ConfigType = dict(
                     type='EffectiveSELayer', act_cfg=dict(type='HSigmoid')),
                 norm_eval: bool = False,
                 init_cfg: OptMultiConfig = None,
                 use_large_stem: bool = False):
        arch_setting = self.arch_settings[arch]
        if arch_ovewrite:
            arch_setting = arch_ovewrite
        arch_setting = [[
            int(in_channels * widen_factor),
            int(out_channels * widen_factor),
            round(num_blocks * deepen_factor)
        ] for in_channels, out_channels, num_blocks in arch_setting]
        self.block_cfg = block_cfg
        self.use_large_stem = use_large_stem
        self.attention_cfg = attention_cfg

        super().__init__(
            arch_setting,
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
        if self.use_large_stem:
            stem = nn.Sequential(
                ConvModule(
                    self.input_channels,
                    self.arch_setting[0][0] // 2,
                    3,
                    stride=2,
                    padding=1,
                    act_cfg=self.act_cfg,
                    norm_cfg=self.norm_cfg),
                ConvModule(
                    self.arch_setting[0][0] // 2,
                    self.arch_setting[0][0] // 2,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg),
                ConvModule(
                    self.arch_setting[0][0] // 2,
                    self.arch_setting[0][0],
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        else:
            stem = nn.Sequential(
                ConvModule(
                    self.input_channels,
                    self.arch_setting[0][0] // 2,
                    3,
                    stride=2,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg),
                ConvModule(
                    self.arch_setting[0][0] // 2,
                    self.arch_setting[0][0],
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        return stem

    def build_stage_layer(self, stage_idx: int, setting: list) -> list:
        """Build a stage layer.

        Args:
            stage_idx (int): The index of a stage layer.
            setting (list): The architecture setting of a stage layer.
        """
        in_channels, out_channels, num_blocks = setting

        cspres_layer = CSPResLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            num_block=num_blocks,
            block_cfg=self.block_cfg,
            stride=2,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            attention_cfg=self.attention_cfg,
            use_spp=False)
        return [cspres_layer]
