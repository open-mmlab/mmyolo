# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.models.backbones import BaseBackbone
from mmyolo.models.layers.yolo_bricks import EffectiveSELayer, RepVGGBlock
from mmyolo.models.utils import make_divisible, make_round
from mmyolo.registry import MODELS


class CSPResStage(nn.Module):
    """PPYOLOE Backbone Stage."""

    def __init__(self,
                 block_fn: nn.Module,
                 input_channels: int,
                 output_channels: int,
                 num_layer: int,
                 stride: int = 1,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='Swish'),
                 use_attn: bool = True,
                 use_alpha: bool = False):
        super().__init__()
        middle_channels = (input_channels + output_channels) // 2
        if stride == 2:
            self.conv_down = ConvModule(
                input_channels,
                middle_channels,
                3,
                stride=2,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.conv_down = None
        self.conv1 = ConvModule(
            middle_channels,
            middle_channels // 2,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = ConvModule(
            middle_channels,
            middle_channels // 2,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.blocks = nn.Sequential(*[
            block_fn(
                middle_channels // 2,
                middle_channels // 2,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                shortcut=True,
                use_alpha=use_alpha) for i in range(num_layer)
        ])
        if use_attn:
            self.attn = EffectiveSELayer(
                middle_channels, act_cfg=dict(type='HSigmoid'))
        else:
            self.attn = None

        self.conv3 = ConvModule(
            middle_channels,
            output_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv_down is not None:
            x = self.conv_down(x)
        y1 = self.conv1(x)
        y2 = self.blocks(self.conv2(x))
        y = torch.cat([y1, y2], axis=1)
        if self.attn is not None:
            y = self.attn(y)
        y = self.conv3(y)
        return y


class BasicBlock(nn.Module):
    """PPYOLOE Backbone BasicBlock."""

    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.1, eps=1e-5),
                 act_cfg: ConfigType = dict(type='Swish'),
                 shortcut: bool = True,
                 use_alpha: bool = False):
        super().__init__()
        assert input_channels == output_channels
        assert act_cfg is None or isinstance(act_cfg, dict)
        self.conv1 = ConvModule(
            input_channels,
            output_channels,
            3,
            stride=1,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.conv2 = RepVGGBlock(
            output_channels,
            output_channels,
            alpha=use_alpha,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            mode='ppyoloe')
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.conv2(y)
        if self.shortcut:
            return x + y
        else:
            return y


@MODELS.register_module()
class CSPResNet(BaseBackbone):
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
        norm_cfg (:obj:`ConfigDict` or dict): Dictionary to construct and
            config norm layer. Defaults to dict(type='BN', requires_grad=True).
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Defaults to dict(type='SiLU').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`]): Initialization config dict.
        use_large_stem (bool): Whether to use large stem layer.
            Defaults to False.
        use_alpha (bool): Whether to use alpha in `RepVGGBlock`. PPYOLOE do
            not use, but PPYOLOE+ use.
            Defaults to False.
    """
    # From left to right:
    # in_channels, out_channels, num_blocks, add_identity, use_spp
    arch_settings = {
        'P5': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, 1024, 3, True, False]]
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
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.1, eps=1e-5),
                 act_cfg: ConfigType = dict(type='Swish'),
                 norm_eval: bool = False,
                 init_cfg: OptMultiConfig = None,
                 use_large_stem: bool = False,
                 use_alpha: bool = False):
        arch_setting = self.arch_settings[arch]
        if arch_ovewrite:
            arch_setting = arch_ovewrite
        self.use_large_stem = use_large_stem
        self.use_alpha = use_alpha
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
                    make_divisible(self.arch_setting[0][0], self.widen_factor)
                    // 2,
                    3,
                    stride=2,
                    padding=1,
                    act_cfg=self.act_cfg,
                    norm_cfg=self.norm_cfg),
                ConvModule(
                    make_divisible(self.arch_setting[0][0],
                                   self.widen_factor) // 2,
                    make_divisible(self.arch_setting[0][0], self.widen_factor)
                    // 2,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg),
                ConvModule(
                    make_divisible(self.arch_setting[0][0], self.widen_factor)
                    // 2,
                    make_divisible(self.arch_setting[0][0], self.widen_factor),
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        else:
            stem = nn.Sequential(
                ConvModule(
                    self.input_channels,
                    make_divisible(self.arch_setting[0][0], self.widen_factor)
                    // 2,
                    3,
                    stride=2,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg),
                ConvModule(
                    make_divisible(self.arch_setting[0][0], self.widen_factor)
                    // 2,
                    make_divisible(self.arch_setting[0][0], self.widen_factor),
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
        in_channels, out_channels, num_blocks, add_identity, _ = setting
        in_channels = make_divisible(in_channels, self.widen_factor)
        out_channels = make_divisible(out_channels, self.widen_factor)
        num_blocks = make_round(num_blocks, self.deepen_factor)

        cspres_layer = CSPResStage(
            BasicBlock,
            in_channels,
            out_channels,
            num_blocks,
            2,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            use_alpha=self.use_alpha)
        return [
            cspres_layer,
        ]
