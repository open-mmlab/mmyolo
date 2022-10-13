# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.models.backbones import BaseBackbone
from mmyolo.models.layers.yolo_bricks import EffectiveSELayer, RepVGGBlock
from mmyolo.models.utils import make_divisible, make_round
from mmyolo.registry import MODELS


class CSPResStage(nn.Module):

    def __init__(self,
                 block_fn,
                 ch_in,
                 ch_out,
                 n,
                 stride,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 use_attn=True,
                 use_alpha=False):
        super().__init__()
        ch_mid = (ch_in + ch_out) // 2
        if stride == 2:
            self.conv_down = ConvModule(
                ch_in,
                ch_mid,
                3,
                stride=2,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.conv_down = None
        self.conv1 = ConvModule(
            ch_mid, ch_mid // 2, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvModule(
            ch_mid, ch_mid // 2, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.blocks = nn.Sequential(*[
            block_fn(
                ch_mid // 2,
                ch_mid // 2,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                shortcut=True,
                use_alpha=use_alpha) for i in range(n)
        ])
        if use_attn:
            self.attn = EffectiveSELayer(ch_mid, act_cfg=dict(type='HSigmoid'))
        else:
            self.attn = None

        self.conv3 = ConvModule(
            ch_mid, ch_out, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
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

    def __init__(self,
                 ch_in,
                 ch_out,
                 norm_cfg=dict(type='BN', momentum=0.1, eps=1e-5),
                 act_cfg=dict(type='Swish'),
                 shortcut=True,
                 use_alpha=False):
        super().__init__()
        assert ch_in == ch_out
        assert act_cfg is None or isinstance(act_cfg, dict)
        self.conv1 = ConvModule(
            ch_in,
            ch_out,
            3,
            stride=1,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.conv2 = RepVGGBlock(
            ch_out,
            ch_out,
            alpha=use_alpha,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            mode='ppyoloe')
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.shortcut:
            return x + y
        else:
            return y


@MODELS.register_module()
class CSPResNet(BaseBackbone):
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
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.1, eps=1e-5),
                 act_cfg: ConfigType = dict(type='Swish'),
                 norm_eval: bool = False,
                 init_cfg: OptMultiConfig = None,
                 use_large_stem: bool = False,
                 use_alpha=False):
        self.use_large_stem = use_large_stem
        self.use_alpha = use_alpha
        super().__init__(
            self.arch_settings[arch],
            deepen_factor,
            widen_factor,
            input_channels=input_channels,
            out_indices=out_indices,
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
        in_channels, out_channels, num_blocks, add_identity, _ = setting
        in_channels = make_divisible(in_channels, self.widen_factor)
        out_channels = make_divisible(out_channels, self.widen_factor)
        num_blocks = make_round(num_blocks, self.deepen_factor)

        stage = []
        cspres_layer = CSPResStage(
            BasicBlock,
            in_channels,
            out_channels,
            num_blocks,
            2,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            use_alpha=self.use_alpha)
        stage.append(cspres_layer)
        return stage

    def init_weights(self):
        raise NotImplementedError
