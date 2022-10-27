# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.utils import ConfigType, OptMultiConfig
from torch import Tensor

from mmyolo.models.backbones import BaseBackbone
from mmyolo.models.layers.yolo_bricks import SPP, EffectiveSELayer, RepVGGBlock
from mmyolo.registry import MODELS


@MODELS.register_module()
class PPYOLOEBasicBlock(nn.Module):
    """PPYOLOE Backbone BasicBlock.

    Args:
         in_channels (int): The input channels of this Module.
         out_channels (int): The output channels of this Module.
         norm_cfg (dict): Config dict for normalization layer.
             Defaults to dict(type='BN', momentum=0.1, eps=1e-5).
         act_cfg (dict): Config dict for activation layer.
             Defaults to dict(type='SiLU', inplace=True).
         shortcut (bool): Whether to add inputs and outputs together
         at the end of this layer. Defaults to True.
         use_alpha (bool): Whether to use `alpha` parameter at 1x1 conv.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.1, eps=1e-5),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 shortcut: bool = True,
                 use_alpha: bool = False):
        super().__init__()
        assert act_cfg is None or isinstance(act_cfg, dict)
        self.conv1 = ConvModule(
            in_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.conv2 = RepVGGBlock(
            out_channels,
            out_channels,
            use_alpha=use_alpha,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            use_bn_first=False)
        self.shortcut = shortcut

    def forward(self, x: Tensor) -> Tensor:
        """Forward process.
        Args:
            inputs (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        y = self.conv1(x)
        y = self.conv2(y)
        if self.shortcut:
            return x + y
        else:
            return y


class CSPResStage(nn.Module):
    """PPYOLOE Backbone Stage.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        num_block (int): Number of blocks in this stage.
        block_cfg (dict): Config dict for block. Default config is
            suitable for PPYOLOE+ backbone. And in PPYOLOE+ neck,
            block_cfg is set to dict(type='PPYOLOEBasicBlock',
            shortcut=False, use_alpha=False). Defaults to
            dict(type='PPYOLOEBasicBlock', shortcut=True, use_alpha=True).
        stride (int): Stride of the convolution. In backbone, the stride
            must be set to 2. In neck, the stride must be set to 1.
            Defaults to 1.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.1, eps=1e-5).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        use_effective_se (bool): Whether to use `EffectiveSELayer`
            after block layer. Defaults to True.
        use_spp (bool): Whether to use `SPP` layer.
            Defaults to False.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_block: int,
                 block_cfg: ConfigType = dict(
                     type='PPYOLOEBasicBlock', shortcut=True, use_alpha=True),
                 stride: int = 1,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.1, eps=1e-5),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 use_effective_se: bool = True,
                 use_spp: bool = False):
        super().__init__()

        self.num_block = num_block
        self.block_cfg = block_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.use_spp = use_spp

        if stride == 2:
            conv1_in_channels = conv2_in_channels = conv3_in_channels = (
                in_channels + out_channels) // 2
            blocks_channels = conv1_in_channels // 2
            self.conv_down = ConvModule(
                in_channels,
                conv1_in_channels,
                3,
                stride=2,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            conv1_in_channels = conv2_in_channels = in_channels
            conv3_in_channels = out_channels
            blocks_channels = out_channels // 2
            self.conv_down = None

        self.conv1 = ConvModule(
            conv1_in_channels,
            blocks_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.conv2 = ConvModule(
            conv2_in_channels,
            blocks_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.blocks = self.build_blocks_layer(blocks_channels)

        self.conv3 = ConvModule(
            conv3_in_channels,
            out_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        if use_effective_se:
            self.attn = EffectiveSELayer(
                blocks_channels * 2, act_cfg=dict(type='HSigmoid'))
        else:
            self.attn = None

    def build_blocks_layer(self, blocks_channels: int) -> nn.Module:
        """Build blocks layer.

        Args:
            blocks_channels: The channels of this Module.
        """
        blocks = nn.Sequential()
        block_cfg = self.block_cfg.copy()
        block_cfg.update(
            dict(
                in_channels=blocks_channels,
                out_channels=blocks_channels,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))

        for i in range(self.num_block):
            blocks.add_module(str(i), MODELS.build(block_cfg))

            if i == (self.num_block - 1) // 2 and self.use_spp:
                blocks.add_module(
                    'spp',
                    SPP(blocks_channels * 4,
                        blocks_channels,
                        1, [5, 9, 13],
                        act_cfg=self.act_cfg))

        return blocks

    def forward(self, x: Tensor) -> Tensor:
        """Forward process
         Args:
             x (Tensor): The input tensor.
         """
        if self.conv_down is not None:
            x = self.conv_down(x)
        y1 = self.conv1(x)
        y2 = self.blocks(self.conv2(x))
        y = torch.cat([y1, y2], axis=1)
        if self.attn is not None:
            y = self.attn(y)
        y = self.conv3(y)
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
        block_cfg (dict): Config dict for block. Defaults to
            dict(type='PPYOLOEBasicBlock', shortcut=True, use_alpha=True)
        norm_cfg (:obj:`ConfigDict` or dict): Dictionary to construct and
            config norm layer. Defaults to dict(type='BN', momentum=0.1,
            eps=1e-5).
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
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

        cspres_layer = CSPResStage(
            in_channels=in_channels,
            out_channels=out_channels,
            num_block=num_blocks,
            block_cfg=self.block_cfg,
            stride=2,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        return [cspres_layer]
