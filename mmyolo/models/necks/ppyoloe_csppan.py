# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.models.layers.yolo_bricks import RepVGGBlock
from mmyolo.models.necks import BaseYOLONeck
from mmyolo.models.utils import make_divisible, make_round
from mmyolo.registry import MODELS


class SPP(nn.Module):
    """SPP layer."""

    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 kernel_size: int,
                 pool_size: List[int],
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.1, eps=1e-5),
                 act_cfg: ConfigType = dict(type='Swish')):
        super().__init__()
        self.pool = []
        for i, size in enumerate(pool_size):
            pool = nn.MaxPool2d(
                kernel_size=size, stride=1, padding=size // 2, ceil_mode=False)
            self.add_module(f'pool{i}', pool)
            self.pool.append(pool)
        self.conv = ConvModule(
            input_channels,
            output_channels,
            kernel_size,
            padding=kernel_size // 2,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [x]

        for pool in self.pool:
            outs.append(pool(x))
        y = torch.cat(outs, axis=1)

        y = self.conv(y)
        return y


class BasicBlock(nn.Module):
    """PPYOLOE neck Basic Block."""

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


class CSPStage(nn.Module):
    """PPYOLOE Neck Stage."""

    def __init__(self,
                 block_fn: nn.Module,
                 input_channels: int,
                 output_channels: int,
                 num_layer: int,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.1, eps=1e-5),
                 act_cfg: ConfigType = dict(type='Swish'),
                 spp: bool = False):
        super().__init__()

        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.spp = spp

        middle_channels = int(output_channels // 2)
        self.conv1 = ConvModule(
            input_channels,
            middle_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = ConvModule(
            input_channels,
            middle_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.convs = self.build_convs_layer(block_fn, middle_channels,
                                            num_layer)
        self.conv3 = ConvModule(
            middle_channels * 2,
            output_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def build_convs_layer(self, block_fn: nn.Module, middle_channels: int,
                          num_layer: int):
        convs = nn.Sequential()

        for i in range(num_layer):
            convs.add_module(
                str(i),
                block_fn(
                    middle_channels,
                    middle_channels,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    shortcut=False))

            if i == (num_layer - 1) // 2 and self.spp:
                convs.add_module(
                    'spp',
                    SPP(middle_channels * 4,
                        middle_channels,
                        1, [5, 9, 13],
                        act_cfg=self.act_cfg))

        return convs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y2 = self.convs(y2)
        y = torch.cat([y1, y2], axis=1)
        y = self.conv3(y)
        return y


@MODELS.register_module()
class PPYOLOECSPPAN(BaseYOLONeck):
    """CSPPAN in PPYOLOE.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (List[int]): Number of output channels
            (used at each scale).
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        freeze_all(bool): Whether to freeze the model.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
        num_stage (int): Number of stage per layer.
            Defaults to 1.
        num_block (int): Number of block per stage.
            Defaults to 3.
        spp (bool): Whether to use `SPP` in reduce layer.
            Defaults to False.
        use_drop_block (bool): Whether to use `DropBlock` after `CSPStage`.
            Defaults to False.
    """

    def __init__(self,
                 in_channels: List[int] = [256, 512, 1024],
                 out_channels: Union[int, List[int]] = [256, 512, 1024],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 freeze_all: bool = False,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='Swish'),
                 init_cfg: OptMultiConfig = None,
                 num_stage: int = 1,
                 num_block: int = 3,
                 spp: bool = False,
                 use_drop_block: bool = False,
                 drop_block_cfg: ConfigType = dict(
                     type='mmdet.DropBlock',
                     drop_prob=0.1,
                     block_size=3,
                     warm_iters=0)):
        self.num_stage = num_stage
        self.num_block = num_block
        self.spp = spp
        self.use_drop_block = use_drop_block
        self.drop_block_cfg = drop_block_cfg

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            freeze_all=freeze_all,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)

    def build_reduce_layer(self, idx: int):
        """build reduce layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The reduce layer.
        """
        if idx == len(self.in_channels) - 1:
            # fpn_stage
            in_channels = make_divisible(self.in_channels[idx],
                                         self.widen_factor)
            output_channels = make_divisible(self.out_channels[idx],
                                             self.widen_factor)
            channels = [in_channels] + [output_channels] * self.num_stage
            num_layer = make_round(self.num_block, self.deepen_factor)

            layer = [
                CSPStage(
                    BasicBlock,
                    input_channels=channels[i],
                    output_channels=channels[i + 1],
                    num_layer=num_layer,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    spp=self.spp) for i in range(self.num_stage)
            ]

            if self.use_drop_block:
                layer.append(MODELS.build(self.drop_block_cfg))
            layer = nn.Sequential(*layer)
        else:
            layer = nn.Identity()

        return layer

    def build_upsample_layer(self, idx: int) -> nn.Module:
        """build upsample layer."""
        # fpn_route
        in_channels = make_divisible(self.out_channels[idx], self.widen_factor)
        return nn.Sequential(
            ConvModule(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            nn.Upsample(scale_factor=2, mode='nearest'))

    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
        """
        # fpn_stage
        in_channels = make_divisible(
            self.in_channels[idx - 1], self.widen_factor) + (
                make_divisible(self.out_channels[idx], self.widen_factor) // 2)
        out_channels = make_divisible(self.out_channels[idx - 1],
                                      self.widen_factor)
        channels = [in_channels] + [out_channels] * self.num_stage
        num_layer = make_round(self.num_block, self.deepen_factor)

        layer = [
            CSPStage(
                BasicBlock,
                input_channels=channels[i],
                output_channels=channels[i + 1],
                num_layer=num_layer,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                spp=False) for i in range(self.num_stage)
        ]

        if self.use_drop_block:
            layer.append(MODELS.build(self.drop_block_cfg))

        return nn.Sequential(*layer)

    def build_downsample_layer(self, idx: int) -> nn.Module:
        """build downsample layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The downsample layer.
        """
        # pan_route
        return ConvModule(
            in_channels=make_divisible(self.out_channels[idx],
                                       self.widen_factor),
            out_channels=make_divisible(self.out_channels[idx],
                                        self.widen_factor),
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The bottom up layer.
        """
        # pan_stage
        in_channels = make_divisible(
            self.out_channels[idx + 1], self.widen_factor) + make_divisible(
                self.out_channels[idx], self.widen_factor)
        out_channels = make_divisible(self.out_channels[idx + 1],
                                      self.widen_factor)
        channels = [in_channels] + [out_channels] * self.num_stage
        num_layer = make_round(self.num_block, self.deepen_factor)

        layer = [
            CSPStage(
                BasicBlock,
                input_channels=channels[i],
                output_channels=channels[i + 1],
                num_layer=num_layer,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                spp=False) for i in range(self.num_stage)
        ]

        if self.use_drop_block:
            layer.append(MODELS.build(self.drop_block_cfg))

        return nn.Sequential(*layer)

    def build_out_layer(self, *args, **kwargs) -> nn.Module:
        """build out layer."""
        return nn.Identity()
