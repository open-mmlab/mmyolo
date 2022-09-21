# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.registry import MODELS
from ..layers import RepStageBlock, RepVGGBlock
from ..utils import make_divisible, make_round
from .base_yolo_neck import BaseYOLONeck


@MODELS.register_module()
class YOLOv6RepPAFPN(BaseYOLONeck):
    """Path Aggregation Network used in YOLOv6.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Defaults to 1.
        freeze_all(bool): Whether to freeze the model.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='ReLU', inplace=True).
        block (nn.Module): block used to build each layer.
            Defaults to RepVGGBlock.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channels: int,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 12,
                 freeze_all: bool = False,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='ReLU', inplace=True),
                 block: nn.Module = RepVGGBlock,
                 init_cfg: OptMultiConfig = None):
        self.num_csp_blocks = num_csp_blocks
        self.block = block
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            freeze_all=freeze_all,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)

    def build_reduce_layer(self, idx: int) -> nn.Module:
        """build reduce layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The reduce layer.
        """
        if idx == 2:
            layer = ConvModule(
                in_channels=make_divisible(self.in_channels[idx],
                                           self.widen_factor),
                out_channels=make_divisible(self.out_channels[idx - 1],
                                            self.widen_factor),
                kernel_size=1,
                stride=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            layer = nn.Identity()

        return layer

    def build_upsample_layer(self, idx: int) -> nn.Module:
        """build upsample layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The upsample layer.
        """
        return nn.ConvTranspose2d(
            in_channels=make_divisible(self.out_channels[idx - 1],
                                       self.widen_factor),
            out_channels=make_divisible(self.out_channels[idx - 1],
                                        self.widen_factor),
            kernel_size=2,
            stride=2,
            bias=True)

    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
        """
        layer0 = RepStageBlock(
            in_channels=make_divisible(
                self.out_channels[idx - 1] + self.in_channels[idx - 1],
                self.widen_factor),
            out_channels=make_divisible(self.out_channels[idx - 1],
                                        self.widen_factor),
            n=make_round(self.num_csp_blocks, self.deepen_factor),
            block=self.block)
        if idx == 1:
            return layer0
        elif idx == 2:
            layer1 = ConvModule(
                in_channels=make_divisible(self.out_channels[idx - 1],
                                           self.widen_factor),
                out_channels=make_divisible(self.out_channels[idx - 2],
                                            self.widen_factor),
                kernel_size=1,
                stride=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            return nn.Sequential(layer0, layer1)

    def build_downsample_layer(self, idx: int) -> nn.Module:
        """build downsample layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The downsample layer.
        """
        return ConvModule(
            in_channels=make_divisible(self.out_channels[idx],
                                       self.widen_factor),
            out_channels=make_divisible(self.out_channels[idx],
                                        self.widen_factor),
            kernel_size=3,
            stride=2,
            padding=3 // 2,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The bottom up layer.
        """
        return RepStageBlock(
            in_channels=make_divisible(self.out_channels[idx] * 2,
                                       self.widen_factor),
            out_channels=make_divisible(self.out_channels[idx + 1],
                                        self.widen_factor),
            n=make_round(self.num_csp_blocks, self.deepen_factor),
            block=self.block)

    def build_out_layer(self, *args, **kwargs) -> nn.Module:
        """build out layer."""
        return nn.Identity()

    def init_weights(self):
        """Initialize the parameters."""
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                # In order to be consistent with the source code,
                # reset the Conv2d initialization parameters
                m.reset_parameters()
