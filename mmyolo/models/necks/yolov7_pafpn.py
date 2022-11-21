# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.registry import MODELS
from ..layers import MaxPoolAndStrideConvBlock, RepVGGBlock, SPPFCSPBlock
from .base_yolo_neck import BaseYOLONeck


@MODELS.register_module()
class YOLOv7PAFPN(BaseYOLONeck):
    """Path Aggregation Network used in YOLOv7.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        block_cfg (dict): Config dict for block.
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        spp_expand_ratio (float): Expand ratio of SPPCSPBlock.
            Defaults to 0.5.
        is_tiny_version (bool): Is tiny version of neck. If True,
            it means it is a yolov7 tiny model. Defaults to False.
        use_maxpool_in_downsample (bool): Whether maxpooling is
            used in downsample layers. Defaults to True.
        use_in_channels_in_downsample (bool): MaxPoolAndStrideConvBlock
            module input parameters. Defaults to False.
        use_repconv_outs (bool): Whether to use `repconv` in the output
            layer. Defaults to True.
        upsample_feats_cat_first (bool): Whether the output features are
            concat first after upsampling in the topdown module.
            Defaults to True. Currently only YOLOv7 is false.
        freeze_all(bool): Whether to freeze the model. Defaults to False.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channels: List[int],
                 block_cfg: dict = dict(
                     type='ELANBlock',
                     middle_ratio=0.5,
                     block_ratio=0.25,
                     num_blocks=4,
                     num_convs_in_block=1),
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 spp_expand_ratio: float = 0.5,
                 is_tiny_version: bool = False,
                 use_maxpool_in_downsample: bool = True,
                 use_in_channels_in_downsample: bool = False,
                 use_repconv_outs: bool = True,
                 upsample_feats_cat_first: bool = False,
                 freeze_all: bool = False,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):

        self.is_tiny_version = is_tiny_version
        self.use_maxpool_in_downsample = use_maxpool_in_downsample
        self.use_in_channels_in_downsample = use_in_channels_in_downsample
        self.spp_expand_ratio = spp_expand_ratio
        self.use_repconv_outs = use_repconv_outs
        self.block_cfg = block_cfg
        self.block_cfg.setdefault('norm_cfg', norm_cfg)
        self.block_cfg.setdefault('act_cfg', act_cfg)

        super().__init__(
            in_channels=[
                int(channel * widen_factor) for channel in in_channels
            ],
            out_channels=[
                int(channel * widen_factor) for channel in out_channels
            ],
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            upsample_feats_cat_first=upsample_feats_cat_first,
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
        if idx == len(self.in_channels) - 1:
            layer = SPPFCSPBlock(
                self.in_channels[idx],
                self.out_channels[idx],
                expand_ratio=self.spp_expand_ratio,
                is_tiny_version=self.is_tiny_version,
                kernel_sizes=5,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            layer = ConvModule(
                self.in_channels[idx],
                self.out_channels[idx],
                1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

        return layer

    def build_upsample_layer(self, idx: int) -> nn.Module:
        """build upsample layer."""
        return nn.Sequential(
            ConvModule(
                self.out_channels[idx],
                self.out_channels[idx - 1],
                1,
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
        block_cfg = self.block_cfg.copy()
        block_cfg['in_channels'] = self.out_channels[idx - 1] * 2
        block_cfg['out_channels'] = self.out_channels[idx - 1]
        return MODELS.build(block_cfg)

    def build_downsample_layer(self, idx: int) -> nn.Module:
        """build downsample layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The downsample layer.
        """
        if self.use_maxpool_in_downsample and not self.is_tiny_version:
            return MaxPoolAndStrideConvBlock(
                self.out_channels[idx],
                self.out_channels[idx + 1],
                use_in_channels_of_middle=self.use_in_channels_in_downsample,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            return ConvModule(
                self.out_channels[idx],
                self.out_channels[idx + 1],
                3,
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
        block_cfg = self.block_cfg.copy()
        block_cfg['in_channels'] = self.out_channels[idx + 1] * 2
        block_cfg['out_channels'] = self.out_channels[idx + 1]
        return MODELS.build(block_cfg)

    def build_out_layer(self, idx: int) -> nn.Module:
        """build out layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The out layer.
        """
        if len(self.in_channels) == 4:
            # P6
            return nn.Identity()

        out_channels = self.out_channels[idx] * 2

        if self.use_repconv_outs:
            return RepVGGBlock(
                self.out_channels[idx],
                out_channels,
                3,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            return ConvModule(
                self.out_channels[idx],
                out_channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
