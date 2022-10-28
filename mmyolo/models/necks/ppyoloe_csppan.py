# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.models.backbones.csp_resnet import CSPResLayer
from mmyolo.models.necks import BaseYOLONeck
from mmyolo.registry import MODELS


@MODELS.register_module()
class PPYOLOECSPPAFPN(BaseYOLONeck):
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
        num_csplayer (int): Number of `CSPResLayer` in per layer.
            Defaults to 1.
        num_blocks_per_layer (int): Number of blocks per `CSPResLayer`.
            Defaults to 3.
        block_cfg (dict): Config dict for block. Defaults to
            dict(type='PPYOLOEBasicBlock', shortcut=True, use_alpha=False)
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.1, eps=1e-5).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        drop_block_cfg (dict, optional): Drop block config.
            Defaults to None. If you want to use Drop block after
            `CSPResLayer`, you can set this para as
            dict(type='mmdet.DropBlock', drop_prob=0.1,
            block_size=3, warm_iters=0).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
        use_spp (bool): Whether to use `SPP` in reduce layer.
            Defaults to False.
    """

    def __init__(self,
                 in_channels: List[int] = [256, 512, 1024],
                 out_channels: List[int] = [256, 512, 1024],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 freeze_all: bool = False,
                 num_csplayer: int = 1,
                 num_blocks_per_layer: int = 3,
                 block_cfg: ConfigType = dict(
                     type='PPYOLOEBasicBlock', shortcut=False,
                     use_alpha=False),
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.1, eps=1e-5),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 drop_block_cfg: ConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 use_spp: bool = False):
        self.block_cfg = block_cfg
        self.num_csplayer = num_csplayer
        self.num_blocks_per_layer = round(num_blocks_per_layer * deepen_factor)
        # Only use spp in last reduce_layer, if use_spp=True.
        self.use_spp = use_spp
        self.drop_block_cfg = drop_block_cfg
        assert drop_block_cfg is None or isinstance(drop_block_cfg, dict)

        super().__init__(
            in_channels=[
                int(channel * widen_factor) for channel in in_channels
            ],
            out_channels=[
                int(channel * widen_factor) for channel in out_channels
            ],
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
            in_channels = self.in_channels[idx]
            out_channels = self.out_channels[idx]

            layer = [
                CSPResLayer(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    num_block=self.num_blocks_per_layer,
                    block_cfg=self.block_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    attention_cfg=None,
                    use_spp=self.use_spp) for i in range(self.num_csplayer)
            ]

            if self.drop_block_cfg:
                layer.append(MODELS.build(self.drop_block_cfg))
            layer = nn.Sequential(*layer)
        else:
            layer = nn.Identity()

        return layer

    def build_upsample_layer(self, idx: int) -> nn.Module:
        """build upsample layer."""
        # fpn_route
        in_channels = self.out_channels[idx]
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
        in_channels = self.in_channels[idx - 1] + self.out_channels[idx] // 2
        out_channels = self.out_channels[idx - 1]

        layer = [
            CSPResLayer(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                num_block=self.num_blocks_per_layer,
                block_cfg=self.block_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                attention_cfg=None,
                use_spp=False) for i in range(self.num_csplayer)
        ]

        if self.drop_block_cfg:
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
            in_channels=self.out_channels[idx],
            out_channels=self.out_channels[idx],
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
        in_channels = self.out_channels[idx + 1] + self.out_channels[idx]
        out_channels = self.out_channels[idx + 1]

        layer = [
            CSPResLayer(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                num_block=self.num_blocks_per_layer,
                block_cfg=self.block_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                attention_cfg=None,
                use_spp=False) for i in range(self.num_csplayer)
        ]

        if self.drop_block_cfg:
            layer.append(MODELS.build(self.drop_block_cfg))

        return nn.Sequential(*layer)

    def build_out_layer(self, *args, **kwargs) -> nn.Module:
        """build out layer."""
        return nn.Identity()
