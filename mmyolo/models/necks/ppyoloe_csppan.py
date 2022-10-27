# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.models.backbones.csp_resnet import CSPResStage
from mmyolo.models.necks import BaseYOLONeck
from mmyolo.registry import MODELS


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
        block_cfg (dict): Config dict for block. Defaults to
            dict(type='PPYOLOEBasicBlock', shortcut=True, use_alpha=False)
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.1, eps=1e-5).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
        num_stages_per_layer (int): Number of stages per layer.
            Defaults to 1.
        num_blocks_per_stage (int): Number of blocks per stage.
            Defaults to 3.
        use_spp (bool): Whether to use `SPP` in reduce layer.
            Defaults to False.
        use_drop_block (bool): Whether to use `DropBlock` after `CSPStage`.
            Defaults to False.
        drop_block_cfg (dict): If use_drop_block is True, build drop_block
            by this cfg. Defaults to dict(type='mmdet.DropBlock',drop_prob=0.1,
            block_size=3, warm_iters=0)
    """

    def __init__(self,
                 in_channels: List[int] = [256, 512, 1024],
                 out_channels: Union[int, List[int]] = [256, 512, 1024],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 freeze_all: bool = False,
                 block_cfg: ConfigType = dict(
                     type='PPYOLOEBasicBlock', shortcut=False,
                     use_alpha=False),
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.1, eps=1e-5),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None,
                 num_stages_per_layer: int = 1,
                 num_blocks_per_stage: int = 3,
                 use_spp: bool = False,
                 use_drop_block: bool = False,
                 drop_block_cfg: ConfigType = dict(
                     type='mmdet.DropBlock',
                     drop_prob=0.1,
                     block_size=3,
                     warm_iters=0)):
        self.block_cfg = block_cfg
        self.num_stages_per_layer = num_stages_per_layer
        self.num_blocks_per_stage = round(num_blocks_per_stage * deepen_factor)
        self.use_spp = use_spp
        self.use_drop_block = use_drop_block
        self.drop_block_cfg = drop_block_cfg

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
            channels = [
                self.in_channels[idx]
            ] + [self.out_channels[idx]] * self.num_stages_per_layer

            layer = [
                CSPResStage(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    num_block=self.num_blocks_per_stage,
                    block_cfg=self.block_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    use_effective_se=False,
                    use_spp=self.use_spp)
                for i in range(self.num_stages_per_layer)
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
        channels = [in_channels] + [out_channels] * self.num_stages_per_layer

        layer = [
            CSPResStage(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                num_block=self.num_blocks_per_stage,
                block_cfg=self.block_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                use_effective_se=False,
                use_spp=False) for i in range(self.num_stages_per_layer)
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
        channels = [in_channels] + [out_channels] * self.num_stages_per_layer

        layer = [
            CSPResStage(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                num_block=self.num_blocks_per_stage,
                block_cfg=self.block_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                use_effective_se=False,
                use_spp=False) for i in range(self.num_stages_per_layer)
        ]

        if self.use_drop_block:
            layer.append(MODELS.build(self.drop_block_cfg))

        return nn.Sequential(*layer)

    def build_out_layer(self, *args, **kwargs) -> nn.Module:
        """build out layer."""
        return nn.Identity()
