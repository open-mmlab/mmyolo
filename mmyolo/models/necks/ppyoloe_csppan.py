# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.utils import ConfigType, OptMultiConfig
from torch import Tensor

from mmyolo.models.backbones.csp_resnet import BasicBlock
from mmyolo.models.layers.yolo_bricks import SPP
from mmyolo.models.necks import BaseYOLONeck
from mmyolo.registry import MODELS


class CSPStage(nn.Module):
    """PPYOLOE Neck Stage.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        num_block (int): Number of block in this stage.
        block (nn.Module): Basic unit of CSPStage. Defaults to BasicBlock.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.1, eps=1e-5).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='Swish').
        use_spp (bool): Whether to use `SPP` layer.
            Defaults to False.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_block: int,
                 block: nn.Module = BasicBlock,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.1, eps=1e-5),
                 act_cfg: ConfigType = dict(type='Swish'),
                 use_spp: bool = False):
        super().__init__()

        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.use_spp = use_spp

        middle_channels = int(out_channels // 2)
        self.conv1 = ConvModule(
            in_channels,
            middle_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = ConvModule(
            in_channels,
            middle_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.convs = self.build_convs_layer(block, middle_channels, num_block)
        self.conv3 = ConvModule(
            middle_channels * 2,
            out_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def build_convs_layer(self, block: nn.Module, middle_channels: int,
                          num_block: int) -> nn.Module:
        """Build convs layer maybe with `SPP`

        Args:
            block (nn.Module): Basic unit of CSPStage. Defaults to BasicBlock.
            middle_channels: The channels of this Module.
            num_block (int): Number of block in this stage.
        """
        convs = nn.Sequential()

        for i in range(num_block):
            convs.add_module(
                str(i),
                block(
                    middle_channels,
                    middle_channels,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    shortcut=False))

            if i == (num_block - 1) // 2 and self.use_spp:
                convs.add_module(
                    'spp',
                    SPP(middle_channels * 4,
                        middle_channels,
                        1, [5, 9, 13],
                        act_cfg=self.act_cfg))

        return convs

    def forward(self, x: Tensor) -> Tensor:
        """Forward process
         Args:
             x (Tensor): The input tensor.
         """
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
            Defaults to dict(type='BN', momentum=0.1, eps=1e-5).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='Swish').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
        num_stage (int): Number of stage per layer.
            Defaults to 1.
        num_block (int): Number of block per stage.
            Defaults to 3.
        use_spp (bool): Whether to use `SPP` in reduce layer.
            Defaults to False.
        use_drop_block (bool): Whether to use `DropBlock` after `CSPStage`.
            Defaults to False.
        drop_block_cfg (bool): If use_drop_block is True, build drop_block
            by this cfg. Defaults to dict(type='mmdet.DropBlock',drop_prob=0.1,
            block_size=3, warm_iters=0)
    """

    def __init__(self,
                 in_channels: List[int] = [256, 512, 1024],
                 out_channels: Union[int, List[int]] = [256, 512, 1024],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 freeze_all: bool = False,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.1, eps=1e-5),
                 act_cfg: ConfigType = dict(type='Swish'),
                 init_cfg: OptMultiConfig = None,
                 num_stage: int = 1,
                 num_block: int = 3,
                 use_spp: bool = False,
                 use_drop_block: bool = False,
                 drop_block_cfg: ConfigType = dict(
                     type='mmdet.DropBlock',
                     drop_prob=0.1,
                     block_size=3,
                     warm_iters=0)):
        self.num_stage = num_stage
        self.num_block = round(num_block * deepen_factor)
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
            channels = [self.in_channels[idx]
                        ] + [self.out_channels[idx]] * self.num_stage

            layer = [
                CSPStage(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    num_block=self.num_block,
                    block=BasicBlock,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    use_spp=self.use_spp) for i in range(self.num_stage)
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
        channels = [in_channels] + [out_channels] * self.num_stage

        layer = [
            CSPStage(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                num_block=self.num_block,
                block=BasicBlock,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                use_spp=False) for i in range(self.num_stage)
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
        channels = [in_channels] + [out_channels] * self.num_stage

        layer = [
            CSPStage(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                num_block=self.num_block,
                block=BasicBlock,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                use_spp=False) for i in range(self.num_stage)
        ]

        if self.use_drop_block:
            layer.append(MODELS.build(self.drop_block_cfg))

        return nn.Sequential(*layer)

    def build_out_layer(self, *args, **kwargs) -> nn.Module:
        """build out layer."""
        return nn.Identity()
