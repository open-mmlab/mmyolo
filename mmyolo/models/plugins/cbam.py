# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.utils import OptMultiConfig
from mmengine.model import BaseModule

from mmyolo.registry import MODELS


class ChannelAttention(BaseModule):
    """ChannelAttention.

    Args:
        channels (int): The input (and output) channels of the
            ChannelAttention.
        reduce_ratio (int): Squeeze ratio in ChannelAttention, the intermediate
            channel will be ``int(channels/ratio)``. Defaults to 16.
        act_cfg (dict): Config dict for activation layer
            Defaults to dict(type='ReLU').
    """

    def __init__(self,
                 channels: int,
                 reduce_ratio: int = 16,
                 act_cfg: dict = dict(type='ReLU')):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            ConvModule(
                in_channels=channels,
                out_channels=int(channels / reduce_ratio),
                kernel_size=1,
                stride=1,
                conv_cfg=None,
                act_cfg=act_cfg),
            ConvModule(
                in_channels=int(channels / reduce_ratio),
                out_channels=channels,
                kernel_size=1,
                stride=1,
                conv_cfg=None,
                act_cfg=None))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        avgpool_out = self.fc(self.avg_pool(x))
        maxpool_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avgpool_out + maxpool_out)
        return out


class SpatialAttention(BaseModule):
    """SpatialAttention
    Args:
         kernel_size (int): The size of the convolution kernel in
            SpatialAttention. Defaults to 7.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()

        self.conv = ConvModule(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            conv_cfg=None,
            act_cfg=dict(type='Sigmoid'))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return out


@MODELS.register_module()
class CBAM(BaseModule):
    """Convolutional Block Attention Module. arxiv link:
    https://arxiv.org/abs/1807.06521v2.

    Args:
        in_channels (int): The input (and output) channels of the CBAM.
        reduce_ratio (int): Squeeze ratio in ChannelAttention, the intermediate
            channel will be ``int(channels/ratio)``. Defaults to 16.
        kernel_size (int): The size of the convolution kernel in
            SpatialAttention. Defaults to 7.
        act_cfg (dict): Config dict for activation layer in ChannelAttention
            Defaults to dict(type='ReLU').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 reduce_ratio: int = 16,
                 kernel_size: int = 7,
                 act_cfg: dict = dict(type='ReLU'),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)
        self.channel_attention = ChannelAttention(
            channels=in_channels, reduce_ratio=reduce_ratio, act_cfg=act_cfg)

        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out
