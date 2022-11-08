# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.utils import is_tuple_of

from mmyolo.registry import MODELS


class ChannelAttention(nn.Module):
    """ChannelAttention
    Args:
        channels (int): The input (and output) channels of the
            ChannelAttention.
        ratio (int): Squeeze ratio in ChannelAttention, the intermediate
            channel will be ``int(channels/ratio)``. Default: 16.
        act_cfg (dict or Sequence[dict]): Config dict for activation layer.
            If act_cfg is a dict, two activation layers will be configurated
            by this dict. If act_cfg is a sequence of dicts, the first
            activation layer will be configurated by the first dict and the
            second activation layer will be configurated by the second dict.
            Default: (dict(type='ReLU'), dict(type='Sigmoid'))
    """

    def __init__(self,
                 channels: int,
                 ratio: int = 16,
                 act_cfg: Union[int, Sequence[int]] = (dict(type='ReLU'),
                                                       dict(type='Sigmoid'))):
        super().__init__()
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        assert is_tuple_of(act_cfg, dict)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=int(channels / ratio),
            kernel_size=1,
            stride=1,
            conv_cfg=None,
            act_cfg=None)

        self.activate1 = MODELS.build(act_cfg[0])

        self.conv2 = ConvModule(
            in_channels=int(channels / ratio),
            out_channels=channels,
            kernel_size=1,
            stride=1,
            conv_cfg=None,
            act_cfg=None)
        self.activate2 = MODELS.build(act_cfg[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avgpool_out = self.conv2(self.activate1(self.conv1(self.avg_pool(x))))
        maxpool_out = self.conv2(self.activate1(self.conv1(self.max_pool(x))))
        out = self.activate2(avgpool_out + maxpool_out)
        return out


class SpatialAttention(nn.Module):
    """SpatialAttention
    Args:
         kernel_size (int): The size of the convolution kernel in
            SpatialAttention. Default: 7.
         act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='Sigmoid').
    """

    def __init__(self,
                 kernel_size: int = 7,
                 act_cfg: dict = dict(type='Sigmoid')):
        super().__init__()

        self.conv = ConvModule(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            conv_cfg=None,
            act_cfg=None)

        self.activate = MODELS.build(act_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.activate(self.conv(out))
        return out


@MODELS.register_module()
class CBAM(nn.Module):
    """Convolutional Block Attention Module.

    arxiv link: https://arxiv.org/abs/1807.06521v2
    Args:
        in_channels (int): The input (and output) channels of the CBAM.
        ratio (int): Squeeze ratio in ChannelAttention, the intermediate
            channel will be ``int(channels/ratio)``. Default: 16.
        kernel_size (int): The size of the convolution kernel in
            SpatialAttention. Default: 7.
        act_cfg (dict): Config dict for activation layer in ChannelAttention
            and SpatialAttention. Default: dict(ChannelAttention=
            (dict(type='ReLU'), dict(type='Sigmoid')),SpatialAttention=
            dict(type='Sigmoid'))
    """

    def __init__(self,
                 in_channels: int,
                 ratio: int = 16,
                 kernel_size: int = 7,
                 act_cfg: dict = dict(
                     ChannelAttention=(dict(type='ReLU'),
                                       dict(type='Sigmoid')),
                     SpatialAttention=dict(type='Sigmoid'))):
        super().__init__()
        self.channel_attention = ChannelAttention(
            channels=in_channels,
            ratio=ratio,
            act_cfg=act_cfg['ChannelAttention'])

        self.spatial_attention = SpatialAttention(
            kernel_size, act_cfg=act_cfg['SpatialAttention'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out
