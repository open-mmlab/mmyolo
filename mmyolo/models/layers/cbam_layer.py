# Copyright (c) OpenMMLab. All rights reserved.
from mmyolo.registry import MODELS
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule


class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=int(channels / ratio),
            kernel_size=1,
            stride=1,
            conv_cfg=None,
            act_cfg=dict(type='ReLU', inplace=True))
        self.conv2 = ConvModule(
            in_channels=int(channels / ratio),
            out_channels=channels,
            kernel_size=1,
            stride=1,
            conv_cfg=None,
            act_cfg=None)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgpool_out = self.conv2(self.conv1(self.avg_pool(x)))
        maxpool_out = self.conv2(self.conv1(self.max_pool(x)))
        out = self.sigmoid(avgpool_out + maxpool_out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        self.conv = ConvModule(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            conv_cfg=None,
            act_cfg=dict(type='Sigmoid'))

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return out


@MODELS.register_module()
class CBAMLayer(BaseModule):
    """Convolutional Block Attention Module.
      arxiv link: https://arxiv.org/abs/1807.06521v2
      Args:
          in_channels (int): The input (and output) channels of the CBAM layer.
          ratio (int): Squeeze ratio in ChannelAttention, the intermediate
              channel will be ``int(channels/ratio)``. Default: 16.
          kernel_size(int): The size of the convolution kernel in
              SpatialAttention. The value can only be 3 or 7. Default: 7.
    """

    def __init__(self,
                 in_channels,
                 ratio=16,
                 kernel_size=7
                 ):
        super(CBAMLayer, self).__init__()
        self.channel_attention = ChannelAttention(channels=in_channels,
                                                  ratio=ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out
