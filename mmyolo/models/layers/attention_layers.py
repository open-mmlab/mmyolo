from mmyolo.registry import MODELS
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmengine.utils import is_tuple_of


@MODELS.register_module()
class SELayer(BaseModule):
    """Squeeze-and-Excitation Module.
    Args:
        in_channels (int): The input (and output) channels of the SE layer.
        ratio (int): Squeeze ratio in SELayer, the intermediate channel will be
            ``int(channels/ratio)``. Default: 16.
        conv_cfg (None or dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        act_cfg (dict or Sequence[dict]): Config dict for activation layer.
            If act_cfg is a dict, two activation layers will be configurated
            by this dict. If act_cfg is a sequence of dicts, the first
            activation layer will be configurated by the first dict and the
            second activation layer will be configurated by the second dict.
            Default: (dict(type='ReLU'), dict(type='Sigmoid'))
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels,
                 ratio=16,
                 conv_cfg=None,
                 act_cfg=(dict(type='ReLU'), dict(type='Sigmoid')),
                 init_cfg=None):
        super(SELayer, self).__init__(init_cfg)
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        assert is_tuple_of(act_cfg, dict)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=int(in_channels / ratio),
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=int(in_channels / ratio),
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[1])

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return x * out


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
            act_cfg=dict(dict(type='ReLU')))
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
      Args:
          in_channels (int): The input (and output) channels of the CBAM layer.
          ratio (int): Squeeze ratio in ChannelAttention, the intermediate channel will be
              ``int(channels/ratio)``. Default: 16.
          kernel_size(int): The size of the convolution kernel in SpatialAttention. The
              value can only be 3 or 7. Default: 7.
    """

    def __init__(self,
                 in_channels,
                 ratio=16,
                 kernel_size=7
                 ):
        super(CBAMLayer, self).__init__()
        self.channel_attention = ChannelAttention(channels=in_channels, ratio=ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out
