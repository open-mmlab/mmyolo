# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_activation_layer
from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.models import BaseYOLONeck, RepVGGBlock
from mmyolo.models.utils import make_divisible, make_round
from mmyolo.registry import MODELS


def drop_block_2d(x,
                  drop_prob: float = 0.1,
                  block_size: int = 7,
                  gamma_scale: float = 1.0,
                  with_noise: bool = False,
                  inplace: bool = False,
                  batchwise: bool = False):
    """DropBlock. See https://arxiv.org/pdf/1810.12890.pdf.

    DropBlock with an experimental gaussian noise option. This layer has been
    tested on a few training runs with success, but needs further validation
    and possibly optimization for lower runtime impact.
    """
    B, C, H, W = x.shape
    total_size = W * H
    clipped_block_size = min(block_size, min(W, H))
    # seed_drop_rate, the gamma parameter
    gamma = gamma_scale * drop_prob * total_size / clipped_block_size**2 / (
        (W - block_size + 1) * (H - block_size + 1))

    # Forces the block to be inside the feature map.
    w_i, h_i = torch.meshgrid(
        torch.arange(W).to(x.device),
        torch.arange(H).to(x.device))
    valid_block = ((
                           w_i >= clipped_block_size // 2) & (
                           w_i < W - (clipped_block_size - 1) // 2)) & \
                  ((h_i >= clipped_block_size // 2) & (
                          h_i < H - (clipped_block_size - 1) // 2))
    valid_block = torch.reshape(valid_block, (1, 1, H, W)).to(dtype=x.dtype)

    if batchwise:
        # one mask for whole batch, quite a bit faster
        uniform_noise = torch.rand((1, C, H, W),
                                   dtype=x.dtype,
                                   device=x.device)
    else:
        uniform_noise = torch.rand_like(x)
    block_mask = ((2 - gamma - valid_block + uniform_noise) >= 1).to(
        dtype=x.dtype)
    block_mask = -F.max_pool2d(
        -block_mask,
        kernel_size=clipped_block_size,  # block_size,
        stride=1,
        padding=clipped_block_size // 2)

    if with_noise:
        normal_noise = torch.randn(
            (1, C, H, W), dtype=x.dtype,
            device=x.device) if batchwise else torch.randn_like(x)
        if inplace:
            x.mul_(block_mask).add_(normal_noise * (1 - block_mask))
        else:
            x = x * block_mask + normal_noise * (1 - block_mask)
    else:
        normalize_scale = (
            block_mask.numel() /
            block_mask.to(dtype=torch.float32).sum().add(1e-7)).to(x.dtype)
        if inplace:
            x.mul_(block_mask * normalize_scale)
        else:
            x = x * block_mask * normalize_scale
    return x


def drop_block_fast_2d(x: torch.Tensor,
                       drop_prob: float = 0.1,
                       block_size: int = 7,
                       gamma_scale: float = 1.0,
                       with_noise: bool = False,
                       inplace: bool = False,
                       batchwise: bool = False):
    """DropBlock. See https://arxiv.org/pdf/1810.12890.pdf.

    DropBlock with an experimental gaussian noise option. Simplied from above
    without concern for valid block mask at edges.
    """
    B, C, H, W = x.shape
    total_size = W * H
    clipped_block_size = min(block_size, min(W, H))
    gamma = gamma_scale * drop_prob * total_size / clipped_block_size**2 / (
        (W - block_size + 1) * (H - block_size + 1))

    if batchwise:
        # one mask for whole batch, quite a bit faster
        block_mask = torch.rand(
            (1, C, H, W), dtype=x.dtype, device=x.device) < gamma
    else:
        # mask per batch element
        block_mask = torch.rand_like(x) < gamma
    block_mask = F.max_pool2d(
        block_mask.to(x.dtype),
        kernel_size=clipped_block_size,
        stride=1,
        padding=clipped_block_size // 2)

    if with_noise:
        normal_noise = torch.randn(
            (1, C, H, W), dtype=x.dtype,
            device=x.device) if batchwise else torch.randn_like(x)
        if inplace:
            x.mul_(1. - block_mask).add_(normal_noise * block_mask)
        else:
            x = x * (1. - block_mask) + normal_noise * block_mask
    else:
        block_mask = 1 - block_mask
        normalize_scale = (
            block_mask.numel() /
            block_mask.to(dtype=torch.float32).sum().add(1e-7)).to(
                dtype=x.dtype)
        if inplace:
            x.mul_(block_mask * normalize_scale)
        else:
            x = x * block_mask * normalize_scale
    return x


class DropBlock2d(nn.Module):
    """DropBlock.

    See https://arxiv.org/pdf/1810.12890.pdf
    """

    def __init__(self,
                 drop_prob=0.1,
                 block_size=7,
                 gamma_scale=1.0,
                 with_noise=False,
                 inplace=False,
                 batchwise=False,
                 fast=True):
        super().__init__()
        self.drop_prob = drop_prob
        self.gamma_scale = gamma_scale
        self.block_size = block_size
        self.with_noise = with_noise
        self.inplace = inplace
        self.batchwise = batchwise
        self.fast = fast  # FIXME finish comparisons of fast vs not

    def forward(self, x):
        if not self.training or not self.drop_prob:
            return x
        if self.fast:
            return drop_block_fast_2d(x, self.drop_prob, self.block_size,
                                      self.gamma_scale, self.with_noise,
                                      self.inplace, self.batchwise)
        else:
            return drop_block_2d(x, self.drop_prob, self.block_size,
                                 self.gamma_scale, self.with_noise,
                                 self.inplace, self.batchwise)


class SPP(nn.Module):

    def __init__(
            self,
            ch_in,
            ch_out,
            k,
            pool_size,
            act_cfg=dict(type='Swish'),
    ):
        super().__init__()
        self.pool = []
        for i, size in enumerate(pool_size):
            pool = nn.MaxPool2d(
                kernel_size=size, stride=1, padding=size // 2, ceil_mode=False)
            self.add_module(f'pool{i}', pool)
            self.pool.append(pool)
        self.conv = ConvModule(
            ch_in, ch_out, k, padding=k // 2, act_cfg=act_cfg)

    def forward(self, x):
        outs = [x]

        for pool in self.pool:
            outs.append(pool(x))
        y = torch.cat(outs, axis=1)

        y = self.conv(y)
        return y


class BasicBlock(nn.Module):

    def __init__(self,
                 ch_in,
                 ch_out,
                 act_cfg=dict(type='Swish'),
                 shortcut=True,
                 inplace=True):
        super().__init__()
        assert ch_in == ch_out
        assert act_cfg is None or isinstance(act_cfg, dict)
        self.conv1 = ConvModule(
            ch_in, ch_out, 3, stride=1, padding=1, act_cfg=act_cfg)

        # build activation layer
        self.with_activation = act_cfg is not None
        if self.with_activation:
            act_cfg_ = act_cfg.copy()  # type: ignore
            # nn.Tanh has no 'inplace' argument
            if act_cfg_['type'] not in [
                    'Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish', 'GELU'
            ]:
                act_cfg_.setdefault('inplace', inplace)
            self.activate = build_activation_layer(act_cfg_)
            self.conv2 = nn.Sequential(
                RepVGGBlock(ch_out, ch_out), self.activate)
        else:
            self.conv2 = RepVGGBlock(ch_out, ch_out)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.shortcut:
            return x + y
        else:
            return y


class CSPStage(nn.Module):

    def __init__(self,
                 block_fn,
                 in_channels,
                 out_channels,
                 n,
                 act_cfg=dict(type='swish'),
                 spp=False):
        super().__init__()

        ch_mid = int(out_channels // 2)
        self.conv1 = ConvModule(in_channels, ch_mid, 1, act_cfg=act_cfg)
        self.conv2 = ConvModule(in_channels, ch_mid, 1, act_cfg=act_cfg)
        self.convs = nn.Sequential()

        next_ch_in = ch_mid
        for i in range(n):
            self.convs.add_module(
                str(i),
                block_fn(next_ch_in, ch_mid, act_cfg=act_cfg, shortcut=False))

            if i == (n - 1) // 2 and spp:
                self.convs.add_module(
                    'spp',
                    SPP(ch_mid * 4, ch_mid, 1, [5, 9, 13], act_cfg=act_cfg))
            next_ch_in = ch_mid
        # self.convs = nn.Sequential(*convs)
        self.conv3 = ConvModule(ch_mid * 2, out_channels, 1, act_cfg=act_cfg)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y2 = self.convs(y2)
        y = torch.cat([y1, y2], axis=1)
        y = self.conv3(y)
        return y


@MODELS.register_module()
class PPYOLOECustomCSPPAN(BaseYOLONeck):

    def __init__(self,
                 in_channels=[256, 512, 1024],
                 out_channels=[256, 512, 1024],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 1,
                 drop_block=False,
                 freeze_all: bool = False,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None,
                 stage_num=1,
                 block_num=3,
                 spp=False,
                 keep_prob=0.9,
                 block_size=3):
        self.stage_num = stage_num
        self.block_num = block_num
        self.spp = spp
        self.drop_block = drop_block
        self.keep_prob = keep_prob
        self.block_size = block_size
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
        if idx == 2:
            layer = []
            for j in range(self.stage_num):
                layer.append(
                    CSPStage(
                        BasicBlock,
                        in_channels=make_divisible(self.in_channels[idx],
                                                   self.widen_factor),
                        out_channels=make_divisible(self.out_channels[idx],
                                                    self.widen_factor),
                        n=make_round(self.block_num, self.deepen_factor),
                        act_cfg=self.act_cfg,
                        spp=self.spp))
            if self.drop_block:
                layer.append(
                    DropBlock2d(
                        drop_prob=1 - self.keep_prob,
                        block_size=self.block_size))
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
                act_cfg=self.act_cfg),
            nn.Upsample(scale_factor=2, mode='nearest'))

    def build_top_down_layer(self, idx: int) -> nn.Module:
        # fpn_stage
        layer = []
        in_channels = make_divisible(
            self.in_channels[idx - 1], self.widen_factor) + (
                make_divisible(self.in_channels[idx], self.widen_factor) // 2)
        out_channels = make_divisible(self.out_channels[idx - 1],
                                      self.widen_factor)
        for j in range(self.stage_num):
            layer.append(
                CSPStage(
                    BasicBlock,
                    in_channels=in_channels if j == 0 else out_channels,
                    out_channels=out_channels,
                    n=make_round(self.block_num, self.deepen_factor),
                    act_cfg=self.act_cfg,
                    spp=self.spp))
        if self.drop_block:
            layer.append(
                DropBlock2d(
                    drop_prob=1 - self.keep_prob, block_size=self.block_size))
        layer = nn.Sequential(*layer)
        return layer

    def build_downsample_layer(self, idx: int) -> nn.Module:
        # pan_route
        return ConvModule(
            in_channels=make_divisible(self.out_channels[idx],
                                       self.widen_factor),
            out_channels=make_divisible(self.out_channels[idx],
                                        self.widen_factor),
            kernel_size=3,
            stride=2,
            padding=1,
            act_cfg=self.act_cfg)

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        # pan_stage
        layer = []
        in_channels = make_divisible(
            self.out_channels[idx + 1], self.widen_factor) + make_divisible(
                self.out_channels[idx], self.widen_factor)
        out_channels = make_divisible(self.out_channels[idx + 1],
                                      self.widen_factor)
        for j in range(self.stage_num):
            layer.append(
                CSPStage(
                    BasicBlock,
                    in_channels=in_channels if j == 0 else out_channels,
                    out_channels=out_channels,
                    n=make_round(self.block_num, self.deepen_factor),
                    act_cfg=self.act_cfg,
                    spp=False))
        if self.drop_block:
            layer.append(
                DropBlock2d(
                    drop_prob=self.keep_prob, block_size=self.block_size))
        layer = nn.Sequential(*layer)
        return layer

    def build_out_layer(self, *args, **kwargs) -> nn.Module:
        return nn.Identity()


if __name__ == '__main__':
    from mmyolo.utils import register_all_modules

    register_all_modules()
    model = PPYOLOECustomCSPPAN()
    data = []
    for shape in [(1, 256, 80, 80), (1, 512, 40, 40), (1, 1024, 20, 20)]:
        data.append(torch.zeros(shape))
    res = model(data)
