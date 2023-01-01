# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import torch
import torch.nn as nn

from ..necks.giraffe_neck import RepConv

from mmyolo.registry import MODELS


def get_norm(name, out_channels, inplace=True):
    if name == 'bn':
        module = nn.BatchNorm2d(out_channels)
    else:
        raise NotImplementedError
    return module


def get_activation(name='silu', inplace=True):
    if name is None:
        return nn.Identity()

    if isinstance(name, str):
        if name == 'silu':
            module = nn.SiLU(inplace=inplace)
        elif name == 'relu':
            module = nn.ReLU(inplace=inplace)
        elif name == 'lrelu':
            module = nn.LeakyReLU(0.1, inplace=inplace)
        elif name == 'hardsigmoid':
            module = nn.Hardsigmoid(inplace=inplace)
        elif name == 'identity':
            module = nn.Identity()
        else:
            raise AttributeError('Unsupported act type: {}'.format(name))
        return module

    elif isinstance(name, nn.Module):
        return name

    else:
        raise AttributeError('Unsupported act type: {}'.format(name))


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes=(5, 9, 13),
                 activation='silu'):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = ConvBNAct(in_channels,
                               hidden_channels,
                               1,
                               stride=1,
                               act=activation)
        self.m = nn.ModuleList([
            nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
            for ks in kernel_sizes
        ])
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = ConvBNAct(conv2_channels,
                               out_channels,
                               1,
                               stride=1,
                               act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class ConvKXBN(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride):
        super(ConvKXBN, self).__init__()
        self.conv1 = nn.Conv2d(in_c,
                               out_c,
                               kernel_size,
                               stride, (kernel_size - 1) // 2,
                               groups=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn1(self.conv1(x))

    def fuseforward(self, x):
        return self.conv1(x)


class ConvKXBNRELU(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride, act='silu'):
        super(ConvKXBNRELU, self).__init__()
        self.conv = ConvKXBN(in_c, out_c, kernel_size, stride)
        if act is None:
            self.activation_function = torch.relu
        else:
            self.activation_function = get_activation(act)

    def forward(self, x):
        output = self.conv(x)
        return self.activation_function(output)


class ResConvBlock(nn.Module):
    def __init__(self,
                 in_c,
                 out_c,
                 btn_c,
                 kernel_size,
                 stride,
                 act='silu',
                 reparam=False,
                 block_type='k1kx'):
        super(ResConvBlock, self).__init__()
        self.stride = stride
        if block_type == 'k1kx':
            self.conv1 = ConvKXBN(in_c, btn_c, kernel_size=1, stride=1)
        else:
            self.conv1 = ConvKXBN(in_c,
                                  btn_c,
                                  kernel_size=kernel_size,
                                  stride=1)
        if not reparam:
            self.conv2 = ConvKXBN(btn_c, out_c, kernel_size, stride)
        else:
            self.conv2 = RepConv(btn_c,
                                 out_c,
                                 kernel_size,
                                 stride,
                                 act='identity')

        self.activation_function = get_activation(act)

        if in_c != out_c and stride != 2:
            self.residual_proj = ConvKXBN(in_c, out_c, kernel_size=1, stride=1)
        else:
            self.residual_proj = None

    def forward(self, x):
        if self.residual_proj is not None:
            reslink = self.residual_proj(x)
        else:
            reslink = x
        x = self.conv1(x)
        x = self.activation_function(x)
        x = self.conv2(x)
        if self.stride != 2:
            x = x + reslink
        x = self.activation_function(x)
        return x


class ConvBNAct(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
            self,
            in_channels,
            out_channels,
            ksize,
            stride=1,
            groups=1,
            bias=False,
            act='silu',
            norm='bn',
            reparam=False,
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        if norm is not None:
            self.bn = get_norm(norm, out_channels, inplace=True)
        if act is not None:
            self.act = get_activation(act, inplace=True)
        self.with_norm = norm is not None
        self.with_act = act is not None

    def forward(self, x):
        x = self.conv(x)
        if self.with_norm:
            x = self.bn(x)
        if self.with_act:
            x = self.act(x)
        return x

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=1,
                 stride=1,
                 act='silu'):
        super().__init__()
        self.conv = ConvBNAct(in_channels * 4,
                              out_channels,
                              ksize,
                              stride,
                              act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)


class CSPStem(nn.Module):
    def __init__(self,
                 in_c,
                 out_c,
                 btn_c,
                 stride,
                 kernel_size,
                 num_blocks,
                 act='silu',
                 reparam=False,
                 block_type='k1kx'):
        super(CSPStem, self).__init__()
        self.in_channels = in_c
        self.out_channels = out_c
        self.stride = stride
        if self.stride == 2:
            self.num_blocks = num_blocks - 1
        else:
            self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.act = act
        self.block_type = block_type
        out_c = out_c // 2

        if act is None:
            self.act = torch.relu
        else:
            self.act = get_activation(act)
        self.block_list = nn.ModuleList()
        for block_id in range(self.num_blocks):
            if self.stride == 1 and block_id == 0:
                in_c = in_c // 2
            else:
                in_c = out_c
            the_block = ResConvBlock(in_c,
                                     out_c,
                                     btn_c,
                                     kernel_size,
                                     stride=1,
                                     act=act,
                                     reparam=reparam,
                                     block_type=block_type)
            self.block_list.append(the_block)

    def forward(self, x):
        output = x
        for block in self.block_list:
            output = block(output)
        return output


@MODELS.register_module()
class TinyNAS_csp(nn.Module):
    structure_info_m = [{'class': 'ConvKXBNRELU', 'in': 3, 'k': 3, 'nbitsA': 8, 'nbitsW': 8, 'out': 32, 's': 1},
                        {'L': 2,
                         'btn': 64,
                         'class': 'SuperResConvKXKX',
                         'in': 32,
                         'inner_class': 'ResConvKXKX',
                         'k': 3,
                         'nbitsA': [8, 8, 8, 8],
                         'nbitsW': [8, 8, 8, 8],
                         'out': 128,
                         's': 2},
                        {'L': 4,
                         'btn': 64,
                         'class': 'SuperResConvKXKX',
                         'in': 128,
                         'inner_class': 'ResConvKXKX',
                         'k': 3,
                         'nbitsA': [8, 8, 8, 8, 8, 8, 8, 8],
                         'nbitsW': [8, 8, 8, 8, 8, 8, 8, 8],
                         'out': 128,
                         's': 2},
                        {'L': 4,
                         'btn': 256,
                         'class': 'SuperResConvKXKX',
                         'in': 128,
                         'inner_class': 'ResConvKXKX',
                         'k': 3,
                         'nbitsA': [8, 8, 8, 8, 8, 8, 8, 8],
                         'nbitsW': [8, 8, 8, 8, 8, 8, 8, 8],
                         'out': 256,
                         's': 2},
                        {'L': 4,
                         'btn': 256,
                         'class': 'SuperResConvKXKX',
                         'in': 256,
                         'inner_class': 'ResConvKXKX',
                         'k': 3,
                         'nbitsA': [8, 8, 8, 8, 8, 8, 8, 8],
                         'nbitsW': [8, 8, 8, 8, 8, 8, 8, 8],
                         'out': 256,
                         's': 1},
                        {'L': 3,
                         'btn': 256,
                         'class': 'SuperResConvKXKX',
                         'in': 256,
                         'inner_class': 'ResConvKXKX',
                         'k': 3,
                         'nbitsA': [8, 8, 8, 8, 8, 8],
                         'nbitsW': [8, 8, 8, 8, 8, 8],
                         'out': 512,
                         's': 2}
                        ]

    def __init__(self,
                 out_indices=[2, 3, 4],
                 with_spp=False,
                 use_focus=False,
                 act='silu',
                 backbone_structure='m',
                 reparam=False):
        super(TinyNAS_csp, self).__init__()
        self.out_indices = out_indices
        self.block_list = nn.ModuleList()
        self.stride_list = []

        if backbone_structure == 'm':
            structure_info = self.structure_info_m
        else:
            raise ValueError('please specify the right backbone structure')

        for idx, block_info in enumerate(structure_info):
            the_block_class = block_info['class']
            if the_block_class == 'ConvKXBNRELU':
                if use_focus and idx == 0:
                    the_block = Focus(block_info['in'],
                                      block_info['out'],
                                      block_info['k'],
                                      act=act)
                else:
                    the_block = ConvKXBNRELU(block_info['in'],
                                             block_info['out'],
                                             block_info['k'],
                                             block_info['s'],
                                             act=act)
            elif the_block_class == 'SuperResConvK1KX':
                the_block = CSPStem(block_info['in'],
                                    block_info['out'],
                                    block_info['btn'],
                                    block_info['s'],
                                    block_info['k'],
                                    block_info['L'],
                                    act=act,
                                    reparam=reparam,
                                    block_type='k1kx')
            elif the_block_class == 'SuperResConvKXKX':
                the_block = CSPStem(block_info['in'],
                                    block_info['out'],
                                    block_info['btn'],
                                    block_info['s'],
                                    block_info['k'],
                                    block_info['L'],
                                    act=act,
                                    reparam=reparam,
                                    block_type='kxkx')
            else:
                raise NotImplementedError

            self.block_list.append(the_block)

        self.csp_stage = nn.ModuleList()
        self.csp_stage.append(self.block_list[0])
        self.csp_stage.append(CSPWrapper(self.block_list[1]))
        self.csp_stage.append(CSPWrapper(self.block_list[2]))
        self.csp_stage.append(
            CSPWrapper((self.block_list[3], self.block_list[4])))
        self.csp_stage.append(CSPWrapper(self.block_list[5],
                                         with_spp=with_spp))
        del self.block_list

    def init_weights(self, pretrain=None):
        pass

    def forward(self, x):
        output = x
        stage_feature_list = []
        for idx, block in enumerate(self.csp_stage):
            output = block(output)
            if idx in self.out_indices:
                stage_feature_list.append(output)
        return stage_feature_list


class CSPWrapper(nn.Module):
    def __init__(self, convstem, act='relu', reparam=False, with_spp=False):

        super(CSPWrapper, self).__init__()
        self.with_spp = with_spp
        if isinstance(convstem, tuple):
            in_c = convstem[0].in_channels
            out_c = convstem[-1].out_channels
            hidden_dim = convstem[0].out_channels // 2
            _convstem = nn.ModuleList()
            for modulelist in convstem:
                for layer in modulelist.block_list:
                    _convstem.append(layer)
        else:
            in_c = convstem.in_channels
            out_c = convstem.out_channels
            hidden_dim = out_c // 2
            _convstem = convstem.block_list

        self.convstem = nn.ModuleList()
        for layer in _convstem:
            self.convstem.append(layer)

        self.act = get_activation(act)
        self.downsampler = ConvKXBNRELU(in_c,
                                        hidden_dim * 2,
                                        3,
                                        2,
                                        act=self.act)
        if self.with_spp:
            self.spp = SPPBottleneck(hidden_dim * 2, hidden_dim * 2)
        if len(self.convstem) > 0:
            self.conv_start = ConvKXBNRELU(hidden_dim * 2,
                                           hidden_dim,
                                           1,
                                           1,
                                           act=self.act)
            self.conv_shortcut = ConvKXBNRELU(hidden_dim * 2,
                                              out_c // 2,
                                              1,
                                              1,
                                              act=self.act)
            self.conv_fuse = ConvKXBNRELU(out_c, out_c, 1, 1, act=self.act)

    def forward(self, x):
        x = self.downsampler(x)
        if self.with_spp:
            x = self.spp(x)
        if len(self.convstem) > 0:
            shortcut = self.conv_shortcut(x)
            x = self.conv_start(x)
            for block in self.convstem:
                x = block(x)
            x = torch.cat((x, shortcut), dim=1)
            x = self.conv_fuse(x)
        return x

# def load_tinynas_net(backbone_cfg):
#     # load masternet model to path
#     import ast
#
#     struct_str = ''.join([x.strip() for x in backbone_cfg.net_structure_str])
#     struct_info = ast.literal_eval(struct_str)
#     for layer in struct_info:
#         if 'nbitsA' in layer:
#             del layer['nbitsA']
#         if 'nbitsW' in layer:
#             del layer['nbitsW']
#
#     model = TinyNAS(structure_info=struct_info,
#                     out_indices=backbone_cfg.out_indices,
#                     with_spp=backbone_cfg.with_spp,
#                     use_focus=backbone_cfg.use_focus,
#                     act=backbone_cfg.act,
#                     reparam=backbone_cfg.reparam)
#
#     return model
