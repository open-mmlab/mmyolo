from mmyolo.registry import MODELS
import torch.nn as nn
import torch

from ..necks.giraffe_neck import RepConv


def get_norm(name, out_channels, inplace=True):
    if name == 'bn':
        module = nn.BatchNorm2d(out_channels)
    else:
        raise NotImplementedError
    return module


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
            self.residual_proj = ConvKXBN(in_c, out_c, 1, 1)
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


class SuperResStem(nn.Module):
    def __init__(self,
                 in_c,
                 out_c,
                 btn_c,
                 kernel_size,
                 stride,
                 num_blocks,
                 with_spp=False,
                 act='silu',
                 reparam=False,
                 block_type='k1kx'):
        super(SuperResStem, self).__init__()
        if act is None:
            self.act = torch.relu
        else:
            self.act = get_activation(act)
        self.block_list = nn.ModuleList()
        for block_id in range(num_blocks):
            if block_id == 0:
                in_channels = in_c
                out_channels = out_c
                this_stride = stride
                this_kernel_size = kernel_size
            else:
                in_channels = out_c
                out_channels = out_c
                this_stride = 1
                this_kernel_size = kernel_size
            the_block = ResConvBlock(in_channels,
                                     out_channels,
                                     btn_c,
                                     this_kernel_size,
                                     this_stride,
                                     act=act,
                                     reparam=reparam,
                                     block_type=block_type)
            self.block_list.append(the_block)
            if block_id == 0 and with_spp:
                self.block_list.append(
                    SPPBottleneck(out_channels, out_channels))

    def forward(self, x):
        output = x
        for block in self.block_list:
            output = block(output)
        return output


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


@MODELS.register_module()
class TinyNAS(nn.Module):
    # S
    structure_info = [{'class': 'ConvKXBNRELU', 'in': 3, 'k': 3, 'out': 32, 's': 1},
                      {'L': 1,
                       'btn': 24,
                       'class': 'SuperResConvK1KX',
                       'in': 32,
                       'inner_class': 'ResConvK1KX',
                       'k': 3,
                       'out': 128,
                       's': 2},
                      {'L': 5,
                       'btn': 88,
                       'class': 'SuperResConvK1KX',
                       'in': 128,
                       'inner_class': 'ResConvK1KX',
                       'k': 3,
                       'out': 128,
                       's': 2},
                      {'L': 3,
                       'btn': 128,
                       'class': 'SuperResConvK1KX',
                       'in': 128,
                       'inner_class': 'ResConvK1KX',
                       'k': 3,
                       'out': 256,
                       's': 2},
                      {'L': 2,
                       'btn': 120,
                       'class': 'SuperResConvK1KX',
                       'in': 256,
                       'inner_class': 'ResConvK1KX',
                       'k': 3,
                       'out': 256,
                       's': 1},
                      {'L': 1,
                       'btn': 144,
                       'class': 'SuperResConvK1KX',
                       'in': 256,
                       'inner_class': 'ResConvK1KX',
                       'k': 3,
                       'out': 512,
                       's': 2}]

    def __init__(self,
                 out_indices=[2, 4, 5],
                 with_spp=True,
                 use_focus=True,
                 act='silu',
                 reparam=True):
        super(TinyNAS, self).__init__()
        self.out_indices = out_indices
        self.block_list = nn.ModuleList()

        for idx, block_info in enumerate(self.structure_info):
            the_block_class = block_info['class']
            if the_block_class == 'ConvKXBNRELU':
                if use_focus:
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
                self.block_list.append(the_block)
            elif the_block_class == 'SuperResConvK1KX':
                spp = with_spp if idx == len(self.structure_info) - 1 else False
                the_block = SuperResStem(block_info['in'],
                                         block_info['out'],
                                         block_info['btn'],
                                         block_info['k'],
                                         block_info['s'],
                                         block_info['L'],
                                         spp,
                                         act=act,
                                         reparam=reparam,
                                         block_type='k1kx')
                self.block_list.append(the_block)
            elif the_block_class == 'SuperResConvKXKX':
                spp = with_spp if idx == len(self.structure_info) - 1 else False
                the_block = SuperResStem(block_info['in'],
                                         block_info['out'],
                                         block_info['btn'],
                                         block_info['k'],
                                         block_info['s'],
                                         block_info['L'],
                                         spp,
                                         act=act,
                                         reparam=reparam,
                                         block_type='kxkx')
                self.block_list.append(the_block)
            else:
                raise NotImplementedError

    def forward(self, x):
        output = x
        stage_feature_list = []
        for idx, block in enumerate(self.block_list):
            output = block(output)
            if idx in self.out_indices:
                stage_feature_list.append(output)
        return stage_feature_list