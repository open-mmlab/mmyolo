import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import BACKBONES
from torch.nn.modules.batchnorm import _BatchNorm


def get_activation(name="silu"):
    if name == "silu":
        module = nn.SiLU()
    elif name == "relu":
        module = nn.ReLU()
    elif name in ["LeakyReLU", 'leakyrelu', 'lrelu']:
        module = nn.LeakyReLU(0.1)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class BaseConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 groups=1,
                 bias=False,
                 act="silu"):
        super(BaseConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=(ksize - 1) // 2,
            groups=groups,
            bias=bias)
        self.bn = nn.BatchNorm2d(
            out_channels,
            eps=1e-3,  # for amp(fp16)
            momentum=0.97)
        self.act = get_activation(act)
        
    def forward(self, x):
        self.to("cuda:0")
        x = self.bn(self.conv(x))
        if self.training:
            y = self.act(x)
        else:
            if isinstance(self.act, nn.SiLU):
                self.act = SiLU()
            y = self.act(x)
        return y


class DWConv(nn.Module):
    """Depthwise Conv"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride=1,
                 bias=False,
                 act="silu"):
        super(DWConv, self).__init__()
        self.dw_conv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            bias=bias,
            act=act)
        self.pw_conv = BaseConv(
            in_channels,
            out_channels,
            ksize=1,
            stride=1,
            groups=1,
            bias=bias,
            act=act)

    def forward(self, x):
        return self.pw_conv(self.dw_conv(x))


class Focus(nn.Module):
    """Focus width and height information into channel space, used in YOLOX."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=3,
                 stride=1,
                 bias=False,
                 act="silu"):
        super(Focus, self).__init__()
        self.conv = BaseConv(
            in_channels * 4,
            out_channels,
            ksize=ksize,
            stride=stride,
            bias=bias,
            act=act)

    def forward(self, inputs):
        # inputs [bs, C, H, W] -> outputs [bs, 4C, W/2, H/2]
        top_left = inputs[:, :, 0::2, 0::2]
        top_right = inputs[:, :, 0::2, 1::2]
        bottom_left = inputs[:, :, 1::2, 0::2]
        bottom_right = inputs[:, :, 1::2, 1::2]
        outputs = torch.concat(
            [top_left, bottom_left, top_right, bottom_right], 1)
        return self.conv(outputs)


class BottleNeck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 shortcut=True,
                 expansion=0.5,
                 depthwise=False,
                 bias=False,
                 act="silu"):
        super(BottleNeck, self).__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.conv2 = Conv(
            hidden_channels,
            out_channels,
            ksize=3,
            stride=1,
            bias=bias,
            act=act)
        self.add_shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.add_shortcut:
            y = y + x
        return y


class SPPLayer(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer used in YOLOv3-SPP and YOLOX"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes=(5, 9, 13),
                 bias=False,
                 act="silu"):
        super(SPPLayer, self).__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.maxpoolings = nn.ModuleList([
            nn.MaxPool2d(
                kernel_size=ks, stride=1, padding=ks // 2)
            for ks in kernel_sizes
        ])
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(
            conv2_channels, out_channels, ksize=1, stride=1, bias=bias, act=act)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.concat([x] + [mp(x) for mp in self.maxpoolings], axis=1)
        x = self.conv2(x)
        return x


class SPPFLayer(nn.Module):
    """ Spatial Pyramid Pooling - Fast (SPPF) layer used in YOLOv5 by Glenn Jocher,
        equivalent to SPP(k=(5, 9, 13))
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=5,
                 bias=False,
                 act='silu'):
        super(SPPFLayer, self).__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.maxpooling = nn.MaxPool2d(
            kernel_size=ksize, stride=1, padding=ksize // 2)
        conv2_channels = hidden_channels * 4
        self.conv2 = BaseConv(
            conv2_channels, out_channels, ksize=1, stride=1, bias=bias, act=act)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.maxpooling(x)
        y2 = self.maxpooling(y1)
        y3 = self.maxpooling(y2)
        concats = torch.concat([x, y1, y2, y3], axis=1)
        out = self.conv2(concats)
        return out


class CSPLayer(nn.Module):
    """CSP (Cross Stage Partial) layer with 3 convs, named C3 in YOLOv5"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=1,
                 shortcut=True,
                 expansion=0.5,
                 depthwise=False,
                 bias=False,
                 act="silu"):
        super(CSPLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.conv2 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(* [
            BottleNeck(
                hidden_channels,
                hidden_channels,
                shortcut=shortcut,
                expansion=1.0,
                depthwise=depthwise,
                bias=bias,
                act=act) for _ in range(num_blocks)
        ])
        self.conv3 = BaseConv(
            hidden_channels * 2,
            out_channels,
            ksize=1,
            stride=1,
            bias=bias,
            act=act)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        x = torch.concat([x_1, x_2], axis=1)
        x = self.conv3(x)
        return x


##### YOLOv7 ####


class ELANLayer(nn.Module):
    """ELAN layer used in YOLOv7, like CSPLayer(C3) in YOLOv5/YOLOX"""

    def __init__(self,
                 in_channels,
                 mid_channels1,
                 mid_channels2,
                 out_channels,
                 num_blocks=4,
                 concat_list=[-1, -3, -5, -6],
                 depthwise=False,
                 bias=False,
                 act="silu"):
        super(ELANLayer, self).__init__()
        self.num_blocks = num_blocks
        self.concat_list = concat_list

        self.conv1 = BaseConv(
            in_channels, mid_channels1, ksize=1, stride=1, bias=bias, act=act)
        self.conv2 = BaseConv(
            in_channels, mid_channels1, ksize=1, stride=1, bias=bias, act=act)

        self.bottlenecks = nn.Sequential(* [
            BaseConv(
                mid_channels1 if i == 0 else mid_channels2,
                mid_channels2,
                ksize=3,
                stride=1,
                bias=bias,
                act=act) for i in range(num_blocks)
        ])

        concat_chs = mid_channels1 * 2 + mid_channels2 * (len(concat_list) - 2)
        self.conv3 = BaseConv(
            int(concat_chs),
            out_channels,
            ksize=1,
            stride=1,
            bias=bias,
            act=act)

    def forward(self, x):
        outs = []
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        outs.append(x_1)
        outs.append(x_2)
        idx = [i + self.num_blocks for i in self.concat_list[:-2]]
        for i in range(self.num_blocks):
            x_2 = self.bottlenecks[i](x_2)
            if i in idx:
                outs.append(x_2)
        outs = outs[::-1]  # [-1, -3]
        x_all = torch.concat(outs, axis=1)
        y = self.conv3(x_all)
        return y


class ELAN2Layer(nn.Module):
    """ELAN2 layer used in YOLOv7-E6E"""

    def __init__(self,
                 in_channels,
                 mid_channels1,
                 mid_channels2,
                 out_channels,
                 num_blocks=4,
                 concat_list=[-1, -3, -5, -6],
                 depthwise=False,
                 bias=False,
                 act="silu"):
        super(ELAN2Layer, self).__init__()
        self.elan_layer1 = ELANLayer(in_channels, mid_channels1, mid_channels2,
                                     out_channels, num_blocks, concat_list,
                                     depthwise, bias, act)
        self.elan_layer2 = ELANLayer(in_channels, mid_channels1, mid_channels2,
                                     out_channels, num_blocks, concat_list,
                                     depthwise, bias, act)

    def forward(self, x):
        return self.elan_layer1(x) + self.elan_layer2(x)


class MPConvLayer(nn.Module):
    """MPConvLayer used in YOLOv7"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=0.5,
                 depthwise=False,
                 bias=False,
                 act="silu"):
        super(MPConvLayer, self).__init__()
        mid_channels = int(out_channels * expansion)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = BaseConv(
            in_channels, mid_channels, ksize=1, stride=1, bias=bias, act=act)

        self.conv2 = BaseConv(
            in_channels, mid_channels, ksize=1, stride=1, bias=bias, act=act)
        self.conv3 = BaseConv(
            mid_channels, mid_channels, ksize=3, stride=2, bias=bias, act=act)

    def forward(self, x):
        x_1 = self.conv1(self.maxpool(x))
        x_2 = self.conv3(self.conv2(x))
        x = torch.concat([x_2, x_1], axis=1)
        return x


class MP(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super(MP, self).__init__()
        self.mp = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.mp(x)


class DownC(nn.Module):
    def __init__(self, c1, c2, k=2, act='silu'):
        super(DownC, self).__init__()
        c_ = int(c1)  # hidden channels
        self.mp = nn.MaxPool2d(kernel_size=k, stride=k)
        self.cv1 = BaseConv(c1, c_, 1, 1, act=act)
        self.cv2 = BaseConv(c_, c2 // 2, 3, k, act=act)
        self.cv3 = BaseConv(c1, c2 // 2, 1, 1, act=act)

    def forward(self, x):
        x_2 = self.cv2(self.cv1(x))
        x_3 = self.cv3(self.mp(x))
        return torch.concat([x_2, x_3], 1)


class SPPCSPC(nn.Module):
    def __init__(self, c1, c2, g=1, e=0.5, k=(5, 9, 13), act='silu'):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = BaseConv(c1, c_, 1, 1, act=act)
        self.cv2 = BaseConv(c1, c_, 1, 1, act=act)
        self.cv3 = BaseConv(c_, c_, 3, 1, act=act)
        self.cv4 = BaseConv(c_, c_, 1, 1, act=act)
        self.maxpoolings = nn.ModuleList(
            [nn.MaxPool2d(
                kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = BaseConv(4 * c_, c_, 1, 1, act=act)
        self.cv6 = BaseConv(c_, c_, 3, 1, act=act)
        self.cv7 = BaseConv(2 * c_, c2, 1, 1, act=act)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(
            self.cv5(
                torch.concat([x1] + [mp(x1) for mp in self.maxpoolings], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.concat([y1, y2], axis=1))


class SPPELAN(nn.Module):
    def __init__(self, c1, c2, g=1, e=0.5, k=(5, 9, 13), act='silu'):
        super(SPPELAN, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = BaseConv(c1, c_, 1, 1, act=act)
        self.cv2 = BaseConv(c1, c_, 1, 1, act=act)
        self.maxpoolings = nn.ModuleList(
            [nn.MaxPool2d(
                kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv3 = BaseConv(4 * c_, c_, 1, 1, act=act)
        self.cv4 = BaseConv(2 * c_, c2, 1, 1, act=act)

    def forward(self, x):
        x_1 = self.cv1(x)
        x_2 = self.cv2(x)
        x_cats = [x_2] + [mp(x_2) for mp in self.maxpoolings]
        y_cats = self.cv3(torch.concat(x_cats[::-1], 1))
        y = torch.concat([y_cats, x_1], 1)
        return self.cv4(y)


# class ImplicitA(nn.Module):
#     def __init__(self, channel, mean=0., std=.02):
#         super(ImplicitA, self).__init__()
#         self.ia = self.create_parameter(
#             shape=([1, channel, 1, 1]),
#             attr=ParamAttr(initializer=Constant(0.)))
#         normal_(self.ia, mean=mean, std=std)

#     def forward(self, x):
#         return self.ia + x


# class ImplicitM(nn.Module):
#     def __init__(self, channel, mean=0., std=.02):
#         super(ImplicitM, self).__init__()
#         self.im = self.create_parameter(
#             shape=([1, channel, 1, 1]),
#             attr=ParamAttr(initializer=Constant(1.)))
#         normal_(self.im, mean=mean, std=std)

#     def forward(self, x):
#         return self.im * x


class RepConv(nn.Module):
    # RepVGG, see https://arxiv.org/abs/2101.03697
    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, act='silu', deploy=False):
        super(RepConv, self).__init__()
        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2
        assert k == 3
        assert p == 1
        padding_11 = p - k // 2

        self.act = get_activation(act)

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                c1, c2, k, s, p, groups=g, bias_attr=True)
        else:
            self.rbr_identity = (nn.BatchNorm2d(c1)
                                 if c2 == c1 and s == 1 else None)
            self.rbr_dense = nn.Sequential(* [
                nn.Conv2d(
                    c1, c2, k, s, p, groups=g, bias_attr=False),
                nn.BatchNorm2d(c2),
            ])
            self.rbr_1x1 = nn.Sequential(* [
                nn.Conv2d(
                    c1, c2, 1, s, padding_11, groups=g, bias_attr=False),
                nn.BatchNorm2d(c2),
            ])

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            x = self.rbr_reparam(inputs)
            if self.training:
                y = self.act(x)
            else:
                if isinstance(self.act, nn.Silu):
                    self.act = SiLU()
                y = self.act(x)
            return y

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        x = self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out
        if self.training:
            y = self.act(x)
        else:
            if isinstance(self.act, nn.Silu):
                self.act = SiLU()
            y = self.act(x)
        return y

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid, )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1]._mean
            running_var = branch[1]._variance
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1]._epsilon
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros([self.in_channels, input_dim, 3, 3])

                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch._mean
            running_var = branch._variance
            gamma = branch.weight
            beta = branch.bias
            eps = branch._epsilon
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))
        return kernel * t, beta - running_mean * gamma / std

    def convert_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        if not hasattr(self, 'rbr_reparam'):
            self.rbr_reparam = nn.Conv2d(
                self.in_channels,
                self.out_channels,
                3,
                1,
                1,
                groups=self.groups,
                bias_attr=True)
        self.rbr_reparam.weight.set_value(kernel)
        self.rbr_reparam.bias.set_value(bias)
        self.__delattr__("rbr_dense")
        self.__delattr__("rbr_1x1")
        if hasattr(self, "rbr_identity"):
            self.__delattr__("rbr_identity")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")
        self.deploy = True


@BACKBONES.register_module()
class ELANNet(nn.Module):
    """
    ELANNet, YOLOv7's backbone.
    Args:
        arch (str): Architecture of ELANNet, from {tiny, L, X, W6, E6, D6, E6E}, default as 'L',
        depth_mult (float): Depth multiplier, multiply number of channels in
            each layer, default as 1.0.
        width_mult (float): Width multiplier, multiply number of blocks in
            CSPLayer, default as 1.0.
        depthwise (bool): Whether to use depth-wise conv layer.
        act (str): Activation function type, default as 'silu'.
        trt (bool): Whether use trt infer.
        return_idx (list): Index of stages whose feature maps are returned.
    """
    __shared__ = ['arch', 'depth_mult', 'width_mult', 'act', 'trt']

    # in_channels, out_channels of 1 stem + 4 stages
    ch_settings = {
        'tiny': [[32, 64], [64, 64], [64, 128], [128, 256], [256, 512]],
        'L': [[32, 64], [64, 256], [256, 512], [512, 1024], [1024, 1024]],
        'X': [[40, 80], [80, 320], [320, 640], [640, 1280], [1280, 1280]],
        'W6':
        [[64, 64], [64, 128], [128, 256], [256, 512], [512, 768], [768, 1024]],
        'E6':
        [[80, 80], [80, 160], [160, 320], [320, 640], [640, 960], [960, 1280]],
        'D6': [[96, 96], [96, 192], [192, 384], [384, 768], [768, 1152],
               [1152, 1536]],
        'E6E':
        [[80, 80], [80, 160], [160, 320], [320, 640], [640, 960], [960, 1280]],
    }
    # mid_ch1, mid_ch2 of 4 stages' ELANLayer
    mid_ch_settings = {
        'tiny': [[32, 32], [64, 64], [128, 128], [256, 256]],
        'L': [[64, 64], [128, 128], [256, 256], [256, 256]],
        'X': [[64, 64], [128, 128], [256, 256], [256, 256]],
        'W6': [[64, 64], [128, 128], [256, 256], [384, 384], [512, 512]],
        'E6': [[64, 64], [128, 128], [256, 256], [384, 384], [512, 512]],
        'D6': [[64, 64], [128, 128], [256, 256], [384, 384], [512, 512]],
        'E6E': [[64, 64], [128, 128], [256, 256], [384, 384], [512, 512]],
    }
    # concat_list of 4 stages
    concat_list_settings = {
        'tiny': [-1, -2, -3, -4],
        'L': [-1, -3, -5, -6],
        'X': [-1, -3, -5, -7, -8],
        'W6': [-1, -3, -5, -6],
        'E6': [-1, -3, -5, -7, -8],
        'D6': [-1, -3, -5, -7, -9, -10],
        'E6E': [-1, -3, -5, -7, -8],
    }
    num_blocks = {
        'tiny': 2,
        'L': 4,
        'X': 6,
        'W6': 4,
        'E6': 6,
        'D6': 8,
        'E6E': 6
    }

    def __init__(self,
                 arch='L',
                 depth_mult=1.0,
                 width_mult=1.0,
                 depthwise=False,
                 act='silu',
                 trt=False,
                 return_idx=[2, 3, 4]):
        super(ELANNet, self).__init__()
        self.arch = arch
        self.return_idx = return_idx
        Conv = DWConv if depthwise else BaseConv

        ch_settings = self.ch_settings[arch]
        mid_ch_settings = self.mid_ch_settings[arch]
        concat_list_settings = self.concat_list_settings[arch]
        num_blocks = self.num_blocks[arch]

        layers_num = 0
        ch_1 = ch_settings[0][0]
        ch_2 = ch_settings[0][0] * 2
        ch_out = ch_settings[0][-1]
        if self.arch in ['L', 'X']:
            self.stem = nn.Sequential(* [
                Conv(
                    3, ch_1, 3, 1, bias=False, act=act),
                Conv(
                    ch_1, ch_2, 3, 2, bias=False, act=act),
                Conv(
                    ch_2, ch_out, 3, 1, bias=False, act=act),
            ])
            layers_num = 3
        elif self.arch in ['tiny']:
            self.stem = nn.Sequential(* [
                Conv(
                    3, ch_1, 3, 2, bias=False, act=act),
                Conv(
                    ch_1, ch_out, 3, 2, bias=False, act=act),
            ])
            layers_num = 2
        elif self.arch in ['W6', 'E6', 'D6', 'E6E']:
            # ReOrg
            self.stem = Focus(3, ch_out, 3, 1, bias=False, act=act)
            layers_num = 2
        else:
            raise AttributeError("Unsupported arch type: {}".format(self.arch))

        self._out_channels = [chs[-1] for chs in ch_settings]
        # for SPPCSPC(L,X,W6,E6,D6,E6E) or SPPELAN(tiny)
        self._out_channels[-1] //= 2
        self._out_channels = [self._out_channels[i] for i in self.return_idx]
        self.strides = [[2, 4, 8, 16, 32, 64][i] for i in self.return_idx]

        self.blocks = []
        for i, (in_ch, out_ch) in enumerate(ch_settings[1:]):
            stage = []

            # 1.Downsample methods: Conv, DownC, MPConvLayer, MP, None
            if i == 0:
                if self.arch in ['L', 'X', 'W6']:
                    # Conv
                    _out_ch = out_ch if self.arch == 'W6' else out_ch // 2
                    conv_layer = Conv(
                            in_ch, _out_ch, 3, 2, bias=False, act=act)
                    stage.append(conv_layer)
                    layers_num += 1
                elif self.arch in ['E6', 'D6', 'E6E']:
                    # DownC
                    downc_layer = DownC(
                            in_ch, out_ch, 2, act=act)
                    stage.append(downc_layer)
                    layers_num += 1
                elif self.arch in ['tiny']:
                    # None
                    pass
                else:
                    raise AttributeError("Unsupported arch type: {}".format(
                        self.arch))
            else:
                if self.arch in ['L', 'X']:
                    # MPConvLayer
                    # Note: out channels of MPConvLayer is int(in_ch * 0.5 * 2)
                    # no relationship with out_ch when used in backbone
                    conv_res_layer = MPConvLayer(
                            in_ch, in_ch, 0.5, depthwise, bias=False, act=act)
                    stage.append(conv_res_layer)
                    layers_num += 5  # 1 maxpool + 3 convs + 1 concat
                elif self.arch in ['tiny']:
                    # MP
                    mp_layer = MP(kernel_size=2, stride=2)
                    stage.append(mp_layer)
                    layers_num += 1
                elif self.arch in ['W6']:
                    # Conv
                    conv_layer = Conv(
                            in_ch, out_ch, 3, 2, bias=False, act=act)
                    stage.append(conv_layer)
                    layers_num += 1
                elif self.arch in ['E6', 'D6', 'E6E']:
                    # DownC
                    downc_layer = DownC(
                            in_ch, out_ch, 2, act=act)
                    stage.append(downc_layer)
                    layers_num += 1
                else:
                    raise AttributeError("Unsupported arch type: {}".format(
                        self.arch))

            # 2.ELANLayer Block: like CSPLayer(C3) in YOLOv5/YOLOX
            elan_in_ch = in_ch
            if i == 0 and self.arch in ['L', 'X']:
                elan_in_ch = in_ch * 2
            if self.arch in ['W6', 'E6', 'D6', 'E6E']:
                elan_in_ch = out_ch
            ELANBlock = ELAN2Layer if self.arch in ['E6E'] else ELANLayer
            elan_layer = ELANBlock(
                    elan_in_ch,
                    mid_ch_settings[i][0],
                    mid_ch_settings[i][1],
                    out_ch,
                    num_blocks=num_blocks,
                    concat_list=concat_list_settings,
                    depthwise=depthwise,
                    bias=False,
                    act=act)
            stage.append(elan_layer)
            layers_num += int(2 + num_blocks + 2)
            # conv1 + conv2 + bottleneck + concat + conv3

            # 3.SPP(Spatial Pyramid Pooling) methods: SPPCSPC, SPPELAN
            if i == len(ch_settings[1:]) - 1:
                if self.arch in ['L', 'X', 'W6', 'E6', 'D6', 'E6E']:
                    sppcspc_layer = SPPCSPC(
                            out_ch, out_ch // 2, k=(5, 9, 13), act=act)
                    stage.append(sppcspc_layer)
                    layers_num += 1
                elif self.arch in ['tiny']:
                    sppelan_layer = SPPELAN(
                            out_ch, out_ch // 2, k=(5, 9, 13), act=act)
                    stage.append(sppelan_layer)
                    layers_num += 9
                else:
                    raise AttributeError("Unsupported arch type: {}".format(
                        self.arch))

            self.blocks.append(nn.Sequential(*stage))



    def train(self, mode=True):
        super(ELANNet, self).train(mode)
        # self._freeze_stages()
        if mode and False:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, inputs):
        x = inputs
        outputs = []
        x = self.stem(x)
        for i, layer in enumerate(self.blocks):
            x = layer(x)
            if i + 1 in self.return_idx:
                outputs.append(x)
        return outputs