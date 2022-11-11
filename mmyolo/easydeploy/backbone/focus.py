# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn


class DeployFocus(nn.Module):

    def __init__(self, orin_Focus: nn.Module):
        super().__init__()
        self.__dict__.update(orin_Focus.__dict__)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1, 2, W)
        x = x.reshape(B, C, x.shape[2], 2, -1, 2)
        half_H = x.shape[2]
        half_W = x.shape[4]
        x = x.permute(0, 5, 3, 1, 2, 4)
        x = x.reshape(B, C * 4, half_H, half_W)

        return self.conv(x)


class NcnnFocus(nn.Module):

    def __init__(self, orin_Focus: nn.Module):
        super().__init__()
        self.__dict__.update(orin_Focus.__dict__)

    def forward(self, x):
        batch_size, c, h, w = x.shape
        assert h % 2 == 0 and w % 2 == 0, f'focus for yolox needs even feature\
            height and width, got {(h, w)}.'

        x = x.reshape(batch_size, c * h, 1, w)
        _b, _c, _h, _w = x.shape
        g = _c // 2
        # fuse to ncnn's shufflechannel
        x = x.view(_b, g, 2, _h, _w)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(_b, -1, _h, _w)

        x = x.reshape(_b, c * h * w, 1, 1)

        _b, _c, _h, _w = x.shape
        g = _c // 2
        # fuse to ncnn's shufflechannel
        x = x.view(_b, g, 2, _h, _w)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(_b, -1, _h, _w)

        x = x.reshape(_b, c * 4, h // 2, w // 2)

        return self.conv(x)
