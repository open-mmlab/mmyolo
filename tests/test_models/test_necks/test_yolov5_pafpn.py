# Copyright (c) OpenMMLab. All rights reserved.
from random import randint
from unittest import TestCase

import torch

from mmyolo.models.necks import YOLOv5PAFPN
from mmyolo.utils import register_all_modules

register_all_modules()


class TestYOLOv5PAFPN(TestCase):

    def test_forward(self):
        s = 64
        for e in range(99):
            in_channels = [randint(7, 999), randint(7, 999), randint(7, 999)]
            feat_sizes = [s // 2**i for i in range(4)]  # [32, 16, 8]
            out_channels = [randint(7, 999), randint(7, 999), randint(7, 999)]
            feats = [
                torch.rand(1, in_channels[i], feat_sizes[i], feat_sizes[i])
                for i in range(len(in_channels))
            ]
            neck = YOLOv5PAFPN(
                in_channels=in_channels, out_channels=out_channels)
            outs = neck(feats)
            assert len(outs) == len(feats)
            for i in range(len(feats)):
                assert outs[i].shape[1] == out_channels[i]
                assert outs[i].shape[2] == outs[i].shape[3] == s // (2**i)
            print(f'测试{e}通过')


if __name__ == '__main__':
    TestYOLOv5PAFPN().test_forward()
