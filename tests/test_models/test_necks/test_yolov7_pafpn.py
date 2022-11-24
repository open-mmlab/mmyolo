# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmcv.cnn import ConvModule

from mmyolo.models.necks import YOLOv7PAFPN
from mmyolo.utils import register_all_modules

register_all_modules()


class TestYOLOv7PAFPN(TestCase):

    def test_forward(self):
        # test P5
        s = 64
        in_channels = [8, 16, 32]
        feat_sizes = [s // 2**i for i in range(4)]  # [32, 16, 8]
        out_channels = [8, 16, 32]
        feats = [
            torch.rand(1, in_channels[i], feat_sizes[i], feat_sizes[i])
            for i in range(len(in_channels))
        ]
        neck = YOLOv7PAFPN(in_channels=in_channels, out_channels=out_channels)
        outs = neck(feats)
        assert len(outs) == len(feats)
        for i in range(len(feats)):
            assert outs[i].shape[1] == out_channels[i] * 2
            assert outs[i].shape[2] == outs[i].shape[3] == s // (2**i)

        # test is_tiny_version
        neck = YOLOv7PAFPN(
            in_channels=in_channels,
            out_channels=out_channels,
            is_tiny_version=True)
        outs = neck(feats)
        assert len(outs) == len(feats)
        for i in range(len(feats)):
            assert outs[i].shape[1] == out_channels[i] * 2
            assert outs[i].shape[2] == outs[i].shape[3] == s // (2**i)

        # test use_in_channels_in_downsample
        neck = YOLOv7PAFPN(
            in_channels=in_channels,
            out_channels=out_channels,
            use_in_channels_in_downsample=True)
        for f in feats:
            print(f.shape)
        outs = neck(feats)
        for f in outs:
            print(f.shape)
        assert len(outs) == len(feats)
        for i in range(len(feats)):
            assert outs[i].shape[1] == out_channels[i] * 2
            assert outs[i].shape[2] == outs[i].shape[3] == s // (2**i)

        # test use_repconv_outs is False
        neck = YOLOv7PAFPN(
            in_channels=in_channels,
            out_channels=out_channels,
            use_repconv_outs=False)
        self.assertIsInstance(neck.out_layers[0], ConvModule)

        # test P6
        s = 64
        in_channels = [8, 16, 32, 64]
        feat_sizes = [s // 2**i for i in range(4)]
        out_channels = [8, 16, 32, 64]
        feats = [
            torch.rand(1, in_channels[i], feat_sizes[i], feat_sizes[i])
            for i in range(len(in_channels))
        ]
        neck = YOLOv7PAFPN(in_channels=in_channels, out_channels=out_channels)
        outs = neck(feats)
        assert len(outs) == len(feats)
        for i in range(len(feats)):
            assert outs[i].shape[1] == out_channels[i]
            assert outs[i].shape[2] == outs[i].shape[3] == s // (2**i)
