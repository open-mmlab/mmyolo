# Copyright (c) OpenMMLab. All rights reserved.

from unittest import TestCase

import torch

from mmyolo.models.layers import SPPFBottleneck
from mmyolo.utils import register_all_modules

register_all_modules()


class TestSPPFBottleneck(TestCase):

    def test_forward(self):
        input_tensor = torch.randn((1, 3, 20, 20))
        bottleneck = SPPFBottleneck(3, 16)
        out_tensor = bottleneck(input_tensor)
        self.assertEqual(out_tensor.shape, (1, 16, 20, 20))

        bottleneck = SPPFBottleneck(3, 16, kernel_sizes=[3, 5, 7])
        out_tensor = bottleneck(input_tensor)
        self.assertEqual(out_tensor.shape, (1, 16, 20, 20))

        # set len(kernel_sizes)=4
        bottleneck = SPPFBottleneck(3, 16, kernel_sizes=[3, 5, 7, 9])
        out_tensor = bottleneck(input_tensor)
        self.assertEqual(out_tensor.shape, (1, 16, 20, 20))

        # set use_conv_first=False
        bottleneck = SPPFBottleneck(
            3, 16, use_conv_first=False, kernel_sizes=[3, 5, 7, 9])
        out_tensor = bottleneck(input_tensor)
        self.assertEqual(out_tensor.shape, (1, 16, 20, 20))
