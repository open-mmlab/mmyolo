# Copyright (c) OpenMMLab. All rights reserved.

from unittest import TestCase

import torch

from mmyolo.models.layers import CBAMLayer
from mmyolo.utils import register_all_modules

register_all_modules()


class TestCBAMLayer(TestCase):

    def test_forward(self):
        # kernel size must be 3 or 7
        with self.assertRaisesRegex(AssertionError,
                                    'kernel size must be 3 or 7'):
            CBAMLayer(16, kernel_size=5)

        images = torch.randn(2, 16, 20, 20)
        cbam_layer = CBAMLayer(16)
        out = cbam_layer(images)
        self.assertEqual(out.shape, (2, 16, 20, 20))

        # test other ratio
        cbam_layer = CBAMLayer(16, ratio=8)
        out = cbam_layer(images)
        self.assertEqual(out.shape, (2, 16, 20, 20))
