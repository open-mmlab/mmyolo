# Copyright (c) OpenMMLab. All rights reserved.

from unittest import TestCase

import torch

from mmyolo.models.plugins import CBAM
from mmyolo.utils import register_all_modules

register_all_modules()


class TestCBAM(TestCase):

    def test_forward(self):

        images = torch.randn(2, 16, 20, 20)
        cbam = CBAM(16)
        out = cbam(images)
        self.assertEqual(out.shape, (2, 16, 20, 20))

        # test other ratio
        cbam = CBAM(16, ratio=8)
        out = cbam(images)
        self.assertEqual(out.shape, (2, 16, 20, 20))
