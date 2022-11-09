# Copyright (c) OpenMMLab. All rights reserved.

from unittest import TestCase

import pytest
import torch

from mmyolo.models.plugins import CBAM
from mmyolo.utils import register_all_modules

register_all_modules()


class TestCBAM(TestCase):

    def test_init(self):
        with pytest.raises(AssertionError):
            # change act_cfg in ChannelAttention
            CBAM(in_channels=16, act_cfg=dict(type='Sigmoid'))

    def test_forward(self):
        tensor_shape = (2, 16, 20, 20)

        images = torch.randn(*tensor_shape)
        cbam = CBAM(16)
        out = cbam(images)
        self.assertEqual(out.shape, tensor_shape)

        # test other ratio
        cbam = CBAM(16, reduce_ratio=8)
        out = cbam(images)
        self.assertEqual(out.shape, tensor_shape)

        # test other act_cfg in ChannelAttention
        cbam = CBAM(in_channels=16, act_cfg=dict(type='Sigmoid'))
        out = cbam(images)
        self.assertEqual(out.shape, tensor_shape)
