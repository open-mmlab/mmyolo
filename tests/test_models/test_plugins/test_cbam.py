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
            # act_cfg in ChannelAttention sequence length must equal to 2
            CBAM(
                in_channels=16,
                act_cfg=dict(
                    ChannelAttention=(dict(type='ReLU'), ),
                    SpatialAttention=dict(type='Sigmoid')))

        with pytest.raises(AssertionError):
            # act_cfg in ChannelAttention sequence must be a tuple of dict
            CBAM(
                in_channels=16,
                act_cfg=dict(
                    ChannelAttention=[dict(type='ReLU'),
                                      dict(type='Sigmoid')],
                    SpatialAttention=dict(type='Sigmoid')))

    def test_forward(self):
        images = torch.randn(2, 16, 20, 20)
        cbam = CBAM(16)
        out = cbam(images)
        self.assertEqual(out.shape, (2, 16, 20, 20))

        # test other ratio
        cbam = CBAM(16, ratio=8)
        out = cbam(images)
        self.assertEqual(out.shape, (2, 16, 20, 20))

        # test other act_cfg in ChannelAttention
        cbam = CBAM(
            16,
            ratio=8,
            act_cfg=dict(
                ChannelAttention=(dict(type='ReLU'), dict(type='HSigmoid')),
                SpatialAttention=dict(type='Sigmoid')))
        out = cbam(images)
        self.assertEqual(out.shape, (2, 16, 20, 20))

        # test other act_cfg in SpatialAttention
        cbam = CBAM(
            16,
            ratio=8,
            act_cfg=dict(
                ChannelAttention=(dict(type='ReLU'), dict(type='Sigmoid')),
                SpatialAttention=dict(type='HSigmoid')))
        out = cbam(images)
        self.assertEqual(out.shape, (2, 16, 20, 20))
