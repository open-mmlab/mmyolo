# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import pytest
import torch
from torch.nn.modules.batchnorm import _BatchNorm

from mmyolo.models import PPYOLOECSPResNet
from mmyolo.utils import register_all_modules
from .utils import check_norm_state, is_norm

register_all_modules()


class TestPPYOLOECSPResNet(TestCase):

    def test_init(self):
        # out_indices in range(len(arch_setting) + 1)
        with pytest.raises(AssertionError):
            PPYOLOECSPResNet(out_indices=(6, ))

        with pytest.raises(ValueError):
            # frozen_stages must in range(-1, len(arch_setting) + 1)
            PPYOLOECSPResNet(frozen_stages=6)

    def test_forward(self):
        # Test PPYOLOECSPResNet with first stage frozen
        frozen_stages = 1
        model = PPYOLOECSPResNet(frozen_stages=frozen_stages)
        model.init_weights()
        model.train()

        for mod in model.stem.modules():
            for param in mod.parameters():
                assert param.requires_grad is False
        for i in range(1, frozen_stages + 1):
            layer = getattr(model, f'stage{i}')
            for mod in layer.modules():
                if isinstance(mod, _BatchNorm):
                    assert mod.training is False
            for param in layer.parameters():
                assert param.requires_grad is False

        # Test PPYOLOECSPResNet with norm_eval=True
        model = PPYOLOECSPResNet(norm_eval=True)
        model.train()

        assert check_norm_state(model.modules(), False)

        # Test PPYOLOECSPResNet-P5 forward with widen_factor=0.25
        model = PPYOLOECSPResNet(
            arch='P5', widen_factor=0.25, out_indices=range(0, 5))
        model.train()

        imgs = torch.randn(1, 3, 64, 64)
        feat = model(imgs)
        assert len(feat) == 5
        assert feat[0].shape == torch.Size((1, 16, 32, 32))
        assert feat[1].shape == torch.Size((1, 32, 16, 16))
        assert feat[2].shape == torch.Size((1, 64, 8, 8))
        assert feat[3].shape == torch.Size((1, 128, 4, 4))
        assert feat[4].shape == torch.Size((1, 256, 2, 2))

        # Test PPYOLOECSPResNet forward with dict(type='ReLU')
        model = PPYOLOECSPResNet(
            widen_factor=0.125,
            act_cfg=dict(type='ReLU'),
            out_indices=range(0, 5))
        model.train()

        imgs = torch.randn(1, 3, 64, 64)
        feat = model(imgs)
        assert len(feat) == 5
        assert feat[0].shape == torch.Size((1, 8, 32, 32))
        assert feat[1].shape == torch.Size((1, 16, 16, 16))
        assert feat[2].shape == torch.Size((1, 32, 8, 8))
        assert feat[3].shape == torch.Size((1, 64, 4, 4))
        assert feat[4].shape == torch.Size((1, 128, 2, 2))

        # Test PPYOLOECSPResNet with BatchNorm forward
        model = PPYOLOECSPResNet(widen_factor=0.125, out_indices=range(0, 5))
        for m in model.modules():
            if is_norm(m):
                assert isinstance(m, _BatchNorm)
        model.train()

        imgs = torch.randn(1, 3, 64, 64)
        feat = model(imgs)
        assert len(feat) == 5
        assert feat[0].shape == torch.Size((1, 8, 32, 32))
        assert feat[1].shape == torch.Size((1, 16, 16, 16))
        assert feat[2].shape == torch.Size((1, 32, 8, 8))
        assert feat[3].shape == torch.Size((1, 64, 4, 4))
        assert feat[4].shape == torch.Size((1, 128, 2, 2))

        # Test PPYOLOECSPResNet with BatchNorm forward
        model = PPYOLOECSPResNet(plugins=[
            dict(
                cfg=dict(type='mmdet.DropBlock', drop_prob=0.1, block_size=3),
                stages=(False, False, True, True)),
        ])

        assert len(model.stage1) == 1
        assert len(model.stage2) == 1
        assert len(model.stage3) == 2  # +DropBlock
        assert len(model.stage4) == 2  # +DropBlock
        model.train()
        imgs = torch.randn(1, 3, 256, 256)
        feat = model(imgs)
        assert len(feat) == 3
        assert feat[0].shape == torch.Size((1, 256, 32, 32))
        assert feat[1].shape == torch.Size((1, 512, 16, 16))
        assert feat[2].shape == torch.Size((1, 1024, 8, 8))
