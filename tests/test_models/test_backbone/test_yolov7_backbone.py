# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import pytest
import torch
from torch.nn.modules.batchnorm import _BatchNorm

from mmyolo.models.backbones import YOLOv7Backbone
from mmyolo.utils import register_all_modules
from .utils import check_norm_state

register_all_modules()


class TestYOLOv7Backbone(TestCase):

    def test_init(self):
        # out_indices in range(len(arch_setting) + 1)
        with pytest.raises(AssertionError):
            YOLOv7Backbone(out_indices=(6, ))

        with pytest.raises(ValueError):
            # frozen_stages must in range(-1, len(arch_setting) + 1)
            YOLOv7Backbone(frozen_stages=6)

    def test_forward(self):
        # Test YOLOv7Backbone-L with first stage frozen
        frozen_stages = 1
        model = YOLOv7Backbone(frozen_stages=frozen_stages)
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

        # Test YOLOv7Backbone-L with norm_eval=True
        model = YOLOv7Backbone(norm_eval=True)
        model.train()

        assert check_norm_state(model.modules(), False)

        # Test YOLOv7Backbone-L forward with widen_factor=0.25
        model = YOLOv7Backbone(
            widen_factor=0.25, out_indices=tuple(range(0, 5)))
        model.train()

        imgs = torch.randn(1, 3, 64, 64)
        feat = model(imgs)
        assert len(feat) == 5
        assert feat[0].shape == torch.Size((1, 16, 32, 32))
        assert feat[1].shape == torch.Size((1, 64, 16, 16))
        assert feat[2].shape == torch.Size((1, 128, 8, 8))
        assert feat[3].shape == torch.Size((1, 256, 4, 4))
        assert feat[4].shape == torch.Size((1, 256, 2, 2))

        # Test YOLOv7Backbone-L with plugins
        model = YOLOv7Backbone(
            widen_factor=0.25,
            plugins=[
                dict(
                    cfg=dict(
                        type='mmdet.DropBlock', drop_prob=0.1, block_size=3),
                    stages=(False, False, True, True)),
            ])

        assert len(model.stage1) == 2
        assert len(model.stage2) == 2
        assert len(model.stage3) == 3  # +DropBlock
        assert len(model.stage4) == 3  # +DropBlock
        model.train()
        imgs = torch.randn(1, 3, 128, 128)
        feat = model(imgs)
        assert len(feat) == 3
        assert feat[0].shape == torch.Size((1, 128, 16, 16))
        assert feat[1].shape == torch.Size((1, 256, 8, 8))
        assert feat[2].shape == torch.Size((1, 256, 4, 4))

        # Test YOLOv7Backbone-X forward with widen_factor=0.25
        model = YOLOv7Backbone(arch='X', widen_factor=0.25)
        model.train()

        imgs = torch.randn(1, 3, 64, 64)
        feat = model(imgs)
        assert len(feat) == 3
        assert feat[0].shape == torch.Size((1, 160, 8, 8))
        assert feat[1].shape == torch.Size((1, 320, 4, 4))
        assert feat[2].shape == torch.Size((1, 320, 2, 2))

        # Test YOLOv7Backbone-tiny forward with widen_factor=0.25
        model = YOLOv7Backbone(arch='Tiny', widen_factor=0.25)
        model.train()

        feat = model(imgs)
        assert len(feat) == 3
        assert feat[0].shape == torch.Size((1, 32, 8, 8))
        assert feat[1].shape == torch.Size((1, 64, 4, 4))
        assert feat[2].shape == torch.Size((1, 128, 2, 2))

        # Test YOLOv7Backbone-w forward with widen_factor=0.25
        model = YOLOv7Backbone(
            arch='W', widen_factor=0.25, out_indices=(2, 3, 4, 5))
        model.train()

        imgs = torch.randn(1, 3, 128, 128)
        feat = model(imgs)
        assert len(feat) == 4
        assert feat[0].shape == torch.Size((1, 64, 16, 16))
        assert feat[1].shape == torch.Size((1, 128, 8, 8))
        assert feat[2].shape == torch.Size((1, 192, 4, 4))
        assert feat[3].shape == torch.Size((1, 256, 2, 2))

        # Test YOLOv7Backbone-w forward with widen_factor=0.25
        model = YOLOv7Backbone(
            arch='D', widen_factor=0.25, out_indices=(2, 3, 4, 5))
        model.train()

        feat = model(imgs)
        assert len(feat) == 4
        assert feat[0].shape == torch.Size((1, 96, 16, 16))
        assert feat[1].shape == torch.Size((1, 192, 8, 8))
        assert feat[2].shape == torch.Size((1, 288, 4, 4))
        assert feat[3].shape == torch.Size((1, 384, 2, 2))

        # Test YOLOv7Backbone-w forward with widen_factor=0.25
        model = YOLOv7Backbone(
            arch='E', widen_factor=0.25, out_indices=(2, 3, 4, 5))
        model.train()

        feat = model(imgs)
        assert len(feat) == 4
        assert feat[0].shape == torch.Size((1, 80, 16, 16))
        assert feat[1].shape == torch.Size((1, 160, 8, 8))
        assert feat[2].shape == torch.Size((1, 240, 4, 4))
        assert feat[3].shape == torch.Size((1, 320, 2, 2))

        # Test YOLOv7Backbone-w forward with widen_factor=0.25
        model = YOLOv7Backbone(
            arch='E2E', widen_factor=0.25, out_indices=(2, 3, 4, 5))
        model.train()

        feat = model(imgs)
        assert len(feat) == 4
        assert feat[0].shape == torch.Size((1, 80, 16, 16))
        assert feat[1].shape == torch.Size((1, 160, 8, 8))
        assert feat[2].shape == torch.Size((1, 240, 4, 4))
        assert feat[3].shape == torch.Size((1, 320, 2, 2))
