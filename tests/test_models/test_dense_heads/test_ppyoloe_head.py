# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.config import Config
from mmengine.model import bias_init_with_prob
from mmengine.testing import assert_allclose

from mmyolo.models import PPYOLOEHead
from mmyolo.utils import register_all_modules

register_all_modules()


class TestYOLOXHead(TestCase):

    def setUp(self):
        self.head_module = dict(
            type='PPYOLOEHeadModule', num_classes=4, in_channels=[32, 64, 128])

    def test_init_weights(self):
        head = PPYOLOEHead(head_module=self.head_module)
        head.head_module.init_weights()
        bias_init = bias_init_with_prob(0.01)
        for conv_cls, conv_reg in zip(head.head_module.cls_preds,
                                      head.head_module.reg_preds):
            assert_allclose(conv_cls.weight.data,
                            torch.zeros_like(conv_cls.weight.data))
            assert_allclose(conv_reg.weight.data,
                            torch.zeros_like(conv_reg.weight.data))

            assert_allclose(conv_cls.bias.data,
                            torch.ones_like(conv_cls.bias.data) * bias_init)
            assert_allclose(conv_reg.bias.data,
                            torch.ones_like(conv_reg.bias.data))

    def test_predict_by_feat(self):
        s = 256
        img_metas = [{
            'img_shape': (s, s, 3),
            'ori_shape': (s, s, 3),
            'scale_factor': (1.0, 1.0),
        }]
        test_cfg = Config(
            dict(
                multi_label=True,
                nms_pre=1000,
                score_thr=0.01,
                nms=dict(type='nms', iou_threshold=0.7),
                max_per_img=300))

        head = PPYOLOEHead(head_module=self.head_module, test_cfg=test_cfg)
        feat = [
            torch.rand(1, in_channels, s // feat_size, s // feat_size)
            for in_channels, feat_size in [[32, 8], [64, 16], [128, 32]]
        ]
        cls_scores, bbox_preds = head.forward(feat)
        head.predict_by_feat(
            cls_scores,
            bbox_preds,
            None,
            img_metas,
            cfg=test_cfg,
            rescale=True,
            with_nms=True)
        head.predict_by_feat(
            cls_scores,
            bbox_preds,
            None,
            img_metas,
            cfg=test_cfg,
            rescale=False,
            with_nms=False)
