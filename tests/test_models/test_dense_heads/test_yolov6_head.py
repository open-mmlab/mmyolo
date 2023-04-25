# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.config import Config

from mmyolo.models.dense_heads import YOLOv6Head
from mmyolo.utils import register_all_modules

register_all_modules()


class TestYOLOv6Head(TestCase):

    def setUp(self):
        self.head_module = dict(
            type='YOLOv6HeadModule',
            num_classes=2,
            in_channels=[32, 64, 128],
            featmap_strides=[8, 16, 32])

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
                max_per_img=300,
                score_thr=0.01,
                nms=dict(type='nms', iou_threshold=0.65)))

        head = YOLOv6Head(head_module=self.head_module, test_cfg=test_cfg)
        head.eval()

        feat = []
        for i in range(len(self.head_module['in_channels'])):
            in_channel = self.head_module['in_channels'][i]
            feat_size = self.head_module['featmap_strides'][i]
            feat.append(
                torch.rand(1, in_channel, s // feat_size, s // feat_size))

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
