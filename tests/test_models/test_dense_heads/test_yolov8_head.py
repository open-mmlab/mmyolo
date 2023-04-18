# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine import ConfigDict
from mmengine.config import Config

from mmyolo.models import YOLOv8Head
from mmyolo.utils import register_all_modules

register_all_modules()


class TestYOLOv8Head(TestCase):

    def setUp(self):
        self.head_module = dict(
            type='YOLOv8HeadModule',
            num_classes=4,
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

        head = YOLOv8Head(head_module=self.head_module, test_cfg=test_cfg)
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

    def test_loss_by_feat(self):
        s = 256
        img_metas = [{
            'img_shape': (s, s, 3),
            'batch_input_shape': (s, s),
            'scale_factor': 1,
        }]

        head = YOLOv8Head(
            head_module=self.head_module,
            train_cfg=ConfigDict(
                assigner=dict(
                    type='BatchTaskAlignedAssigner',
                    num_classes=4,
                    topk=10,
                    alpha=0.5,
                    beta=6)))
        head.train()

        feat = []
        for i in range(len(self.head_module['in_channels'])):
            in_channel = self.head_module['in_channels'][i]
            feat_size = self.head_module['featmap_strides'][i]
            feat.append(
                torch.rand(1, in_channel, s // feat_size, s // feat_size))

        cls_scores, bbox_preds, bbox_dist_preds = head.forward(feat)

        # Test that empty ground truth encourages the network to predict
        # background
        gt_instances = torch.empty((0, 6), dtype=torch.float32)

        empty_gt_losses = head.loss_by_feat(cls_scores, bbox_preds,
                                            bbox_dist_preds, gt_instances,
                                            img_metas)
        # When there is no truth, the cls loss should be nonzero but there
        # should be no box loss.
        empty_cls_loss = empty_gt_losses['loss_cls'].sum()
        empty_box_loss = empty_gt_losses['loss_bbox'].sum()
        empty_dfl_loss = empty_gt_losses['loss_dfl'].sum()
        self.assertGreater(empty_cls_loss.item(), 0,
                           'cls loss should be non-zero')
        self.assertEqual(
            empty_box_loss.item(), 0,
            'there should be no box loss when there are no true boxes')
        self.assertEqual(
            empty_dfl_loss.item(), 0,
            'there should be df loss when there are no true boxes')

        # When truth is non-empty then both cls and box loss should be nonzero
        # for random inputs
        gt_instances = torch.Tensor(
            [[0., 0., 23.6667, 23.8757, 238.6326, 151.8874]])

        one_gt_losses = head.loss_by_feat(cls_scores, bbox_preds,
                                          bbox_dist_preds, gt_instances,
                                          img_metas)
        onegt_cls_loss = one_gt_losses['loss_cls'].sum()
        onegt_box_loss = one_gt_losses['loss_bbox'].sum()
        onegt_loss_dfl = one_gt_losses['loss_dfl'].sum()
        self.assertGreater(onegt_cls_loss.item(), 0,
                           'cls loss should be non-zero')
        self.assertGreater(onegt_box_loss.item(), 0,
                           'box loss should be non-zero')
        self.assertGreater(onegt_loss_dfl.item(), 0,
                           'obj loss should be non-zero')

        # test num_class = 1
        self.head_module['num_classes'] = 1
        head = YOLOv8Head(
            head_module=self.head_module,
            train_cfg=ConfigDict(
                assigner=dict(
                    type='BatchTaskAlignedAssigner',
                    num_classes=1,
                    topk=10,
                    alpha=0.5,
                    beta=6)))
        head.train()

        gt_instances = torch.Tensor(
            [[0., 0., 23.6667, 23.8757, 238.6326, 151.8874],
             [1., 0., 24.6667, 27.8757, 28.6326, 51.8874]])
        cls_scores, bbox_preds, bbox_dist_preds = head.forward(feat)

        one_gt_losses = head.loss_by_feat(cls_scores, bbox_preds,
                                          bbox_dist_preds, gt_instances,
                                          img_metas)
        onegt_cls_loss = one_gt_losses['loss_cls'].sum()
        onegt_box_loss = one_gt_losses['loss_bbox'].sum()
        onegt_loss_dfl = one_gt_losses['loss_dfl'].sum()
        self.assertGreater(onegt_cls_loss.item(), 0,
                           'cls loss should be non-zero')
        self.assertGreater(onegt_box_loss.item(), 0,
                           'box loss should be non-zero')
        self.assertGreater(onegt_loss_dfl.item(), 0,
                           'obj loss should be non-zero')
