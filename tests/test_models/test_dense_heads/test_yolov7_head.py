# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.config import Config
from mmengine.structures import InstanceData

from mmyolo.models.dense_heads import YOLOv7Head
from mmyolo.utils import register_all_modules

register_all_modules()


# TODO: Test YOLOv7p6HeadModule
class TestYOLOv7Head(TestCase):

    def setUp(self):
        self.head_module = dict(
            type='YOLOv7HeadModule',
            num_classes=2,
            in_channels=[32, 64, 128],
            featmap_strides=[8, 16, 32],
            num_base_priors=3)

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

        head = YOLOv7Head(head_module=self.head_module, test_cfg=test_cfg)

        feat = []
        for i in range(len(self.head_module['in_channels'])):
            in_channel = self.head_module['in_channels'][i]
            feat_size = self.head_module['featmap_strides'][i]
            feat.append(
                torch.rand(1, in_channel, s // feat_size, s // feat_size))

        cls_scores, bbox_preds, objectnesses = head.forward(feat)
        head.predict_by_feat(
            cls_scores,
            bbox_preds,
            objectnesses,
            img_metas,
            cfg=test_cfg,
            rescale=True,
            with_nms=True)
        head.predict_by_feat(
            cls_scores,
            bbox_preds,
            objectnesses,
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

        head = YOLOv7Head(head_module=self.head_module)

        feat = []
        for i in range(len(self.head_module['in_channels'])):
            in_channel = self.head_module['in_channels'][i]
            feat_size = self.head_module['featmap_strides'][i]
            feat.append(
                torch.rand(1, in_channel, s // feat_size, s // feat_size))

        cls_scores, bbox_preds, objectnesses = head.forward(feat)

        # Test that empty ground truth encourages the network to predict
        # background
        gt_instances = InstanceData(
            bboxes=torch.empty((0, 4)), labels=torch.LongTensor([]))

        empty_gt_losses = head.loss_by_feat(cls_scores, bbox_preds,
                                            objectnesses, [gt_instances],
                                            img_metas)
        # When there is no truth, the cls loss should be nonzero but there
        # should be no box loss.
        empty_cls_loss = empty_gt_losses['loss_cls'].sum()
        empty_box_loss = empty_gt_losses['loss_bbox'].sum()
        empty_obj_loss = empty_gt_losses['loss_obj'].sum()
        self.assertEqual(
            empty_cls_loss.item(), 0,
            'there should be no cls loss when there are no true boxes')
        self.assertEqual(
            empty_box_loss.item(), 0,
            'there should be no box loss when there are no true boxes')
        self.assertGreater(empty_obj_loss.item(), 0,
                           'objectness loss should be non-zero')

        # When truth is non-empty then both cls and box loss should be nonzero
        # for random inputs
        head = YOLOv7Head(head_module=self.head_module)
        gt_instances = InstanceData(
            bboxes=torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
            labels=torch.LongTensor([1]))

        one_gt_losses = head.loss_by_feat(cls_scores, bbox_preds, objectnesses,
                                          [gt_instances], img_metas)
        onegt_cls_loss = one_gt_losses['loss_cls'].sum()
        onegt_box_loss = one_gt_losses['loss_bbox'].sum()
        onegt_obj_loss = one_gt_losses['loss_obj'].sum()
        self.assertGreater(onegt_cls_loss.item(), 0,
                           'cls loss should be non-zero')
        self.assertGreater(onegt_box_loss.item(), 0,
                           'box loss should be non-zero')
        self.assertGreater(onegt_obj_loss.item(), 0,
                           'obj loss should be non-zero')

        # test num_class = 1
        self.head_module['num_classes'] = 1
        head = YOLOv7Head(head_module=self.head_module)
        gt_instances = InstanceData(
            bboxes=torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
            labels=torch.LongTensor([0]))

        cls_scores, bbox_preds, objectnesses = head.forward(feat)

        one_gt_losses = head.loss_by_feat(cls_scores, bbox_preds, objectnesses,
                                          [gt_instances], img_metas)
        onegt_cls_loss = one_gt_losses['loss_cls'].sum()
        onegt_box_loss = one_gt_losses['loss_bbox'].sum()
        onegt_obj_loss = one_gt_losses['loss_obj'].sum()
        self.assertEqual(onegt_cls_loss.item(), 0,
                         'cls loss should be non-zero')
        self.assertGreater(onegt_box_loss.item(), 0,
                           'box loss should be non-zero')
        self.assertGreater(onegt_obj_loss.item(), 0,
                           'obj loss should be non-zero')
