# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import pytest
import torch
from mmengine.config import Config
from mmengine.structures import InstanceData

from mmyolo.models.dense_heads import RTMDetRotatedHead
from mmyolo.utils import register_all_modules

register_all_modules()


class TestRTMDetRotatedHead(TestCase):

    def setUp(self):
        self.head_module = dict(
            type='RTMDetRotatedSepBNHeadModule',
            num_classes=4,
            in_channels=1,
            stacked_convs=1,
            feat_channels=64,
            featmap_strides=[4, 8, 16])

    def test_init_weights(self):
        head = RTMDetRotatedHead(head_module=self.head_module)
        head.head_module.init_weights()

    def test_predict_by_feat(self):
        s = 256
        img_metas = [{
            'img_shape': (s, s, 3),
            'ori_shape': (s, s, 3),
            'scale_factor': (1.0, 1.0),
        }]
        test_cfg = dict(
            multi_label=True,
            decode_with_angle=True,
            nms_pre=2000,
            score_thr=0.01,
            nms=dict(type='nms_rotated', iou_threshold=0.1),
            max_per_img=300)
        test_cfg = Config(test_cfg)

        head = RTMDetRotatedHead(
            head_module=self.head_module, test_cfg=test_cfg)
        feat = [
            torch.rand(1, 1, s // feat_size, s // feat_size)
            for feat_size in [4, 8, 16]
        ]
        cls_scores, bbox_preds, angle_preds = head.forward(feat)
        head.predict_by_feat(
            cls_scores,
            bbox_preds,
            angle_preds,
            batch_img_metas=img_metas,
            cfg=test_cfg,
            rescale=True,
            with_nms=True)
        head.predict_by_feat(
            cls_scores,
            bbox_preds,
            angle_preds,
            batch_img_metas=img_metas,
            cfg=test_cfg,
            rescale=False,
            with_nms=False)

    def test_loss_by_feat(self):
        if not torch.cuda.is_available():
            pytest.skip('test requires GPU and torch+cuda')

        s = 256
        img_metas = [{
            'img_shape': (s, s, 3),
            'batch_input_shape': (s, s),
            'scale_factor': 1,
        }]
        train_cfg = dict(
            assigner=dict(
                type='BatchDynamicSoftLabelAssigner',
                num_classes=80,
                topk=13,
                iou_calculator=dict(type='mmrotate.RBboxOverlaps2D'),
                batch_iou=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)
        train_cfg = Config(train_cfg)
        head = RTMDetRotatedHead(
            head_module=self.head_module, train_cfg=train_cfg).cuda()

        feat = [
            torch.rand(1, 1, s // feat_size, s // feat_size).cuda()
            for feat_size in [4, 8, 16]
        ]
        cls_scores, bbox_preds, angle_preds = head.forward(feat)

        # Test that empty ground truth encourages the network to predict
        # background
        gt_instances = InstanceData(
            bboxes=torch.empty((0, 5)).cuda(),
            labels=torch.LongTensor([]).cuda())

        empty_gt_losses = head.loss_by_feat(cls_scores, bbox_preds,
                                            angle_preds, [gt_instances],
                                            img_metas)
        # When there is no truth, the cls loss should be nonzero but there
        # should be no box loss.
        empty_cls_loss = empty_gt_losses['loss_cls'].sum()
        empty_box_loss = empty_gt_losses['loss_bbox'].sum()
        self.assertGreater(empty_cls_loss.item(), 0,
                           'classification loss should be non-zero')
        self.assertEqual(
            empty_box_loss.item(), 0,
            'there should be no box loss when there are no true boxes')

        # When truth is non-empty then both cls and box loss should be nonzero
        # for random inputs
        head = RTMDetRotatedHead(
            head_module=self.head_module, train_cfg=train_cfg).cuda()
        gt_instances = InstanceData(
            bboxes=torch.Tensor([[130.6667, 86.8757, 100.6326, 70.8874,
                                  0.2]]).cuda(),
            labels=torch.LongTensor([1]).cuda())

        one_gt_losses = head.loss_by_feat(cls_scores, bbox_preds, angle_preds,
                                          [gt_instances], img_metas)
        onegt_cls_loss = one_gt_losses['loss_cls'].sum()
        onegt_box_loss = one_gt_losses['loss_bbox'].sum()
        self.assertGreater(onegt_cls_loss.item(), 0,
                           'cls loss should be non-zero')
        self.assertGreater(onegt_box_loss.item(), 0,
                           'box loss should be non-zero')

        # test num_class = 1
        self.head_module['num_classes'] = 1
        head = RTMDetRotatedHead(
            head_module=self.head_module, train_cfg=train_cfg).cuda()
        gt_instances = InstanceData(
            bboxes=torch.Tensor([[130.6667, 86.8757, 100.6326, 70.8874,
                                  0.2]]).cuda(),
            labels=torch.LongTensor([0]).cuda())

        cls_scores, bbox_preds, angle_preds = head.forward(feat)

        one_gt_losses = head.loss_by_feat(cls_scores, bbox_preds, angle_preds,
                                          [gt_instances], img_metas)
        onegt_cls_loss = one_gt_losses['loss_cls'].sum()
        onegt_box_loss = one_gt_losses['loss_bbox'].sum()
        self.assertGreater(onegt_cls_loss.item(), 0,
                           'cls loss should be non-zero')
        self.assertGreater(onegt_box_loss.item(), 0,
                           'box loss should be non-zero')

    def test_hbb_loss_by_feat(self):

        s = 256
        img_metas = [{
            'img_shape': (s, s, 3),
            'batch_input_shape': (s, s),
            'scale_factor': 1,
        }]
        train_cfg = dict(
            assigner=dict(
                type='BatchDynamicSoftLabelAssigner',
                num_classes=80,
                topk=13,
                iou_calculator=dict(type='mmrotate.RBboxOverlaps2D'),
                batch_iou=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)
        train_cfg = Config(train_cfg)
        hbb_cfg = dict(
            bbox_coder=dict(
                type='DistanceAnglePointCoder', angle_version='le90'),
            loss_bbox=dict(type='mmdet.GIoULoss', loss_weight=2.0),
            angle_coder=dict(
                type='mmrotate.CSLCoder',
                angle_version='le90',
                omega=1,
                window='gaussian',
                radius=1),
            loss_angle=dict(
                type='mmrotate.SmoothFocalLoss',
                gamma=2.0,
                alpha=0.25,
                loss_weight=0.2),
            use_hbbox_loss=True,
        )
        head = RTMDetRotatedHead(
            head_module=self.head_module, **hbb_cfg, train_cfg=train_cfg)

        feat = [
            torch.rand(1, 1, s // feat_size, s // feat_size)
            for feat_size in [4, 8, 16]
        ]
        cls_scores, bbox_preds, angle_preds = head.forward(feat)

        # Test that empty ground truth encourages the network to predict
        # background
        gt_instances = InstanceData(
            bboxes=torch.empty((0, 5)), labels=torch.LongTensor([]))

        empty_gt_losses = head.loss_by_feat(cls_scores, bbox_preds,
                                            angle_preds, [gt_instances],
                                            img_metas)
        # When there is no truth, the cls loss should be nonzero but there
        # should be no box loss.
        empty_cls_loss = empty_gt_losses['loss_cls'].sum()
        empty_box_loss = empty_gt_losses['loss_bbox'].sum()
        empty_angle_loss = empty_gt_losses['loss_angle'].sum()
        self.assertGreater(empty_cls_loss.item(), 0,
                           'classification loss should be non-zero')
        self.assertEqual(
            empty_box_loss.item(), 0,
            'there should be no box loss when there are no true boxes')
        self.assertEqual(
            empty_angle_loss.item(), 0,
            'there should be no angle loss when there are no true boxes')

        # When truth is non-empty then both cls and box loss should be nonzero
        # for random inputs
        head = RTMDetRotatedHead(
            head_module=self.head_module, **hbb_cfg, train_cfg=train_cfg)
        gt_instances = InstanceData(
            bboxes=torch.Tensor([[130.6667, 86.8757, 100.6326, 70.8874, 0.2]]),
            labels=torch.LongTensor([1]))

        one_gt_losses = head.loss_by_feat(cls_scores, bbox_preds, angle_preds,
                                          [gt_instances], img_metas)
        onegt_cls_loss = one_gt_losses['loss_cls'].sum()
        onegt_box_loss = one_gt_losses['loss_bbox'].sum()
        onegt_angle_loss = one_gt_losses['loss_angle'].sum()
        self.assertGreater(onegt_cls_loss.item(), 0,
                           'cls loss should be non-zero')
        self.assertGreater(onegt_box_loss.item(), 0,
                           'box loss should be non-zero')
        self.assertGreater(onegt_angle_loss.item(), 0,
                           'angle loss should be non-zero')

        # test num_class = 1
        self.head_module['num_classes'] = 1
        head = RTMDetRotatedHead(
            head_module=self.head_module, **hbb_cfg, train_cfg=train_cfg)
        gt_instances = InstanceData(
            bboxes=torch.Tensor([[130.6667, 86.8757, 100.6326, 70.8874, 0.2]]),
            labels=torch.LongTensor([0]))

        cls_scores, bbox_preds, angle_preds = head.forward(feat)

        one_gt_losses = head.loss_by_feat(cls_scores, bbox_preds, angle_preds,
                                          [gt_instances], img_metas)
        onegt_cls_loss = one_gt_losses['loss_cls'].sum()
        onegt_box_loss = one_gt_losses['loss_bbox'].sum()
        onegt_angle_loss = one_gt_losses['loss_angle'].sum()
        self.assertGreater(onegt_cls_loss.item(), 0,
                           'cls loss should be non-zero')
        self.assertGreater(onegt_box_loss.item(), 0,
                           'box loss should be non-zero')
        self.assertGreater(onegt_angle_loss.item(), 0,
                           'angle loss should be non-zero')
