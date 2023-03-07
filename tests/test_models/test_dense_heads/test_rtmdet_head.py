# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import torch
from mmengine.config import Config
from mmengine.structures import InstanceData

from mmyolo.models import RTMDetInsSepBNHead
from mmyolo.models.dense_heads import RTMDetHead
from mmyolo.utils import register_all_modules

register_all_modules()


class TestRTMDetHead(TestCase):

    def setUp(self):
        self.head_module = dict(
            type='RTMDetSepBNHeadModule',
            num_classes=4,
            in_channels=1,
            stacked_convs=1,
            feat_channels=64,
            featmap_strides=[4, 8, 16])

    def test_init_weights(self):
        head = RTMDetHead(head_module=self.head_module)
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
            nms_pre=30000,
            score_thr=0.001,
            nms=dict(type='nms', iou_threshold=0.65),
            max_per_img=300)
        test_cfg = Config(test_cfg)

        head = RTMDetHead(head_module=self.head_module, test_cfg=test_cfg)
        feat = [
            torch.rand(1, 1, s // feat_size, s // feat_size)
            for feat_size in [4, 8, 16]
        ]
        cls_scores, bbox_preds = head.forward(feat)
        head.predict_by_feat(
            cls_scores,
            bbox_preds,
            batch_img_metas=img_metas,
            cfg=test_cfg,
            rescale=True,
            with_nms=True)
        head.predict_by_feat(
            cls_scores,
            bbox_preds,
            batch_img_metas=img_metas,
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
        train_cfg = dict(
            assigner=dict(
                num_classes=80,
                type='BatchDynamicSoftLabelAssigner',
                topk=13,
                iou_calculator=dict(type='mmdet.BboxOverlaps2D')),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)
        train_cfg = Config(train_cfg)
        head = RTMDetHead(head_module=self.head_module, train_cfg=train_cfg)

        feat = [
            torch.rand(1, 1, s // feat_size, s // feat_size)
            for feat_size in [4, 8, 16]
        ]
        cls_scores, bbox_preds = head.forward(feat)

        # Test that empty ground truth encourages the network to predict
        # background
        gt_instances = InstanceData(
            bboxes=torch.empty((0, 4)), labels=torch.LongTensor([]))

        empty_gt_losses = head.loss_by_feat(cls_scores, bbox_preds,
                                            [gt_instances], img_metas)
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
        head = RTMDetHead(head_module=self.head_module, train_cfg=train_cfg)
        gt_instances = InstanceData(
            bboxes=torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
            labels=torch.LongTensor([1]))

        one_gt_losses = head.loss_by_feat(cls_scores, bbox_preds,
                                          [gt_instances], img_metas)
        onegt_cls_loss = one_gt_losses['loss_cls'].sum()
        onegt_box_loss = one_gt_losses['loss_bbox'].sum()
        self.assertGreater(onegt_cls_loss.item(), 0,
                           'cls loss should be non-zero')
        self.assertGreater(onegt_box_loss.item(), 0,
                           'box loss should be non-zero')

        # test num_class = 1
        self.head_module['num_classes'] = 1
        head = RTMDetHead(head_module=self.head_module, train_cfg=train_cfg)
        gt_instances = InstanceData(
            bboxes=torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
            labels=torch.LongTensor([0]))

        cls_scores, bbox_preds = head.forward(feat)

        one_gt_losses = head.loss_by_feat(cls_scores, bbox_preds,
                                          [gt_instances], img_metas)
        onegt_cls_loss = one_gt_losses['loss_cls'].sum()
        onegt_box_loss = one_gt_losses['loss_bbox'].sum()
        self.assertGreater(onegt_cls_loss.item(), 0,
                           'cls loss should be non-zero')
        self.assertGreater(onegt_box_loss.item(), 0,
                           'box loss should be non-zero')


class TestRTMDetInsHead(TestCase):

    def setUp(self):
        self.head_module = dict(
            type='RTMDetInsSepBNHeadModule',
            num_classes=4,
            in_channels=1,
            stacked_convs=1,
            feat_channels=64,
            featmap_strides=[4, 8, 16],
            num_prototypes=8,
            dyconv_channels=8,
            num_dyconvs=3,
            share_conv=True,
            use_sigmoid_cls=True)

    def test_init_weights(self):
        head = RTMDetInsSepBNHead(head_module=self.head_module)
        head.head_module.init_weights()

    def test_predict_by_feat(self):
        s = 256
        img_metas = [{
            'img_shape': (s, s, 3),
            'ori_shape': (s, s, 3),
            'scale_factor': (1.0, 1.0),
            'pad_param': np.array([0., 0., 0., 0.])
        }]
        test_cfg = dict(
            multi_label=False,
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.6),
            max_per_img=100,
            mask_thr_binary=0.5)
        test_cfg = Config(test_cfg)

        head = RTMDetInsSepBNHead(
            head_module=self.head_module, test_cfg=test_cfg)
        feat = [
            torch.rand(1, 1, s // feat_size, s // feat_size)
            for feat_size in [4, 8, 16]
        ]
        cls_scores, bbox_preds, kernel_preds, mask_feat = head.forward(feat)
        head.predict_by_feat(
            cls_scores,
            bbox_preds,
            kernel_preds,
            mask_feat,
            batch_img_metas=img_metas,
            cfg=test_cfg,
            rescale=True,
            with_nms=True)

        img_metas_without_pad_param = [{
            'img_shape': (s, s, 3),
            'ori_shape': (s, s, 3),
            'scale_factor': (1.0, 1.0)
        }]
        head.predict_by_feat(
            cls_scores,
            bbox_preds,
            kernel_preds,
            mask_feat,
            batch_img_metas=img_metas_without_pad_param,
            cfg=test_cfg,
            rescale=True,
            with_nms=True)

        with self.assertRaises(AssertionError):
            head.predict_by_feat(
                cls_scores,
                bbox_preds,
                kernel_preds,
                mask_feat,
                batch_img_metas=img_metas,
                cfg=test_cfg,
                rescale=False,
                with_nms=False)
