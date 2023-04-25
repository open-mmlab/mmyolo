# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import torch
from mmengine.config import Config
from mmengine.structures import InstanceData

from mmyolo.models.dense_heads import YOLOv5Head, YOLOv5InsHead
from mmyolo.utils import register_all_modules

register_all_modules()


class TestYOLOv5Head(TestCase):

    def setUp(self):
        self.head_module = dict(
            type='YOLOv5HeadModule',
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

        head = YOLOv5Head(head_module=self.head_module, test_cfg=test_cfg)

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

        head = YOLOv5Head(head_module=self.head_module)

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
        head = YOLOv5Head(head_module=self.head_module)
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
        head = YOLOv5Head(head_module=self.head_module)
        gt_instances = InstanceData(
            bboxes=torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
            labels=torch.LongTensor([0]))

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

    def test_loss_by_feat_with_ignore(self):
        s = 256
        img_metas = [{
            'img_shape': (s, s, 3),
            'batch_input_shape': (s, s),
            'scale_factor': 1,
        }]

        head = YOLOv5Head(head_module=self.head_module, ignore_iof_thr=0.8)

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
        # ignore boxes
        gt_instances_ignore = torch.tensor(
            [[0, 0, 69.7688, 0, 619.3611, 62.2711]], dtype=torch.float32)

        empty_gt_losses = head._loss_by_feat_with_ignore(
            cls_scores, bbox_preds, objectnesses, [gt_instances], img_metas,
            gt_instances_ignore)
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
        head = YOLOv5Head(head_module=self.head_module, ignore_iof_thr=0.8)
        gt_instances = InstanceData(
            bboxes=torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
            labels=torch.LongTensor([1]))

        gt_instances_ignore = torch.tensor(
            [[0, 0, 69.7688, 0, 619.3611, 62.2711]], dtype=torch.float32)

        one_gt_losses = head._loss_by_feat_with_ignore(cls_scores, bbox_preds,
                                                       objectnesses,
                                                       [gt_instances],
                                                       img_metas,
                                                       gt_instances_ignore)
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
        head = YOLOv5Head(head_module=self.head_module, ignore_iof_thr=0.8)
        gt_instances = InstanceData(
            bboxes=torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
            labels=torch.LongTensor([0]))

        gt_instances_ignore = torch.tensor(
            [[0, 0, 69.7688, 0, 619.3611, 62.2711]], dtype=torch.float32)

        one_gt_losses = head._loss_by_feat_with_ignore(cls_scores, bbox_preds,
                                                       objectnesses,
                                                       [gt_instances],
                                                       img_metas,
                                                       gt_instances_ignore)
        onegt_cls_loss = one_gt_losses['loss_cls'].sum()
        onegt_box_loss = one_gt_losses['loss_bbox'].sum()
        onegt_obj_loss = one_gt_losses['loss_obj'].sum()
        self.assertEqual(onegt_cls_loss.item(), 0,
                         'cls loss should be non-zero')
        self.assertGreater(onegt_box_loss.item(), 0,
                           'box loss should be non-zero')
        self.assertGreater(onegt_obj_loss.item(), 0,
                           'obj loss should be non-zero')


class TestYOLOv5InsHead(TestCase):

    def setUp(self):
        self.head_module = dict(
            type='YOLOv5InsHeadModule',
            num_classes=4,
            in_channels=[32, 64, 128],
            featmap_strides=[8, 16, 32],
            mask_channels=32,
            proto_channels=32,
            widen_factor=1.0)

    def test_init_weights(self):
        head = YOLOv5InsHead(head_module=self.head_module)
        head.head_module.init_weights()

    def test_predict_by_feat(self):
        s = 256
        img_metas = [{
            'img_shape': (s, s, 3),
            'ori_shape': (s, s, 3),
            'batch_input_shape': (s, s),
            'scale_factor': (1.0, 1.0),
        }]
        test_cfg = Config(
            dict(
                multi_label=True,
                nms_pre=30000,
                min_bbox_size=0,
                score_thr=0.001,
                nms=dict(type='nms', iou_threshold=0.6),
                max_per_img=300,
                mask_thr_binary=0.5))

        head = YOLOv5InsHead(head_module=self.head_module, test_cfg=test_cfg)
        head.eval()

        feat = []
        for i in range(len(self.head_module['in_channels'])):
            in_channel = self.head_module['in_channels'][i]
            feat_size = self.head_module['featmap_strides'][i]
            feat.append(
                torch.rand(1, in_channel, s // feat_size, s // feat_size))

        with torch.no_grad():
            res = head.forward(feat)
            cls_scores, bbox_preds, objectnesses,\
                coeff_preds, proto_preds = res
            head.predict_by_feat(
                cls_scores,
                bbox_preds,
                objectnesses,
                coeff_preds,
                proto_preds,
                img_metas,
                cfg=test_cfg,
                rescale=True,
                with_nms=True)

            with self.assertRaises(AssertionError):
                head.predict_by_feat(
                    cls_scores,
                    bbox_preds,
                    coeff_preds,
                    proto_preds,
                    img_metas,
                    cfg=test_cfg,
                    rescale=True,
                    with_nms=False)

    def test_loss_by_feat(self):
        s = 256
        img_metas = [{
            'img_shape': (s, s, 3),
            'batch_input_shape': (s, s),
            'scale_factor': 1,
        }]

        head = YOLOv5InsHead(head_module=self.head_module)
        rng = np.random.RandomState(0)

        feat = []
        for i in range(len(self.head_module['in_channels'])):
            in_channel = self.head_module['in_channels'][i]
            feat_size = self.head_module['featmap_strides'][i]
            feat.append(
                torch.rand(1, in_channel, s // feat_size, s // feat_size))

        cls_scores, bbox_preds, objectnesses,\
            coeff_preds, proto_preds = head.forward(feat)

        # Test that empty ground truth encourages the network to predict
        # background
        gt_bboxes_labels = torch.empty((0, 6))
        gt_masks = rng.rand(0, s // 4, s // 4)

        empty_gt_losses = head.loss_by_feat(cls_scores, bbox_preds,
                                            objectnesses, coeff_preds,
                                            proto_preds, gt_bboxes_labels,
                                            gt_masks, img_metas)
        # When there is no truth, the cls loss should be nonzero but there
        # should be no box loss.
        empty_cls_loss = empty_gt_losses['loss_cls'].sum()
        empty_box_loss = empty_gt_losses['loss_bbox'].sum()
        empty_obj_loss = empty_gt_losses['loss_obj'].sum()
        empty_mask_loss = empty_gt_losses['loss_mask'].sum()
        self.assertEqual(
            empty_cls_loss.item(), 0,
            'there should be no cls loss when there are no true boxes')
        self.assertEqual(
            empty_box_loss.item(), 0,
            'there should be no box loss when there are no true boxes')
        self.assertGreater(empty_obj_loss.item(), 0,
                           'objectness loss should be non-zero')
        self.assertEqual(
            empty_mask_loss.item(), 0,
            'there should be no mask loss when there are no true masks')

        # When truth is non-empty then both cls and box loss should be nonzero
        # for random inputs
        head = YOLOv5InsHead(head_module=self.head_module)

        bboxes = torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]])
        labels = torch.Tensor([1.])
        batch_id = torch.LongTensor([0])
        gt_bboxes_labels = torch.cat([batch_id[None], labels[None], bboxes],
                                     dim=1)
        gt_masks = torch.from_numpy(rng.rand(1, s // 4, s // 4)).int()

        one_gt_losses = head.loss_by_feat(cls_scores, bbox_preds, objectnesses,
                                          coeff_preds, proto_preds,
                                          gt_bboxes_labels, gt_masks,
                                          img_metas)
        onegt_cls_loss = one_gt_losses['loss_cls'].sum()
        onegt_box_loss = one_gt_losses['loss_bbox'].sum()
        onegt_obj_loss = one_gt_losses['loss_obj'].sum()
        onegt_mask_loss = one_gt_losses['loss_mask'].sum()
        self.assertGreater(onegt_cls_loss.item(), 0,
                           'cls loss should be non-zero')
        self.assertGreater(onegt_box_loss.item(), 0,
                           'box loss should be non-zero')
        self.assertGreater(onegt_obj_loss.item(), 0,
                           'obj loss should be non-zero')
        self.assertGreater(onegt_mask_loss.item(), 0,
                           'mask loss should be non-zero')

        # test num_class = 1
        self.head_module['num_classes'] = 1
        head = YOLOv5InsHead(head_module=self.head_module)
        bboxes = torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]])
        labels = torch.Tensor([1.])
        batch_id = torch.LongTensor([0])
        gt_bboxes_labels = torch.cat([batch_id[None], labels[None], bboxes],
                                     dim=1)
        gt_masks = torch.from_numpy(rng.rand(1, s // 4, s // 4)).int()

        one_gt_losses = head.loss_by_feat(cls_scores, bbox_preds, objectnesses,
                                          coeff_preds, proto_preds,
                                          gt_bboxes_labels, gt_masks,
                                          img_metas)
        onegt_cls_loss = one_gt_losses['loss_cls'].sum()
        onegt_box_loss = one_gt_losses['loss_bbox'].sum()
        onegt_obj_loss = one_gt_losses['loss_obj'].sum()
        onegt_mask_loss = one_gt_losses['loss_mask'].sum()
        self.assertEqual(onegt_cls_loss.item(), 0,
                         'cls loss should be non-zero')
        self.assertGreater(onegt_box_loss.item(), 0,
                           'box loss should be non-zero')
        self.assertGreater(onegt_obj_loss.item(), 0,
                           'obj loss should be non-zero')
        self.assertGreater(onegt_mask_loss.item(), 0,
                           'mask loss should be non-zero')
