# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.config import Config
from mmengine.model import bias_init_with_prob
from mmengine.testing import assert_allclose

from mmyolo.models.dense_heads import YOLOXHead, YOLOXPoseHead
from mmyolo.utils import register_all_modules

register_all_modules()


class TestYOLOXHead(TestCase):

    def setUp(self):
        self.head_module = dict(
            type='YOLOXHeadModule',
            num_classes=4,
            in_channels=1,
            stacked_convs=1,
        )

    def test_init_weights(self):
        head = YOLOXHead(head_module=self.head_module)
        head.head_module.init_weights()
        bias_init = bias_init_with_prob(0.01)
        for conv_cls, conv_obj in zip(head.head_module.multi_level_conv_cls,
                                      head.head_module.multi_level_conv_obj):
            assert_allclose(conv_cls.bias.data,
                            torch.ones_like(conv_cls.bias.data) * bias_init)
            assert_allclose(conv_obj.bias.data,
                            torch.ones_like(conv_obj.bias.data) * bias_init)

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

        head = YOLOXHead(head_module=self.head_module, test_cfg=test_cfg)
        feat = [
            torch.rand(1, 1, s // feat_size, s // feat_size)
            for feat_size in [4, 8, 16]
        ]
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
            'scale_factor': 1,
        }]
        train_cfg = Config(
            dict(
                assigner=dict(
                    type='mmdet.SimOTAAssigner',
                    iou_calculator=dict(type='mmdet.BboxOverlaps2D'),
                    center_radius=2.5,
                    candidate_topk=10,
                    iou_weight=3.0,
                    cls_weight=1.0)))

        head = YOLOXHead(head_module=self.head_module, train_cfg=train_cfg)
        assert not head.use_bbox_aux

        feat = [
            torch.rand(1, 1, s // feat_size, s // feat_size)
            for feat_size in [4, 8, 16]
        ]
        cls_scores, bbox_preds, objectnesses = head.forward(feat)

        # Test that empty ground truth encourages the network to predict
        # background
        gt_instances = torch.empty((0, 6))

        empty_gt_losses = head.loss_by_feat(cls_scores, bbox_preds,
                                            objectnesses, gt_instances,
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
        head = YOLOXHead(head_module=self.head_module, train_cfg=train_cfg)
        head.use_bbox_aux = True
        gt_instances = torch.Tensor(
            [[0, 2, 23.6667, 23.8757, 238.6326, 151.8874]])

        one_gt_losses = head.loss_by_feat(cls_scores, bbox_preds, objectnesses,
                                          gt_instances, img_metas)
        onegt_cls_loss = one_gt_losses['loss_cls'].sum()
        onegt_box_loss = one_gt_losses['loss_bbox'].sum()
        onegt_obj_loss = one_gt_losses['loss_obj'].sum()
        onegt_l1_loss = one_gt_losses['loss_bbox_aux'].sum()
        self.assertGreater(onegt_cls_loss.item(), 0,
                           'cls loss should be non-zero')
        self.assertGreater(onegt_box_loss.item(), 0,
                           'box loss should be non-zero')
        self.assertGreater(onegt_obj_loss.item(), 0,
                           'obj loss should be non-zero')
        self.assertGreater(onegt_l1_loss.item(), 0,
                           'l1 loss should be non-zero')

        # Test groud truth out of bound
        gt_instances = torch.Tensor(
            [[0, 2, s * 4, s * 4, s * 4 + 10, s * 4 + 10]])
        empty_gt_losses = head.loss_by_feat(cls_scores, bbox_preds,
                                            objectnesses, gt_instances,
                                            img_metas)
        # When gt_bboxes out of bound, the assign results should be empty,
        # so the cls and bbox loss should be zero.
        empty_cls_loss = empty_gt_losses['loss_cls'].sum()
        empty_box_loss = empty_gt_losses['loss_bbox'].sum()
        empty_obj_loss = empty_gt_losses['loss_obj'].sum()
        self.assertEqual(
            empty_cls_loss.item(), 0,
            'there should be no cls loss when gt_bboxes out of bound')
        self.assertEqual(
            empty_box_loss.item(), 0,
            'there should be no box loss when gt_bboxes out of bound')
        self.assertGreater(empty_obj_loss.item(), 0,
                           'objectness loss should be non-zero')


class TestYOLOXPoseHead(TestCase):

    def setUp(self):
        self.head_module = dict(
            type='YOLOXPoseHeadModule',
            num_classes=1,
            num_keypoints=17,
            in_channels=1,
            stacked_convs=1,
        )
        self.train_cfg = Config(
            dict(
                assigner=dict(
                    type='PoseSimOTAAssigner',
                    center_radius=2.5,
                    oks_weight=3.0,
                    iou_calculator=dict(type='mmdet.BboxOverlaps2D'),
                    oks_calculator=dict(
                        type='OksLoss',
                        metainfo='configs/_base_/pose/coco.py'))))
        self.loss_pose = Config(
            dict(
                type='OksLoss',
                metainfo='configs/_base_/pose/coco.py',
                loss_weight=30.0))

    def test_init_weights(self):
        head = YOLOXPoseHead(
            head_module=self.head_module,
            loss_pose=self.loss_pose,
            train_cfg=self.train_cfg)
        head.head_module.init_weights()
        bias_init = bias_init_with_prob(0.01)
        for conv_cls, conv_obj, conv_vis in zip(
                head.head_module.multi_level_conv_cls,
                head.head_module.multi_level_conv_obj,
                head.head_module.multi_level_conv_vis):
            assert_allclose(conv_cls.bias.data,
                            torch.ones_like(conv_cls.bias.data) * bias_init)
            assert_allclose(conv_obj.bias.data,
                            torch.ones_like(conv_obj.bias.data) * bias_init)
            assert_allclose(conv_vis.bias.data,
                            torch.ones_like(conv_vis.bias.data) * bias_init)

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

        head = YOLOXPoseHead(
            head_module=self.head_module,
            loss_pose=self.loss_pose,
            train_cfg=self.train_cfg,
            test_cfg=test_cfg)
        feat = [
            torch.rand(1, 1, s // feat_size, s // feat_size)
            for feat_size in [4, 8, 16]
        ]
        cls_scores, bbox_preds, objectnesses, \
            offsets_preds, vis_preds = head.forward(feat)
        head.predict_by_feat(
            cls_scores,
            bbox_preds,
            objectnesses,
            offsets_preds,
            vis_preds,
            img_metas,
            cfg=test_cfg,
            rescale=True,
            with_nms=True)

    def test_loss_by_feat(self):
        s = 256
        img_metas = [{
            'img_shape': (s, s, 3),
            'scale_factor': 1,
        }]

        head = YOLOXPoseHead(
            head_module=self.head_module,
            loss_pose=self.loss_pose,
            train_cfg=self.train_cfg)
        assert not head.use_bbox_aux

        feat = [
            torch.rand(1, 1, s // feat_size, s // feat_size)
            for feat_size in [4, 8, 16]
        ]
        cls_scores, bbox_preds, objectnesses, \
            offsets_preds, vis_preds = head.forward(feat)

        # Test that empty ground truth encourages the network to predict
        # background
        gt_instances = torch.empty((0, 6))
        gt_keypoints = torch.empty((0, 17, 2))
        gt_keypoints_visible = torch.empty((0, 17))

        empty_gt_losses = head.loss_by_feat(cls_scores, bbox_preds,
                                            objectnesses, offsets_preds,
                                            vis_preds, gt_instances,
                                            gt_keypoints, gt_keypoints_visible,
                                            img_metas)
        # When there is no truth, the cls loss should be nonzero but there
        # should be no box loss.
        empty_cls_loss = empty_gt_losses['loss_cls'].sum()
        empty_box_loss = empty_gt_losses['loss_bbox'].sum()
        empty_obj_loss = empty_gt_losses['loss_obj'].sum()
        empty_loss_kpt = empty_gt_losses['loss_kpt'].sum()
        empty_loss_vis = empty_gt_losses['loss_vis'].sum()
        self.assertEqual(
            empty_cls_loss.item(), 0,
            'there should be no cls loss when there are no true boxes')
        self.assertEqual(
            empty_box_loss.item(), 0,
            'there should be no box loss when there are no true boxes')
        self.assertGreater(empty_obj_loss.item(), 0,
                           'objectness loss should be non-zero')
        self.assertEqual(
            empty_loss_kpt.item(), 0,
            'there should be no kpt loss when there are no true keypoints')
        self.assertEqual(
            empty_loss_vis.item(), 0,
            'there should be no vis loss when there are no true keypoints')
        # When truth is non-empty then both cls and box loss should be nonzero
        # for random inputs
        head = YOLOXPoseHead(
            head_module=self.head_module,
            loss_pose=self.loss_pose,
            train_cfg=self.train_cfg)
        gt_instances = torch.Tensor(
            [[0, 0, 23.6667, 23.8757, 238.6326, 151.8874]])
        gt_keypoints = torch.Tensor([[[317.1519,
                                       429.8433], [338.3080, 416.9187],
                                      [298.9951,
                                       403.8911], [102.7025, 273.1329],
                                      [255.4321,
                                       404.8712], [400.0422, 554.4373],
                                      [167.7857,
                                       516.7591], [397.4943, 737.4575],
                                      [116.3247,
                                       674.5684], [102.7025, 273.1329],
                                      [66.0319,
                                       808.6383], [102.7025, 273.1329],
                                      [157.6150,
                                       819.1249], [102.7025, 273.1329],
                                      [102.7025,
                                       273.1329], [102.7025, 273.1329],
                                      [102.7025, 273.1329]]])
        gt_keypoints_visible = torch.Tensor([[
            1., 1., 1., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
        ]])

        one_gt_losses = head.loss_by_feat(cls_scores, bbox_preds, objectnesses,
                                          offsets_preds, vis_preds,
                                          gt_instances, gt_keypoints,
                                          gt_keypoints_visible, img_metas)
        onegt_cls_loss = one_gt_losses['loss_cls'].sum()
        onegt_box_loss = one_gt_losses['loss_bbox'].sum()
        onegt_obj_loss = one_gt_losses['loss_obj'].sum()
        onegt_loss_kpt = one_gt_losses['loss_kpt'].sum()
        onegt_loss_vis = one_gt_losses['loss_vis'].sum()

        self.assertGreater(onegt_cls_loss.item(), 0,
                           'cls loss should be non-zero')
        self.assertGreater(onegt_box_loss.item(), 0,
                           'box loss should be non-zero')
        self.assertGreater(onegt_obj_loss.item(), 0,
                           'obj loss should be non-zero')
        self.assertGreater(onegt_loss_kpt.item(), 0,
                           'kpt loss should be non-zero')
        self.assertGreater(onegt_loss_vis.item(), 0,
                           'vis loss should be non-zero')

        # Test groud truth out of bound
        gt_instances = torch.Tensor(
            [[0, 2, s * 4, s * 4, s * 4 + 10, s * 4 + 10]])
        gt_keypoints = torch.Tensor([[[s * 4, s * 4 + 10], [s * 4, s * 4 + 10],
                                      [s * 4, s * 4 + 10], [s * 4, s * 4 + 10],
                                      [s * 4, s * 4 + 10], [s * 4, s * 4 + 10],
                                      [s * 4, s * 4 + 10], [s * 4, s * 4 + 10],
                                      [s * 4, s * 4 + 10], [s * 4, s * 4 + 10],
                                      [s * 4, s * 4 + 10], [s * 4, s * 4 + 10],
                                      [s * 4, s * 4 + 10], [s * 4, s * 4 + 10],
                                      [s * 4, s * 4 + 10], [s * 4, s * 4 + 10],
                                      [s * 4, s * 4 + 10]]])
        empty_gt_losses = head.loss_by_feat(cls_scores, bbox_preds,
                                            objectnesses, offsets_preds,
                                            vis_preds, gt_instances,
                                            gt_keypoints, gt_keypoints_visible,
                                            img_metas)
        # When gt_bboxes out of bound, the assign results should be empty,
        # so the cls and bbox loss should be zero.
        empty_cls_loss = empty_gt_losses['loss_cls'].sum()
        empty_box_loss = empty_gt_losses['loss_bbox'].sum()
        empty_obj_loss = empty_gt_losses['loss_obj'].sum()
        empty_kpt_loss = empty_gt_losses['loss_kpt'].sum()
        empty_vis_loss = empty_gt_losses['loss_vis'].sum()
        self.assertEqual(
            empty_cls_loss.item(), 0,
            'there should be no cls loss when gt_bboxes out of bound')
        self.assertEqual(
            empty_box_loss.item(), 0,
            'there should be no box loss when gt_bboxes out of bound')
        self.assertGreater(empty_obj_loss.item(), 0,
                           'objectness loss should be non-zero')
        self.assertEqual(empty_kpt_loss.item(), 0,
                         'kps loss should be non-zero')
        self.assertEqual(empty_vis_loss.item(), 0,
                         'vis loss should be non-zero')
