# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.structures import InstanceData
from mmengine.testing import assert_allclose

from mmyolo.models.task_modules.assigners import PoseSimOTAAssigner


class TestPoseSimOTAAssigner(TestCase):

    def test_assign(self):
        assigner = PoseSimOTAAssigner(
            center_radius=2.5,
            candidate_topk=1,
            iou_weight=3.0,
            cls_weight=1.0,
            iou_calculator=dict(type='mmdet.BboxOverlaps2D'))
        pred_instances = InstanceData(
            bboxes=torch.Tensor([[23, 23, 43, 43] + [1] * 51,
                                 [4, 5, 6, 7] + [1] * 51]),
            scores=torch.FloatTensor([[0.2], [0.8]]),
            priors=torch.Tensor([[30, 30, 8, 8], [4, 5, 6, 7]]))
        gt_instances = InstanceData(
            bboxes=torch.Tensor([[23, 23, 43, 43]]),
            labels=torch.LongTensor([0]),
            keypoints_visible=torch.Tensor([[
                1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 0., 0., 0., 0., 0.,
                0.
            ]]),
            keypoints=torch.Tensor([[[30, 30], [30, 30], [30, 30], [30, 30],
                                     [30, 30], [30, 30], [30, 30], [30, 30],
                                     [30, 30], [30, 30], [30, 30], [30, 30],
                                     [30, 30], [30, 30], [30, 30], [30, 30],
                                     [30, 30]]]))
        assign_result = assigner.assign(
            pred_instances=pred_instances, gt_instances=gt_instances)

        expected_gt_inds = torch.LongTensor([1, 0])
        assert_allclose(assign_result.gt_inds, expected_gt_inds)

    def test_assign_with_no_valid_bboxes(self):
        assigner = PoseSimOTAAssigner(
            center_radius=2.5,
            candidate_topk=1,
            iou_weight=3.0,
            cls_weight=1.0,
            iou_calculator=dict(type='mmdet.BboxOverlaps2D'))
        pred_instances = InstanceData(
            bboxes=torch.Tensor([[123, 123, 143, 143], [114, 151, 161, 171]]),
            scores=torch.FloatTensor([[0.2], [0.8]]),
            priors=torch.Tensor([[30, 30, 8, 8], [55, 55, 8, 8]]))
        gt_instances = InstanceData(
            bboxes=torch.Tensor([[0, 0, 1, 1]]),
            labels=torch.LongTensor([0]),
            keypoints_visible=torch.zeros((1, 17)),
            keypoints=torch.zeros((1, 17, 2)))
        assign_result = assigner.assign(
            pred_instances=pred_instances, gt_instances=gt_instances)

        expected_gt_inds = torch.LongTensor([0, 0])
        assert_allclose(assign_result.gt_inds, expected_gt_inds)

    def test_assign_with_empty_gt(self):
        assigner = PoseSimOTAAssigner(
            center_radius=2.5,
            candidate_topk=1,
            iou_weight=3.0,
            cls_weight=1.0,
            iou_calculator=dict(type='mmdet.BboxOverlaps2D'))
        pred_instances = InstanceData(
            bboxes=torch.Tensor([[[30, 40, 50, 60]], [[4, 5, 6, 7]]]),
            scores=torch.FloatTensor([[0.2], [0.8]]),
            priors=torch.Tensor([[0, 12, 23, 34], [4, 5, 6, 7]]))
        gt_instances = InstanceData(
            bboxes=torch.empty(0, 4),
            labels=torch.empty(0),
            keypoints_visible=torch.empty(0, 17),
            keypoints=torch.empty(0, 17, 2))

        assign_result = assigner.assign(
            pred_instances=pred_instances, gt_instances=gt_instances)
        expected_gt_inds = torch.LongTensor([0, 0])
        assert_allclose(assign_result.gt_inds, expected_gt_inds)
