# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmyolo.models.task_modules.assigners import BatchATSSAssigner


class TestBatchATSSAssigner(TestCase):

    def test_batch_atss_assigner(self):
        batch_atss_assigner = BatchATSSAssigner(topk=9,
                                                iou2d_calculator=dict(type='mmdet.BboxOverlaps2D'),
                                                num_classes=80)
        batch_size = 16
        priors = torch.FloatTensor([
            [-16., -16., 24., 24.],
            [-8., -16., 32., 24.],
            [0., -16., 40., 24.],
            [8., -16., 48., 24.]
        ]).repeat(2100, 1)
        gt_bboxes = torch.FloatTensor([
            [0, 0, 60, 93],
            [229, 0, 532, 157],
        ]).unsqueeze(0).repeat(batch_size, 1, 1)
        gt_labels = torch.LongTensor([[0], [11]]).unsqueeze(0).repeat(batch_size, 1, 1)
        num_level_bboxes = [6400, 1600, 400]
        pad_bbox_flag = torch.FloatTensor([[1], [0]]).unsqueeze(0).repeat(batch_size, 1, 1)
        pred_bboxes = torch.FloatTensor([
            [-4., -4., 12., 12.],
            [4., -4., 20., 12.],
            [12., -4., 28., 12.],
            [20., -4., 36., 12.]
        ]).unsqueeze(0).repeat(batch_size, 2100, 1)
        batch_assign_result = batch_atss_assigner.forward(priors, num_level_bboxes, gt_labels, gt_bboxes,
                                                          pad_bbox_flag, pred_bboxes)
        assigned_labels, assigned_bboxes, assigned_scores, fg_mask_pre_prior_bool = batch_assign_result
        self.assertEqual(assigned_labels.shape, torch.Size([batch_size, 8400]))
        self.assertEqual(assigned_bboxes.shape, torch.Size([batch_size, 8400, 4]))
        self.assertEqual(assigned_scores.shape, torch.Size([batch_size, 8400, 80]))
        self.assertEqual(fg_mask_pre_prior_bool.shape, torch.Size([batch_size, 8400]))

    def test_batch_atss_assigner_with_empty_gt(self):
        """Test corner case where an image might have no true detections."""
        batch_atss_assigner = BatchATSSAssigner(topk=9,
                                                iou2d_calculator=dict(type='mmdet.BboxOverlaps2D'),
                                                num_classes=80)
        batch_size = 16
        priors = torch.FloatTensor([
            [-16., -16., 24., 24.],
            [-8., -16., 32., 24.],
            [0., -16., 40., 24.],
            [8., -16., 48., 24.]
        ]).repeat(2100, 1)
        num_level_bboxes = [6400, 1600, 400]
        pad_bbox_flag = torch.FloatTensor([[1], [0]]).unsqueeze(0).repeat(batch_size, 1, 1)
        pred_bboxes = torch.FloatTensor([
            [-4., -4., 12., 12.],
            [4., -4., 20., 12.],
            [12., -4., 28., 12.],
            [20., -4., 36., 12.]
        ]).unsqueeze(0).repeat(batch_size, 2100, 1)

        gt_bboxes = torch.empty(batch_size, 2, 4)
        gt_labels = torch.empty(batch_size, 2, 1)

        batch_assign_result = batch_atss_assigner.forward(priors, num_level_bboxes, gt_labels, gt_bboxes,
                                                          pad_bbox_flag, pred_bboxes)

        assigned_labels, assigned_bboxes, assigned_scores, fg_mask_pre_prior_bool = batch_assign_result
        self.assertEqual(assigned_labels.shape, torch.Size([batch_size, 8400]))
        self.assertEqual(assigned_bboxes.shape, torch.Size([batch_size, 8400, 4]))
        self.assertEqual(assigned_scores.shape, torch.Size([batch_size, 8400, 80]))
        self.assertEqual(fg_mask_pre_prior_bool.shape, torch.Size([batch_size, 8400]))

        self.assertTrue(bool(batch_assign_result[1].all()) == 0.)
        self.assertTrue(bool(batch_assign_result[2].all()) == 0.)
        self.assertTrue(bool(batch_assign_result[3].all()) is False)

    def test_batch_atss_assigner_with_empty_boxes(self):
        """Test corner case where a network might predict no boxes."""
        batch_atss_assigner = BatchATSSAssigner(topk=9,
                                                iou2d_calculator=dict(type='mmdet.BboxOverlaps2D'),
                                                num_classes=80)
        batch_size = 16
        priors = torch.empty(0, 1)
        gt_bboxes = torch.FloatTensor([
            [0, 0, 60, 93],
            [229, 0, 532, 157],
        ]).unsqueeze(0).repeat(batch_size, 1, 1)
        gt_labels = torch.LongTensor([[0], [11]]).unsqueeze(0).repeat(batch_size, 1, 1)
        num_level_bboxes = [0, 0, 0]
        pad_bbox_flag = torch.FloatTensor([[1], [0]]).unsqueeze(0).repeat(batch_size, 1, 1)
        pred_bboxes = torch.FloatTensor([
            [-4., -4., 12., 12.],
            [4., -4., 20., 12.],
            [12., -4., 28., 12.],
            [20., -4., 36., 12.]
        ]).unsqueeze(0).repeat(batch_size, 2100, 1)

        with self.assertRaises(AssertionError):
            batch_assign_result = batch_atss_assigner.forward(priors, num_level_bboxes, gt_labels, gt_bboxes,
                                                              pad_bbox_flag, pred_bboxes)

    def test_batch_atss_assigner_with_empty_boxes_and_gt(self):
        """Test corner case where a network might predict no boxes and no
        gt."""
        batch_atss_assigner = BatchATSSAssigner(topk=9,
                                                iou2d_calculator=dict(type='mmdet.BboxOverlaps2D'),
                                                num_classes=80)
        batch_size = 16
        priors = torch.empty(0, 1)
        gt_bboxes = torch.empty(batch_size, 2, 4)
        gt_labels = torch.empty(batch_size, 2, 1)
        num_level_bboxes = [0, 0, 0]
        pad_bbox_flag = torch.empty(batch_size, 2, 1)
        pred_bboxes = torch.empty(batch_size, 8400, 4)
        with self.assertRaises(AssertionError):
            batch_assign_result = batch_atss_assigner.forward(priors, num_level_bboxes, gt_labels, gt_bboxes,
                                                              pad_bbox_flag, pred_bboxes)
