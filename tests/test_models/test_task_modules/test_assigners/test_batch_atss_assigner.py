# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmyolo.models.task_modules.assigners import BatchATSSAssigner


class TestBatchATSSAssigner(TestCase):

    def test_batch_atss_assigner(self):
        num_classes = 2
        batch_size = 2
        batch_atss_assigner = BatchATSSAssigner(
            topk=3,
            iou_calculator=dict(type='mmdet.BboxOverlaps2D'),
            num_classes=num_classes)
        priors = torch.FloatTensor([
            [4., 4., 8., 8.],
            [12., 4., 8., 8.],
            [20., 4., 8., 8.],
            [28., 4., 8., 8.],
        ]).repeat(21, 1)
        gt_bboxes = torch.FloatTensor([
            [0, 0, 60, 93],
            [229, 0, 532, 157],
        ]).unsqueeze(0).repeat(batch_size, 1, 1)
        gt_labels = torch.LongTensor([
            [0],
            [11],
        ]).unsqueeze(0).repeat(batch_size, 1, 1)
        num_level_bboxes = [64, 16, 4]
        pad_bbox_flag = torch.FloatTensor([
            [1],
            [0],
        ]).unsqueeze(0).repeat(batch_size, 1, 1)
        pred_bboxes = torch.FloatTensor([
            [-4., -4., 12., 12.],
            [4., -4., 20., 12.],
            [12., -4., 28., 12.],
            [20., -4., 36., 12.],
        ]).unsqueeze(0).repeat(batch_size, 21, 1)
        batch_assign_result = batch_atss_assigner.forward(
            pred_bboxes, priors, num_level_bboxes, gt_labels, gt_bboxes,
            pad_bbox_flag)

        assigned_labels = batch_assign_result['assigned_labels']
        assigned_bboxes = batch_assign_result['assigned_bboxes']
        assigned_scores = batch_assign_result['assigned_scores']
        fg_mask_pre_prior = batch_assign_result['fg_mask_pre_prior']

        self.assertEqual(assigned_labels.shape, torch.Size([batch_size, 84]))
        self.assertEqual(assigned_bboxes.shape, torch.Size([batch_size, 84,
                                                            4]))
        self.assertEqual(assigned_scores.shape,
                         torch.Size([batch_size, 84, num_classes]))
        self.assertEqual(fg_mask_pre_prior.shape, torch.Size([batch_size, 84]))

    def test_batch_atss_assigner_with_empty_gt(self):
        """Test corner case where an image might have no true detections."""
        num_classes = 2
        batch_size = 2
        batch_atss_assigner = BatchATSSAssigner(
            topk=3,
            iou_calculator=dict(type='mmdet.BboxOverlaps2D'),
            num_classes=num_classes)
        priors = torch.FloatTensor([
            [4., 4., 8., 8.],
            [12., 4., 8., 8.],
            [20., 4., 8., 8.],
            [28., 4., 8., 8.],
        ]).repeat(21, 1)
        num_level_bboxes = [64, 16, 4]
        pad_bbox_flag = torch.FloatTensor([
            [1],
            [0],
        ]).unsqueeze(0).repeat(batch_size, 1, 1)
        pred_bboxes = torch.FloatTensor([
            [-4., -4., 12., 12.],
            [4., -4., 20., 12.],
            [12., -4., 28., 12.],
            [20., -4., 36., 12.],
        ]).unsqueeze(0).repeat(batch_size, 21, 1)

        gt_bboxes = torch.zeros(batch_size, 0, 4)
        gt_labels = torch.zeros(batch_size, 0, 1)

        batch_assign_result = batch_atss_assigner.forward(
            pred_bboxes, priors, num_level_bboxes, gt_labels, gt_bboxes,
            pad_bbox_flag)

        assigned_labels = batch_assign_result['assigned_labels']
        assigned_bboxes = batch_assign_result['assigned_bboxes']
        assigned_scores = batch_assign_result['assigned_scores']
        fg_mask_pre_prior = batch_assign_result['fg_mask_pre_prior']

        self.assertEqual(assigned_labels.shape, torch.Size([batch_size, 84]))
        self.assertEqual(assigned_bboxes.shape, torch.Size([batch_size, 84,
                                                            4]))
        self.assertEqual(assigned_scores.shape,
                         torch.Size([batch_size, 84, num_classes]))
        self.assertEqual(fg_mask_pre_prior.shape, torch.Size([batch_size, 84]))

    def test_batch_atss_assigner_with_empty_boxs(self):
        """Test corner case where a network might predict no boxes."""
        num_classes = 2
        batch_size = 2
        batch_atss_assigner = BatchATSSAssigner(
            topk=3,
            iou_calculator=dict(type='mmdet.BboxOverlaps2D'),
            num_classes=num_classes)
        priors = torch.zeros(84, 4)
        gt_bboxes = torch.FloatTensor([
            [0, 0, 60, 93],
            [229, 0, 532, 157],
        ]).unsqueeze(0).repeat(batch_size, 1, 1)
        gt_labels = torch.LongTensor([
            [0],
            [11],
        ]).unsqueeze(0).repeat(batch_size, 1, 1)
        num_level_bboxes = [64, 16, 4]
        pad_bbox_flag = torch.FloatTensor([[1], [0]]).unsqueeze(0).repeat(
            batch_size, 1, 1)
        pred_bboxes = torch.FloatTensor([
            [-4., -4., 12., 12.],
            [4., -4., 20., 12.],
            [12., -4., 28., 12.],
            [20., -4., 36., 12.],
        ]).unsqueeze(0).repeat(batch_size, 21, 1)

        batch_assign_result = batch_atss_assigner.forward(
            pred_bboxes, priors, num_level_bboxes, gt_labels, gt_bboxes,
            pad_bbox_flag)
        assigned_labels = batch_assign_result['assigned_labels']
        assigned_bboxes = batch_assign_result['assigned_bboxes']
        assigned_scores = batch_assign_result['assigned_scores']
        fg_mask_pre_prior = batch_assign_result['fg_mask_pre_prior']

        self.assertEqual(assigned_labels.shape, torch.Size([batch_size, 84]))
        self.assertEqual(assigned_bboxes.shape, torch.Size([batch_size, 84,
                                                            4]))
        self.assertEqual(assigned_scores.shape,
                         torch.Size([batch_size, 84, num_classes]))
        self.assertEqual(fg_mask_pre_prior.shape, torch.Size([batch_size, 84]))

    def test_batch_atss_assigner_with_empty_boxes_and_gt(self):
        """Test corner case where a network might predict no boxes and no
        gt."""
        num_classes = 2
        batch_size = 2
        batch_atss_assigner = BatchATSSAssigner(
            topk=3,
            iou_calculator=dict(type='mmdet.BboxOverlaps2D'),
            num_classes=num_classes)
        priors = torch.zeros(84, 4)
        gt_bboxes = torch.zeros(batch_size, 0, 4)
        gt_labels = torch.zeros(batch_size, 0, 1)
        num_level_bboxes = [64, 16, 4]
        pad_bbox_flag = torch.zeros(batch_size, 0, 1)
        pred_bboxes = torch.zeros(batch_size, 0, 4)

        batch_assign_result = batch_atss_assigner.forward(
            pred_bboxes, priors, num_level_bboxes, gt_labels, gt_bboxes,
            pad_bbox_flag)
        assigned_labels = batch_assign_result['assigned_labels']
        assigned_bboxes = batch_assign_result['assigned_bboxes']
        assigned_scores = batch_assign_result['assigned_scores']
        fg_mask_pre_prior = batch_assign_result['fg_mask_pre_prior']

        self.assertEqual(assigned_labels.shape, torch.Size([batch_size, 84]))
        self.assertEqual(assigned_bboxes.shape, torch.Size([batch_size, 84,
                                                            4]))
        self.assertEqual(assigned_scores.shape,
                         torch.Size([batch_size, 84, num_classes]))
        self.assertEqual(fg_mask_pre_prior.shape, torch.Size([batch_size, 84]))
