# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmyolo.models.task_modules.assigners import BatchTaskAlignedAssigner


class TestBatchTaskAlignedAssigner(TestCase):

    def test_batch_task_aligned_assigner(self):
        assigner = BatchTaskAlignedAssigner(
            topk=13,
            iou_calculator=dict(type='mmdet.BboxOverlaps2D'),
            alpha=1,
            beta=6)
        batch_size = 16
        pred_scores = torch.FloatTensor([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4],
                                         [0.4, 0.5]]).unsqueeze(0).repeat(
                                             batch_size, 2100, 40)
        priors_points = torch.FloatTensor([[4., 4.], [12., 4.], [20., 4.],
                                           [28., 4.]]).repeat(2100, 1)
        gt_bboxes = torch.FloatTensor([
            [0, 0, 60, 93],
            [229, 0, 532, 157],
        ]).unsqueeze(0).repeat(batch_size, 20, 1)
        gt_labels = torch.LongTensor([[0], [1]]).unsqueeze(0).repeat(
            batch_size, 20, 1)
        pad_bbox_flag = torch.FloatTensor([[1], [0]]).unsqueeze(0).repeat(
            batch_size, 20, 1)
        pred_bboxes = torch.FloatTensor([[-4., -4., 12., 12.],
                                         [4., -4., 20., 12.],
                                         [12., -4., 28., 12.],
                                         [20., -4., 36.,
                                          12.]]).unsqueeze(0).repeat(
                                              batch_size, 2100, 1)

        assign_result = assigner.forward(pred_scores, pred_bboxes,
                                         priors_points, gt_labels, gt_bboxes,
                                         pad_bbox_flag)
        (assigned_labels, assigned_bboxes, assigned_scores,
         fg_mask_pre_prior_bool) = assign_result

        self.assertEqual(assigned_labels.shape, torch.Size([batch_size, 8400]))
        self.assertEqual(assigned_bboxes.shape,
                         torch.Size([batch_size, 8400, 4]))
        self.assertEqual(assigned_scores.shape,
                         torch.Size([batch_size, 8400, 80]))
        self.assertEqual(fg_mask_pre_prior_bool.shape,
                         torch.Size([batch_size, 8400]))
