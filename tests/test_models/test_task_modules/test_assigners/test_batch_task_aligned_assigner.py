# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmyolo.models.task_modules.assigners import BatchTaskAlignedAssigner


class TestBatchTaskAlignedAssigner(TestCase):

    def test_batch_task_aligned_assigner(self):
        assigner = BatchTaskAlignedAssigner(
            num_classes=80,
            alpha=1,
            beta=6,
            topk=13,
            iou_calculator=dict(type='mmdet.BboxOverlaps2D'),
            eps=1e-9
        )
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

        assign_result = assigner.forward(priors_points, gt_labels,
                                         gt_bboxes, pad_bbox_flag,
                                         pred_scores, pred_bboxes)

        assigned_labels = assign_result['assigned_labels']
        assigned_bboxes = assign_result['assigned_bboxes']
        assigned_scores = assign_result['assigned_scores']
        fg_mask_pre_prior = assign_result['fg_mask_pre_prior']

        self.assertEqual(assigned_labels.shape, torch.Size([batch_size, 8400]))
        self.assertEqual(assigned_bboxes.shape,
                         torch.Size([batch_size, 8400, 4]))
        self.assertEqual(assigned_scores.shape,
                         torch.Size([batch_size, 8400, 80]))
        self.assertEqual(fg_mask_pre_prior.shape,
                         torch.Size([batch_size, 8400]))
