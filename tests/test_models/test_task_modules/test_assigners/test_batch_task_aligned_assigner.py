# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmyolo.models.task_modules.assigners import BatchTaskAlignedAssigner


class TestBatchTaskAlignedAssigner(TestCase):

    def test_batch_task_aligned_assigner(self):
        batch_size = 2
        num_classes = 4
        assigner = BatchTaskAlignedAssigner(
            num_classes=num_classes, alpha=1, beta=6, topk=13, eps=1e-9)
        pred_scores = torch.FloatTensor([
            [0.1, 0.2],
            [0.2, 0.3],
            [0.3, 0.4],
            [0.4, 0.5],
        ]).unsqueeze(0).repeat(batch_size, 21, 1)
        priors = torch.FloatTensor([
            [0, 0, 4., 4.],
            [0, 0, 12., 4.],
            [0, 0, 20., 4.],
            [0, 0, 28., 4.],
        ]).repeat(21, 1)
        gt_bboxes = torch.FloatTensor([
            [0, 0, 60, 93],
            [229, 0, 532, 157],
        ]).unsqueeze(0).repeat(batch_size, 1, 1)
        gt_labels = torch.LongTensor([[0], [1]
                                      ]).unsqueeze(0).repeat(batch_size, 1, 1)
        pad_bbox_flag = torch.FloatTensor([[1], [0]]).unsqueeze(0).repeat(
            batch_size, 1, 1)
        pred_bboxes = torch.FloatTensor([
            [-4., -4., 12., 12.],
            [4., -4., 20., 12.],
            [12., -4., 28., 12.],
            [20., -4., 36., 12.],
        ]).unsqueeze(0).repeat(batch_size, 21, 1)

        assign_result = assigner.forward(pred_bboxes, pred_scores, priors,
                                         gt_labels, gt_bboxes, pad_bbox_flag)

        assigned_labels = assign_result['assigned_labels']
        assigned_bboxes = assign_result['assigned_bboxes']
        assigned_scores = assign_result['assigned_scores']
        fg_mask_pre_prior = assign_result['fg_mask_pre_prior']

        self.assertEqual(assigned_labels.shape, torch.Size([batch_size, 84]))
        self.assertEqual(assigned_bboxes.shape, torch.Size([batch_size, 84,
                                                            4]))
        self.assertEqual(assigned_scores.shape,
                         torch.Size([batch_size, 84, num_classes]))
        self.assertEqual(fg_mask_pre_prior.shape, torch.Size([batch_size, 84]))
