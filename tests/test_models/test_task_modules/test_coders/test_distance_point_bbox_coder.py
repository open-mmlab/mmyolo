# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmyolo.models.task_modules.coders import DistancePointBBoxCoder


class TestDistancePointBBoxCoder(TestCase):

    def test_decoder(self):
        coder = DistancePointBBoxCoder()

        points = torch.Tensor([[74., 61.], [-29., 106.], [138., 61.],
                               [29., 170.]])
        pred_bboxes = torch.Tensor([[0, -1, 3, 3], [-1, -7, -4.8, 9],
                                    [-23, -1, 12, 1], [14.5, -13, 10, 18.3]])
        expected_distance = torch.Tensor([[74, 63, 80, 67],
                                          [-25, 134, -48.2, 142],
                                          [276, 67, 210, 67],
                                          [-58, 248, 89, 279.8]])
        strides = torch.Tensor([2, 4, 6, 6])
        out_distance = coder.decode(points, pred_bboxes, strides)
        assert expected_distance.allclose(out_distance)

        batch_priors = points.unsqueeze(0).repeat(2, 1, 1)
        batch_pred_bboxes = pred_bboxes.unsqueeze(0).repeat(2, 1, 1)
        batch_out = coder.decode(batch_priors, batch_pred_bboxes, strides)[0]
        assert out_distance.allclose(batch_out)
