# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmyolo.models.task_modules.coders import YOLOXBBoxCoder


class TestYOLOv5Coder(TestCase):

    def test_decoder(self):
        coder = YOLOXBBoxCoder()

        priors = torch.Tensor([[10., 10.], [8., 8.], [15., 8.], [2., 5.]])
        pred_bboxes = torch.Tensor([[0.0000, 0.0000, 1.0000, 1.0000],
                                    [0.0409, 0.1409, 0.8591, 0.8591],
                                    [0.0000, 0.3161, 0.1945, 0.6839],
                                    [1.0000, 5.0000, 0.2000, 0.6000]])
        strides = torch.Tensor([2, 4, 6, 6])
        expected_decode_bboxes = torch.Tensor(
            [[7.2817, 7.2817, 12.7183, 12.7183],
             [3.4415, 3.8415, 12.8857, 13.2857],
             [11.3559, 3.9518, 18.6441, 15.8414],
             [4.3358, 29.5336, 11.6642, 40.4664]])
        out = coder.decode(priors, pred_bboxes, strides)
        assert expected_decode_bboxes.allclose(out, atol=1e-04)

        batch_priors = priors.unsqueeze(0).repeat(2, 1, 1)
        batch_pred_bboxes = pred_bboxes.unsqueeze(0).repeat(2, 1, 1)
        batch_out = coder.decode(batch_priors, batch_pred_bboxes, strides)[0]
        assert out.allclose(batch_out)
