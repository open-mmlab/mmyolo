# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmyolo.models.task_modules.coders import YOLOv5BBoxCoder


class TestYOLOv5Coder(TestCase):

    def test_decoder(self):
        coder = YOLOv5BBoxCoder()

        priors = torch.Tensor([[10., 10., 20., 20.], [10., 8., 10., 10.],
                               [15., 8., 20., 3.], [2., 5., 5., 8.]])
        pred_bboxes = torch.Tensor([[0.0000, 0.0000, 1.0000, 1.0000],
                                    [0.1409, 0.1409, 2.8591, 2.8591],
                                    [0.0000, 0.3161, 4.1945, 0.6839],
                                    [1.0000, 5.0000, 9.0000, 5.0000]])
        strides = torch.Tensor([2, 4, 8, 8])
        expected_decode_bboxes = torch.Tensor(
            [[4.3111, 4.3111, 25.6889, 25.6889],
             [10.2813, 5.7033, 10.2813, 12.8594],
             [7.7949, 11.1710, 27.2051, 2.3369],
             [1.1984, 8.4730, 13.1955, 20.3129]])
        out = coder.decode(priors, pred_bboxes, strides)
        assert expected_decode_bboxes.allclose(out, atol=1e-04)

        batch_priors = priors.unsqueeze(0).repeat(2, 1, 1)
        batch_pred_bboxes = pred_bboxes.unsqueeze(0).repeat(2, 1, 1)
        batch_out = coder.decode(batch_priors, batch_pred_bboxes, strides)[0]
        assert out.allclose(batch_out)
