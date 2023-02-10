# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import torch
from mmengine.structures import InstanceData
from torch import Tensor

from mmyolo.models.utils import gt_instances_preprocess
from mmyolo.utils import register_all_modules

register_all_modules()


class TestGtInstancesPreprocess(TestCase):

    def test(self):
        gt_instances = InstanceData(
            bboxes=torch.empty((0, 4)), labels=torch.LongTensor([]))
        batch_size = 1
        batch_instance = gt_instances_preprocess([gt_instances], batch_size)
        self.assertIsInstance(batch_instance, Tensor)
        self.assertEqual(
            len(batch_instance.shape), 3, 'the len of result must be 3.')

    def test_fast_version(self):
        gt_instances = torch.from_numpy(
            np.array([[0., 1., 0., 0., 0., 0.]], dtype=np.float32))
        batch_size = 1
        batch_instance = gt_instances_preprocess(gt_instances, batch_size)
        self.assertIsInstance(batch_instance, Tensor)
        self.assertEqual(
            len(batch_instance.shape), 3, 'the len of result must be 3.')
        self.assertEqual(batch_instance.shape[1], 1)
        self.assertEqual(batch_instance.shape[2], 5)
