# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch
from mmengine.structures import InstanceData
from torch import Tensor

from mmyolo.models.utils import gt_instances_preprocess
from mmyolo.utils import register_all_modules

register_all_modules()


class TestGtInstancesPreprocess:

    @pytest.mark.parametrize('box_dim', [4, 5])
    def test(self, box_dim):
        gt_instances = InstanceData(
            bboxes=torch.empty((0, box_dim)), labels=torch.LongTensor([]))
        batch_size = 1
        batch_instance = gt_instances_preprocess([gt_instances], batch_size)
        assert isinstance(batch_instance, Tensor)
        assert len(batch_instance.shape) == 3, 'the len of result must be 3.'
        assert batch_instance.size(-1) == box_dim + 1

    @pytest.mark.parametrize('box_dim', [4, 5])
    def test_fast_version(self, box_dim: int):
        gt_instances = torch.from_numpy(
            np.array([[0., 1., *(0., ) * box_dim]], dtype=np.float32))
        batch_size = 1
        batch_instance = gt_instances_preprocess(gt_instances, batch_size)
        assert isinstance(batch_instance, Tensor)
        assert len(batch_instance.shape) == 3, 'the len of result must be 3.'
        assert batch_instance.shape[1] == 1
        assert batch_instance.shape[2] == box_dim + 1
