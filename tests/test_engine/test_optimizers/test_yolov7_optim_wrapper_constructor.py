# Copyright (c) OpenMMLab. All rights reserved.

import copy
from unittest import TestCase

import torch
import torch.nn as nn
from mmengine.optim import build_optim_wrapper

from mmyolo.engine import YOLOv7OptimWrapperConstructor
from mmyolo.utils import register_all_modules

register_all_modules()


class ExampleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.param1 = nn.Parameter(torch.ones(1))
        self.conv1 = nn.Conv2d(3, 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(4, 2, kernel_size=1)
        self.bn = nn.BatchNorm2d(2)


class TestYOLOv7OptimWrapperConstructor(TestCase):

    def setUp(self):
        self.model = ExampleModel()
        self.base_lr = 0.01
        self.weight_decay = 0.0001
        self.optim_wrapper_cfg = dict(
            type='OptimWrapper',
            optimizer=dict(
                type='SGD',
                lr=self.base_lr,
                momentum=0.9,
                weight_decay=self.weight_decay,
                batch_size_per_gpu=16))

    def test_init(self):
        YOLOv7OptimWrapperConstructor(copy.deepcopy(self.optim_wrapper_cfg))
        YOLOv7OptimWrapperConstructor(
            copy.deepcopy(self.optim_wrapper_cfg),
            paramwise_cfg={'base_total_batch_size': 64})

        # `paramwise_cfg` must include `base_total_batch_size` if not None.
        with self.assertRaises(AssertionError):
            YOLOv7OptimWrapperConstructor(
                copy.deepcopy(self.optim_wrapper_cfg), paramwise_cfg={'a': 64})

    def test_build(self):
        optim_wrapper = YOLOv7OptimWrapperConstructor(
            copy.deepcopy(self.optim_wrapper_cfg))(
                self.model)
        # test param_groups
        assert len(optim_wrapper.optimizer.param_groups) == 3
        for i in range(3):
            param_groups_i = optim_wrapper.optimizer.param_groups[i]
            assert param_groups_i['lr'] == self.base_lr
            if i == 0:
                assert param_groups_i['weight_decay'] == self.weight_decay
            else:
                assert param_groups_i['weight_decay'] == 0

        # test weight_decay linear scaling
        optim_wrapper_cfg = copy.deepcopy(self.optim_wrapper_cfg)
        optim_wrapper_cfg['optimizer']['batch_size_per_gpu'] = 128
        optim_wrapper = YOLOv7OptimWrapperConstructor(optim_wrapper_cfg)(
            self.model)
        assert optim_wrapper.optimizer.param_groups[0][
            'weight_decay'] == self.weight_decay * 2

        # test without batch_size_per_gpu
        optim_wrapper_cfg = copy.deepcopy(self.optim_wrapper_cfg)
        optim_wrapper_cfg['optimizer'].pop('batch_size_per_gpu')
        optim_wrapper = dict(
            optim_wrapper_cfg, constructor='YOLOv7OptimWrapperConstructor')
        optim_wrapper = build_optim_wrapper(self.model, optim_wrapper)
        assert optim_wrapper.optimizer.param_groups[0][
            'weight_decay'] == self.weight_decay
