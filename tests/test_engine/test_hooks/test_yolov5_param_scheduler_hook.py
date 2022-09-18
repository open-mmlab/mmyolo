# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import Mock

import torch
from mmengine.config import Config
from mmengine.optim import build_optim_wrapper
from mmengine.runner import Runner
from torch import nn
from torch.utils.data import Dataset

from mmyolo.engine.hooks import YOLOv5ParamSchedulerHook
from mmyolo.utils import register_all_modules


class ToyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, inputs, data_samples, mode='tensor'):
        labels = torch.stack(data_samples)
        inputs = torch.stack(inputs)
        outputs = self.linear(inputs)
        if mode == 'tensor':
            return outputs
        elif mode == 'loss':
            loss = (labels - outputs).sum()
            outputs = dict(loss=loss)
            return outputs
        else:
            return outputs


class DummyDataset(Dataset):
    METAINFO = dict()  # type: ignore
    data = torch.randn(12, 2)
    label = torch.ones(12)

    @property
    def metainfo(self):
        return self.METAINFO

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return dict(inputs=self.data[index], data_sample=self.label[index])


optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        nesterov=True,
        batch_size_per_gpu=1),
    constructor='YOLOv5OptimizerConstructor')

register_all_modules()


class TestYOLOv5ParamSchelerHook(TestCase):

    def test(self):
        model = ToyModel()
        train_dataloader = dict(
            dataset=DummyDataset(),
            sampler=dict(type='DefaultSampler', shuffle=True),
            batch_size=3,
            num_workers=0)

        runner = Mock()
        runner.model = model
        runner.optim_wrapper = build_optim_wrapper(model, optim_wrapper)
        runner.cfg.train_dataloader = Config(train_dataloader)
        runner.train_dataloader = Runner.build_dataloader(train_dataloader)

        hook = YOLOv5ParamSchedulerHook(
            scheduler_type='linear', lr_factor=0.01, max_epochs=300)

        # test before train
        runner.epoch = 0
        runner.iter = 0
        hook.before_train(runner)

        for group in runner.optim_wrapper.param_groups:
            self.assertEqual(group['lr'], 0.01)
            self.assertEqual(group['momentum'], 0.937)

        self.assertFalse(hook._warmup_end)

        # test after training 10 steps
        for i in range(10):
            runner.iter += 1
            hook.before_train_iter(runner, 0)

        for group_idx, group in enumerate(runner.optim_wrapper.param_groups):
            if group_idx == 2:
                self.assertEqual(round(group['lr'], 5), 0.0991)
            self.assertEqual(group['momentum'], 0.80137)
            self.assertFalse(hook._warmup_end)

        # test after warm up
        runner.iter = 1000
        hook.before_train_iter(runner, 0)
        self.assertFalse(hook._warmup_end)

        for group in runner.optim_wrapper.param_groups:
            self.assertEqual(group['lr'], 0.01)
            self.assertEqual(group['momentum'], 0.937)

        runner.iter = 1001
        hook.before_train_iter(runner, 0)
        self.assertTrue(hook._warmup_end)

        # test after train_epoch
        hook.after_train_epoch(runner)
        for group in runner.optim_wrapper.param_groups:
            self.assertEqual(group['lr'], 0.01)
            self.assertEqual(group['momentum'], 0.937)
