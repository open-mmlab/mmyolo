# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Optional

import numpy as np
from mmengine.hooks import ParamSchedulerHook
from mmengine.runner import Runner

from mmyolo.registry import HOOKS


def linear_fn(lr_factor: float, max_epochs: int):
    """Generate linear function."""
    return lambda x: (1 - x / max_epochs) * (1.0 - lr_factor) + lr_factor


def cosine_fn(lr_factor: float, max_epochs: int):
    """Generate cosine function."""
    return lambda x: (
        (1 - math.cos(x * math.pi / max_epochs)) / 2) * (lr_factor - 1) + 1


@HOOKS.register_module()
class YOLOv5ParamSchedulerHook(ParamSchedulerHook):
    """A hook to update learning rate and momentum in optimizer of YOLOv5."""
    priority = 9

    scheduler_maps = {'linear': linear_fn, 'cosine': cosine_fn}

    def __init__(self,
                 scheduler_type: str = 'linear',
                 lr_factor: float = 0.01,
                 max_epochs: int = 300,
                 warmup_epochs: int = 3,
                 warmup_bias_lr: float = 0.1,
                 warmup_momentum: float = 0.8,
                 warmup_mim_iter: int = 1000,
                 **kwargs):

        assert scheduler_type in self.scheduler_maps

        self.warmup_epochs = warmup_epochs
        self.warmup_bias_lr = warmup_bias_lr
        self.warmup_momentum = warmup_momentum
        self.warmup_mim_iter = warmup_mim_iter

        kwargs.update({'lr_factor': lr_factor, 'max_epochs': max_epochs})
        self.scheduler_fn = self.scheduler_maps[scheduler_type](**kwargs)

        self._warmup_end = False
        self._base_lr = None
        self._base_momentum = None

    def before_train(self, runner: Runner):
        """Operations before train.

        Args:
            runner (Runner): The runner of the training process.
        """
        optimizer = runner.optim_wrapper.optimizer
        for group in optimizer.param_groups:
            # If the param is never be scheduled, record the current value
            # as the initial value.
            group.setdefault('initial_lr', group['lr'])
            group.setdefault('initial_momentum', group.get('momentum', -1))

        self._base_lr = [
            group['initial_lr'] for group in optimizer.param_groups
        ]
        self._base_momentum = [
            group['initial_momentum'] for group in optimizer.param_groups
        ]

    def before_train_iter(self,
                          runner: Runner,
                          batch_idx: int,
                          data_batch: Optional[dict] = None):
        """Operations before each training iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
        """
        cur_iters = runner.iter
        cur_epoch = runner.epoch
        optimizer = runner.optim_wrapper.optimizer

        # The minimum warmup is self.warmup_mim_iter
        warmup_total_iters = max(
            round(self.warmup_epochs * len(runner.train_dataloader)),
            self.warmup_mim_iter)

        if cur_iters <= warmup_total_iters:
            xp = [0, warmup_total_iters]
            for group_idx, param in enumerate(optimizer.param_groups):
                if group_idx == 2:
                    # bias learning rate will be handled specially
                    yp = [
                        self.warmup_bias_lr,
                        self._base_lr[group_idx] * self.scheduler_fn(cur_epoch)
                    ]
                else:
                    yp = [
                        0.0,
                        self._base_lr[group_idx] * self.scheduler_fn(cur_epoch)
                    ]
                param['lr'] = np.interp(cur_iters, xp, yp)

                if 'momentum' in param:
                    param['momentum'] = np.interp(
                        cur_iters, xp,
                        [self.warmup_momentum, self._base_momentum[group_idx]])
        else:
            self._warmup_end = True

    def after_train_epoch(self, runner: Runner):
        """Operations after each training epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        if not self._warmup_end:
            return

        cur_epoch = runner.epoch
        optimizer = runner.optim_wrapper.optimizer
        for group_idx, param in enumerate(optimizer.param_groups):
            param['lr'] = self._base_lr[group_idx] * self.scheduler_fn(
                cur_epoch)
