# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Optional

from mmengine.hooks import ParamSchedulerHook
from mmengine.runner import Runner

from mmyolo.registry import HOOKS


@HOOKS.register_module()
class PPYOLOEParamSchedulerHook(ParamSchedulerHook):
    """A hook to update learning rate and momentum in optimizer of PPYOLOE. We
    use this hook to implement adaptive computation for `warmup_total_iters`,
    which is not possible with the built-in ParamScheduler in mmyolo.

    Args:
        warmup_min_iter (int): Minimum warmup iters. Defaults to 1000.
        start_factor (float): The number we multiply learning rate in the
            first epoch. The multiplication factor changes towards end_factor
            in the following epochs. Defaults to 0.
        warmup_epochs (int): Epochs for warmup. Defaults to 5.
        min_lr_ratio (float): Minimum learning rate ratio.
        total_epochs (int): In PPYOLOE, `total_epochs` is set to
            training_epochs x 1.2. Defaults to 360.
    """
    priority = 9

    def __init__(self,
                 warmup_min_iter: int = 1000,
                 start_factor: float = 0.,
                 warmup_epochs: int = 5,
                 min_lr_ratio: float = 0.0,
                 total_epochs: int = 360):

        self.warmup_min_iter = warmup_min_iter
        self.start_factor = start_factor
        self.warmup_epochs = warmup_epochs
        self.min_lr_ratio = min_lr_ratio
        self.total_epochs = total_epochs

        self._warmup_end = False
        self._base_lr = None

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

        self._base_lr = [
            group['initial_lr'] for group in optimizer.param_groups
        ]
        self._min_lr = [i * self.min_lr_ratio for i in self._base_lr]

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
        optimizer = runner.optim_wrapper.optimizer
        dataloader_len = len(runner.train_dataloader)

        # The minimum warmup is self.warmup_min_iter
        warmup_total_iters = max(
            round(self.warmup_epochs * dataloader_len), self.warmup_min_iter)

        if cur_iters <= warmup_total_iters:
            # warm up
            alpha = cur_iters / warmup_total_iters
            factor = self.start_factor * (1 - alpha) + alpha

            for group_idx, param in enumerate(optimizer.param_groups):
                param['lr'] = self._base_lr[group_idx] * factor
        else:
            for group_idx, param in enumerate(optimizer.param_groups):
                total_iters = self.total_epochs * dataloader_len
                lr = self._min_lr[group_idx] + (
                    self._base_lr[group_idx] -
                    self._min_lr[group_idx]) * 0.5 * (
                        math.cos((cur_iters - warmup_total_iters) * math.pi /
                                 (total_iters - warmup_total_iters)) + 1.0)
                param['lr'] = lr
