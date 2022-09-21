# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Optional

import torch
import torch.nn as nn
from mmdet.models.layers import ExpMomentumEMA as MMDET_ExpMomentumEMA
from torch import Tensor

from mmyolo.registry import MODELS


@MODELS.register_module()
class ExpMomentumEMA(MMDET_ExpMomentumEMA):
    """Exponential moving average (EMA) with exponential momentum strategy,
    which is used in YOLO.

    Args:
        model (nn.Module): The model to be averaged.
        momentum (float): The momentum used for updating ema parameter.
            Ema's parameters are updated with the formula:
           `averaged_param = (1-momentum) * averaged_param + momentum *
           source_param`. Defaults to 0.0002.
        gamma (int): Use a larger momentum early in training and gradually
            annealing to a smaller value to update the ema model smoothly. The
            momentum is calculated as
            `(1 - momentum) * exp(-(1 + steps) / gamma) + momentum`.
            Defaults to 2000.
        interval (int): Interval between two updates. Defaults to 1.
        device (torch.device, optional): If provided, the averaged model will
            be stored on the :attr:`device`. Defaults to None.
        update_buffers (bool): if True, it will compute running averages for
            both the parameters and the buffers of the model. Defaults to
            False.
    """

    def __init__(self,
                 model: nn.Module,
                 momentum: float = 0.0002,
                 gamma: int = 2000,
                 interval=1,
                 device: Optional[torch.device] = None,
                 update_buffers: bool = False):
        super().__init__(
            model=model,
            momentum=momentum,
            interval=interval,
            device=device,
            update_buffers=update_buffers)
        assert gamma > 0, f'gamma must be greater than 0, but got {gamma}'
        self.gamma = gamma

        # Note: There is no need to re-fetch every update,
        # as most models do not change their structure
        # during the training process.
        self.src_parameters = (
            model.state_dict()
            if self.update_buffers else dict(model.named_parameters()))
        if not self.update_buffers:
            self.src_buffers = model.buffers()

    def avg_func(self, averaged_param: Tensor, source_param: Tensor,
                 steps: int):
        """Compute the moving average of the parameters using the exponential
        momentum strategy.

        Args:
            averaged_param (Tensor): The averaged parameters.
            source_param (Tensor): The source parameters.
            steps (int): The number of times the parameters have been
                updated.
        """
        momentum = (1 - self.momentum) * math.exp(
            -float(1 + steps) / self.gamma) + self.momentum
        averaged_param.lerp_(source_param, momentum)

    def update_parameters(self, model: nn.Module):
        """Update the parameters after each training step.

        Args:
            model (nn.Module): The model of the parameter needs to be updated.
        """
        if self.steps == 0:
            for k, p_avg in self.avg_parameters.items():
                p_avg.data.copy_(self.src_parameters[k].data)
        elif self.steps % self.interval == 0:
            for k, p_avg in self.avg_parameters.items():
                if p_avg.dtype.is_floating_point:
                    self.avg_func(p_avg.data, self.src_parameters[k].data,
                                  self.steps)
        if not self.update_buffers:
            # If not update the buffers,
            # keep the buffers in sync with the source model.
            for b_avg, b_src in zip(self.module.buffers(), self.src_buffers):
                b_avg.data.copy_(b_src.data)
        self.steps += 1
