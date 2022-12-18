# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch.nn as nn
from mmengine.dist import get_world_size
from mmengine.logging import print_log
from mmengine.model import is_model_wrapper
from mmengine.optim import OptimWrapper

from mmyolo.registry import (OPTIM_WRAPPER_CONSTRUCTORS, OPTIM_WRAPPERS,
                             OPTIMIZERS)


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class YOLOv5OptimizerConstructor:
    """YOLOv5 constructor for optimizers.

    It has the following functions：

        - divides the optimizer parameters into 3 groups:
        Conv, Bias and BN

        - support `weight_decay` parameter adaption based on
        `batch_size_per_gpu`

    Args:
        optim_wrapper_cfg (dict): The config dict of the optimizer wrapper.
            Positional fields are

                - ``type``: class name of the OptimizerWrapper
                - ``optimizer``: The configuration of optimizer.

            Optional fields are

                - any arguments of the corresponding optimizer wrapper type,
                  e.g., accumulative_counts, clip_grad, etc.

            The positional fields of ``optimizer`` are

                - `type`: class name of the optimizer.

            Optional fields are

                - any arguments of the corresponding optimizer type, e.g.,
                  lr, weight_decay, momentum, etc.

        paramwise_cfg (dict, optional): Parameter-wise options. Must include
            `base_total_batch_size` if not None. If the total input batch
            is smaller than `base_total_batch_size`, the `weight_decay`
            parameter will be kept unchanged, otherwise linear scaling.

    Example:
        >>> model = torch.nn.modules.Conv1d(1, 1, 1)
        >>> optim_wrapper_cfg = dict(
        >>>     dict(type='OptimWrapper', optimizer=dict(type='SGD', lr=0.01,
        >>>         momentum=0.9, weight_decay=0.0001, batch_size_per_gpu=16))
        >>> paramwise_cfg = dict(base_total_batch_size=64)
        >>> optim_wrapper_builder = YOLOv5OptimizerConstructor(
        >>>     optim_wrapper_cfg, paramwise_cfg)
        >>> optim_wrapper = optim_wrapper_builder(model)
    """

    def __init__(self,
                 optim_wrapper_cfg: dict,
                 paramwise_cfg: Optional[dict] = None):
        if paramwise_cfg is None:
            paramwise_cfg = {'base_total_batch_size': 64}
        assert 'base_total_batch_size' in paramwise_cfg

        if not isinstance(optim_wrapper_cfg, dict):
            raise TypeError('optimizer_cfg should be a dict',
                            f'but got {type(optim_wrapper_cfg)}')
        assert 'optimizer' in optim_wrapper_cfg, (
            '`optim_wrapper_cfg` must contain "optimizer" config')

        self.optim_wrapper_cfg = optim_wrapper_cfg
        self.optimizer_cfg = self.optim_wrapper_cfg.pop('optimizer')
        self.base_total_batch_size = paramwise_cfg['base_total_batch_size']

    def __call__(self, model: nn.Module) -> OptimWrapper:
        if is_model_wrapper(model):
            model = model.module
        optimizer_cfg = self.optimizer_cfg.copy()
        weight_decay = optimizer_cfg.pop('weight_decay', 0)

        if 'batch_size_per_gpu' in optimizer_cfg:
            batch_size_per_gpu = optimizer_cfg.pop('batch_size_per_gpu')
            # No scaling if total_batch_size is less than
            # base_total_batch_size, otherwise linear scaling.
            total_batch_size = get_world_size() * batch_size_per_gpu
            accumulate = max(
                round(self.base_total_batch_size / total_batch_size), 1)
            scale_factor = total_batch_size * \
                accumulate / self.base_total_batch_size

            if scale_factor != 1:
                weight_decay *= scale_factor
                print_log(f'Scaled weight_decay to {weight_decay}', 'current')

        params_groups = [], [], []

        for v in model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                params_groups[2].append(v.bias)
            # Includes SyncBatchNorm
            if isinstance(v, nn.modules.batchnorm._NormBase):
                params_groups[1].append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                params_groups[0].append(v.weight)

        # Note: Make sure bias is in the last parameter group
        optimizer_cfg['params'] = []
        # conv
        optimizer_cfg['params'].append({
            'params': params_groups[0],
            'weight_decay': weight_decay
        })
        # bn
        optimizer_cfg['params'].append({'params': params_groups[1]})
        # bias
        optimizer_cfg['params'].append({'params': params_groups[2]})

        print_log(
            'Optimizer groups: %g .bias, %g conv.weight, %g other' %
            (len(params_groups[2]), len(params_groups[0]), len(
                params_groups[1])), 'current')
        del params_groups

        optimizer = OPTIMIZERS.build(optimizer_cfg)
        optim_wrapper = OPTIM_WRAPPERS.build(
            self.optim_wrapper_cfg, default_args=dict(optimizer=optimizer))
        return optim_wrapper
