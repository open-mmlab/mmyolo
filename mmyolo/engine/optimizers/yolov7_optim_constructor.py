# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch.nn as nn
from mmengine.dist import get_world_size
from mmengine.logging import print_log
from mmengine.model import is_model_wrapper
from mmengine.optim import OptimWrapper

from mmyolo.registry import (OPTIM_WRAPPER_CONSTRUCTORS, OPTIM_WRAPPERS,
                             OPTIMIZERS)
from mmyolo.models.dense_heads.yolov7_head import ImplicitA, ImplicitM

@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class YOLOv7OptimizerConstructor:

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

        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in model.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.modules.batchnorm._NormBase):
                pg0.append(v.weight)  # no decay
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay
            # Caution: Strong coupling with model naming
            if isinstance(v, (ImplicitA, ImplicitM)):
                pg0.append(v.implicit)

        # Note: Make sure bias is in the last parameter group
        optimizer_cfg['params'] = []
        # conv
        optimizer_cfg['params'].append({
            'params': pg1,
            'weight_decay': weight_decay
        })
        # bn
        optimizer_cfg['params'].append({'params': pg0})
        # bias
        optimizer_cfg['params'].append({'params': pg2})

        print_log(
            'Optimizer groups: %g .bias, %g conv.weight, %g other' %
            (len(pg2), len(pg1), len(pg0)), 'current')
        del pg0, pg1, pg2

        optimizer = OPTIMIZERS.build(optimizer_cfg)
        optim_wrapper = OPTIM_WRAPPERS.build(
            self.optim_wrapper_cfg, default_args=dict(optimizer=optimizer))
        return optim_wrapper
