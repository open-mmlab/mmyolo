# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Sequence

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.utils import ConfigType, OptMultiConfig
from mmyolo.registry import MODELS


@MODELS.register_module()
class BaseBackbone(BaseModule, metaclass=ABCMeta):
    """BaseBackbone backbone used in YOLO series.

    Args:
        arch_setting (dict): Architecture of BaseBackbone.
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        input_channels: Number of input image channels. Defaults to 3.
        out_indices (Sequence[int]): Output from which stages.
            Defaults to (2, 3, 4).
        frozen_stages (int): Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters. Defaults to -1.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Defaults to None.
        act_cfg (dict): Config dict for activation layer.
            Defaults to None.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Defaults to False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 arch_setting: dict,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 input_channels: int = 3,
                 out_indices: Sequence[int] = (2, 3, 4),
                 frozen_stages: int = -1,
                 norm_cfg: ConfigType = None,
                 act_cfg: ConfigType = None,
                 norm_eval: bool = False,
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)

        self.num_stages = len(arch_setting)
        self.arch_setting = arch_setting

        assert set(out_indices).issubset(
            i for i in range(len(arch_setting) + 1))

        if frozen_stages not in range(-1, len(arch_setting) + 1):
            raise ValueError('"frozen_stages" must be in range(-1, '
                             'len(arch_setting) + 1). But received '
                             f'{frozen_stages}')

        self.input_channels = input_channels
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.widen_factor = widen_factor
        self.deepen_factor = deepen_factor
        self.norm_eval = norm_eval
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.stem = self.build_stem_layer()
        self.layers = ['stem']

        for idx, setting in enumerate(arch_setting):
            stage = []
            stage += self.build_stage_layer(idx, setting)
            self.add_module(f'stage{idx + 1}', nn.Sequential(*stage))
            self.layers.append(f'stage{idx + 1}')

    @abstractmethod
    def build_stem_layer(self):
        """Build a stem layer."""
        pass

    @abstractmethod
    def build_stage_layer(self, stage_idx: int, setting: list):
        """Build a stage layer.

        Args:
            stage_idx (int): The index of a stage layer.
            setting (list): The architecture setting of a stage layer.
        """
        pass

    def _freeze_stages(self):
        """Freeze the parameters of the specified stage so that they are no
        longer updated."""
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self, self.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True):
        """Convert the model into training mode while keep normalization layer
        frozen."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x: torch.Tensor) -> tuple:
        """Forward batch_inputs from the data_preprocessor."""
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)
