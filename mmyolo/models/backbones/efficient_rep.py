# Copyright (c) OpenMMLab. All rights reserved.

from typing import List, Tuple, Union

import torch
import torch.nn as nn
from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.models.layers.yolo_bricks import SPPFBottleneck
from mmyolo.registry import MODELS
from ..layers import RepStageBlock, RepVGGBlock
from ..utils import make_divisible, make_round
from .base_backbone import BaseBackbone


@MODELS.register_module()
class YOLOv6EfficientRep(BaseBackbone):
    """EfficientRep backbone used in YOLOv6.

    Args:
        arch (str): Architecture of BaseDarknet, from {P5, P6}.
            Defaults to P5.
        plugins (list[dict]): List of plugins for stages, each dict contains:

            - cfg (dict, required): Cfg dict to build plugin.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        input_channels (int): Number of input image channels. Defaults to 3.
        out_indices (Tuple[int]): Output from which stages.
            Defaults to (2, 3, 4).
        frozen_stages (int): Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters. Defaults to -1.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Defaults to dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='LeakyReLU', negative_slope=0.1).
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Defaults to False.
        block (nn.Module): block used to build each stage.
        init_cfg (Union[dict, list[dict]], optional): Initialization config
            dict. Defaults to None.

    Example:
        >>> from mmyolo.models import YOLOv6EfficientRep
        >>> import torch
        >>> model = YOLOv6EfficientRep()
        >>> model.eval()
        >>> inputs = torch.rand(1, 3, 416, 416)
        >>> level_outputs = model(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        ...
        (1, 256, 52, 52)
        (1, 512, 26, 26)
        (1, 1024, 13, 13)
    """
    # From left to right:
    # in_channels, out_channels, num_blocks, use_spp
    arch_settings = {
        'P5': [[64, 128, 6, False], [128, 256, 12, False],
               [256, 512, 18, False], [512, 1024, 6, True]]
    }

    def __init__(self,
                 arch: str = 'P5',
                 plugins: Union[dict, List[dict]] = None,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 input_channels: int = 3,
                 out_indices: Tuple[int] = (2, 3, 4),
                 frozen_stages: int = -1,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='ReLU', inplace=True),
                 norm_eval: bool = False,
                 block: nn.Module = RepVGGBlock,
                 init_cfg: OptMultiConfig = None):
        self.block = block
        super().__init__(
            self.arch_settings[arch],
            deepen_factor,
            widen_factor,
            input_channels=input_channels,
            out_indices=out_indices,
            plugins=plugins,
            frozen_stages=frozen_stages,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            norm_eval=norm_eval,
            init_cfg=init_cfg)

    def build_stem_layer(self) -> nn.Module:
        """Build a stem layer."""
        return self.block(
            in_channels=self.input_channels,
            out_channels=make_divisible(self.arch_setting[0][0],
                                        self.widen_factor),
            kernel_size=3,
            stride=2)

    def build_stage_layer(self, stage_idx: int, setting: list) -> list:
        """Build a stage layer.

        Args:
            stage_idx (int): The index of a stage layer.
            setting (list): The architecture setting of a stage layer.
        """
        in_channels, out_channels, num_blocks, use_spp = setting

        in_channels = make_divisible(in_channels, self.widen_factor)
        out_channels = make_divisible(out_channels, self.widen_factor)
        num_blocks = make_round(num_blocks, self.deepen_factor)

        stage = []

        ef_block = nn.Sequential(
            self.block(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2),
            RepStageBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                n=num_blocks,
                block=self.block,
            ))
        stage.append(ef_block)

        if use_spp:
            spp = SPPFBottleneck(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_sizes=5,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            stage.append(spp)
        return stage

    def init_weights(self):
        """Initialize the parameters."""
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                # In order to be consistent with the source code,
                # reset the Conv2d initialization parameters
                m.reset_parameters()
