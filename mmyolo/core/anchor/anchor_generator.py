# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmdet.models.task_modules import YOLOAnchorGenerator
from torch.nn.modules.utils import _pair

from mmyolo.registry import TASK_UTILS


@TASK_UTILS.register_module()
class YOLOAutoAnchorGenerator(nn.Module, YOLOAnchorGenerator):
    """AutoAnchor generator for YOLO.

    Args:
        strides (list[int] | list[tuple[int, int]]): Strides of anchors
            in multiple feature levels.
        base_sizes (list[list[tuple[int, int]]]): The basic sizes
            of anchors in multiple levels.
    """

    def __init__(self, strides, base_sizes, use_box_type: bool = False):
        super().__init__()
        self.strides = [_pair(stride) for stride in strides]
        self.centers = [(stride[0] / 2., stride[1] / 2.)
                        for stride in self.strides]
        self.use_box_type = use_box_type
        self.register_buffer('anchors', torch.tensor(base_sizes))

    @property
    def base_sizes(self):
        T = []
        num_anchor_per_level = len(self.anchors[0])
        for base_sizes_per_level in self.anchors:
            assert num_anchor_per_level == len(base_sizes_per_level)
            T.append([_pair(base_size) for base_size in base_sizes_per_level])
        return T

    @property
    def base_anchors(self):
        return self.gen_base_anchors()
