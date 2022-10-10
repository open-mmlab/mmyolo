# Copyright (c) OpenMMLab. All rights reserved.
# import copy
# import math
# from typing import List, Optional, Sequence, Tuple, Union
from typing import Sequence, Union

# import torch
# import torch.nn as nn
# from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
# from mmdet.models.utils import filter_scores_and_topk, multi_apply
# from mmdet.utils import (ConfigType, OptConfigType, OptInstanceList,
#                          OptMultiConfig)
from mmdet.utils import ConfigType, OptMultiConfig
# from mmengine.config import ConfigDict
# from mmengine.dist import get_dist_info
# from mmengine.logging import print_log
from mmengine.model import BaseModule

# from mmyolo.registry import MODELS, TASK_UTILS
from mmyolo.registry import MODELS
from ..utils import make_divisible

# from mmengine.structures import InstanceData
# from torch import Tensor


@MODELS.register_module()
class PPYOLOEHeadModule(BaseModule):

    def __init__(
            self,
            num_classes: int,
            in_channels: Union[int, Sequence],
            widen_factor: float = 1.0,
            num_base_priors: int = 1,
            featmap_strides: Sequence[int] = (8, 16, 32),
            # norm_cfg: ConfigType = dict(
            #     type='BN', momentum=0.03, eps=0.001),
            act_cfg: ConfigType = dict(type='SiLU', inplace=True),
            init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)

        self.num_classes = num_classes
        self.featmap_strides = featmap_strides
        self.num_base_priors = num_base_priors
        # self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_channels = in_channels

        in_channels = []
        for channel in self.in_channels:
            channel = make_divisible(channel, widen_factor)
            in_channels.append(channel)
        self.in_channels = in_channels

        self._init_layers()

    def init_weights(self):
        raise NotImplementedError

    def _init_layers(self):
        raise NotImplementedError
