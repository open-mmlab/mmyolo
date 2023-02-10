import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from mmpose.evaluation.functional import oks_nms, soft_oks_nms
from mmyolo.registry import MODELS


@MODELS.register_module()
class OksLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, preds, targets):
        pass