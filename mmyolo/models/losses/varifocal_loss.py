# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from torch import Tensor


def varifocal_loss(pred_score: Tensor,
                   gt_score: Tensor,
                   label: Tensor,
                   alpha: float = 0.75,
                   gamma: float = 2.0) -> Tensor:
    weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
    with torch.cuda.amp.autocast(enabled=False):
        loss = (F.binary_cross_entropy(
            pred_score.float(), gt_score.float(), reduction='none') *
                weight).sum()
    return loss
