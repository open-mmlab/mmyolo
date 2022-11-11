# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

from mmyolo.registry import MODELS


@MODELS.register_module()
class DfLoss(nn.Module):
    """# TODO: DOC."""

    def __init__(self, reduction='mean', loss_weight=1.0, reg_max=16):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.reg_max = reg_max
        assert self.reduction in (None, 'none', 'mean', 'sum')

    def forward(self, preds, targets, weight, avg_factor=None):
        targets_left = targets.long()
        targets_right = targets_left + 1
        weight_left = targets_right.float() - targets
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(
            preds.view(-1, self.reg_max + 1),
            targets_left.view(-1),
            reduction='none').view(targets_left.shape) * weight_left
        loss_right = F.cross_entropy(
            preds.view(-1, self.reg_max + 1),
            targets_right.view(-1),
            reduction='none').view(targets_left.shape) * weight_right
        loss = (loss_left + loss_right) * self.loss_weight

        if self.reduction == 'mean':
            loss = (loss.mean(-1, keepdim=True) * weight).sum()
        elif self.reduction == 'sum':
            loss = (loss.sum(-1, keepdim=True) * weight).sum()

        if avg_factor is not None:
            return loss / avg_factor
        return loss
