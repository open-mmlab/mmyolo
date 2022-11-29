# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.losses.utils import weight_reduce_loss

from mmyolo.registry import MODELS


def qfocal_loss(pred,
                target,
                weight=None,
                beta=2.0,
                reduction='mean',
                avg_factor=None):
    """`QualityFocal Loss <https://arxiv.org/abs/2008.13367>`_

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning target of the iou-aware
            classification score with shape (N, C), C is the number of classes.
        weight (torch.Tensor, optional): The weight of loss for each
            prediction. Defaults to None.
        beta (float, optional): A balance factor for the negative part of
            Varifocal Loss, which is different from the alpha of Focal Loss.
            Defaults to 0.75.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    # pred and target should be of the same size
    assert pred.size() == target.size()
    pred_sigmoid = pred.sigmoid()
    scale_factor = pred_sigmoid
    target = target.type_as(pred)

    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = F.binary_cross_entropy_with_logits(
        pred, zerolabel, reduction='none') * scale_factor.pow(beta)

    pos = (target != 0)
    scale_factor = target[pos] - pred_sigmoid[pos]
    loss[pos] = F.binary_cross_entropy_with_logits(
        pred[pos], target[pos],
        reduction='none') * scale_factor.abs().pow(beta)

    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@MODELS.register_module()
class QualityFocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 beta=2.0,
                 reduction='mean',
                 loss_weight=1.0):
        """Quality Focal Loss (QFL) is a variant of `Generalized Focal Loss:

        Learning Qualified and Distributed Bounding Boxes for Dense Object
        Detection <https://arxiv.org/abs/2006.04388>`_.
        Args:
            use_sigmoid (bool): Whether sigmoid operation is conducted in QFL.
                Defaults to True.
            beta (float): The beta parameter for calculating the modulating
                factor.Defaults to 2.0.
            reduction (str): Options are "none", "mean" and "sum".
            loss_weight (float): Loss weight of current loss.
        """
        super().__init__()
        assert use_sigmoid is True, \
            'Only sigmoid qfocal loss supported now.'
        assert beta >= 0.0
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss_cls = self.loss_weight * qfocal_loss(
                pred,
                target,
                weight,
                beta=self.beta,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls
