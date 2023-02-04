# Copyright (c) OpenMMLab. All rights reserved.
from .iou_loss import IoULoss, bbox_overlaps
from .rotated_iou_loss import RotatedIoULoss

__all__ = ['IoULoss', 'bbox_overlaps', 'RotatedIoULoss']
