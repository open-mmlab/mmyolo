# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

import torch
from mmdet.models.task_modules.coders.base_bbox_coder import BaseBBoxCoder

from mmyolo.registry import TASK_UTILS


@TASK_UTILS.register_module()
class YOLOv5BBoxCoder(BaseBBoxCoder):
    """YOLOv5 BBox coder.

    This decoder decodes pred bboxes (delta_x, delta_x, w, h) to bboxes (tl_x,
    tl_y, br_x, br_y).
    """

    def encode(self, **kwargs):
        """Encode deltas between bboxes and ground truth boxes."""
        pass

    def decode(self, priors: torch.Tensor, pred_bboxes: torch.Tensor,
               stride: Union[torch.Tensor, int]) -> torch.Tensor:
        """Decode regression results (delta_x, delta_x, w, h) to bboxes (tl_x,
        tl_y, br_x, br_y).

        Args:
            priors (torch.Tensor): Basic boxes or points, e.g. anchors.
            pred_bboxes (torch.Tensor): Encoded boxes with shape
            stride (torch.Tensor | int): Strides of bboxes.

        Returns:
            torch.Tensor: Decoded boxes.
        """
        assert pred_bboxes.size(-1) == priors.size(-1) == 4

        pred_bboxes = pred_bboxes.sigmoid()

        x_center = (priors[..., 0] + priors[..., 2]) * 0.5
        y_center = (priors[..., 1] + priors[..., 3]) * 0.5
        w = priors[..., 2] - priors[..., 0]
        h = priors[..., 3] - priors[..., 1]

        # The anchor of mmdet has been offset by 0.5
        x_center_pred = (pred_bboxes[..., 0] - 0.5) * 2 * stride + x_center
        y_center_pred = (pred_bboxes[..., 1] - 0.5) * 2 * stride + y_center
        w_pred = (pred_bboxes[..., 2] * 2)**2 * w
        h_pred = (pred_bboxes[..., 3] * 2)**2 * h

        decoded_bboxes = torch.stack(
            (x_center_pred - w_pred / 2, y_center_pred - h_pred / 2,
             x_center_pred + w_pred / 2, y_center_pred + h_pred / 2),
            dim=-1)

        return decoded_bboxes
