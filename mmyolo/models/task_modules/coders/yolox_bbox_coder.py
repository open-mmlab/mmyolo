# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

import torch
from mmdet.models.task_modules.coders.base_bbox_coder import BaseBBoxCoder

from mmyolo.registry import TASK_UTILS


@TASK_UTILS.register_module()
class YOLOXBBoxCoder(BaseBBoxCoder):
    """YOLOX BBox coder.

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
        stride = stride[None, :, None]
        xys = (pred_bboxes[..., :2] * stride) + priors
        whs = pred_bboxes[..., 2:].exp() * stride

        tl_x = (xys[..., 0] - whs[..., 0] / 2)
        tl_y = (xys[..., 1] - whs[..., 1] / 2)
        br_x = (xys[..., 0] + whs[..., 0] / 2)
        br_y = (xys[..., 1] + whs[..., 1] / 2)

        decoded_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], -1)
        return decoded_bboxes
