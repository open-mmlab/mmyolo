# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

import torch
from einops import rearrange, repeat
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


@TASK_UTILS.register_module()
class YOLOXKptCoder(BaseBBoxCoder):
    """YOLOX Kpt coder.

    This decoder decodes pred kpts (delta_x, delta_x, w, h) to kpts (tl_x,
    tl_y, br_x, br_y).
    """

    def encode(self, **kwargs):
        """Encode deltas between bboxes and ground truth boxes."""
        pass

    def decode(self, priors: torch.Tensor, pred_kpts: torch.Tensor,
               stride: Union[torch.Tensor, int]) -> torch.Tensor:
        """Decode regression results (delta_x, delta_y) to kpts (tl_x,
        tl_y).

        Args:
            priors (torch.Tensor): Basic boxes or points, e.g. anchor.
            pred_kpts (torch.Tensor): Encoded kpts with shape (N, 8400, K x 2).
            stride (torch.Tensor | int): Strides of kpts.

        Returns:
            torch.Tensor: Decoded kpts.
        """
        if (pred_kpts.shape[-1] % 2) == 0:
            c = 2
        elif (pred_kpts.shape[-1] % 3) == 0:
            c = 3
        else:
            raise NotImplementedError(
                "Keypoints decoder is for xyv, or xy only.")
        stride = stride[None, :, None, None]
        pred_kpts = rearrange(pred_kpts, 'b h (k c) -> b h k c', c=c)
        priors = repeat(priors, 'anchors xy -> anchors keypoints xy', keypoints=pred_kpts.shape[-2])
        xy_coordinates = (pred_kpts[..., :2] * stride) + priors
        pred_kpts[..., :2] = xy_coordinates
        return pred_kpts
