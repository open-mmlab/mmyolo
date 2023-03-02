# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Union

import torch

from mmyolo.registry import TASK_UTILS

try:
    from mmrotate.models.task_modules.coders import \
        DistanceAnglePointCoder as MMROTATE_DistanceAnglePointCoder
    MMROTATE_AVAILABLE = True
except ImportError:
    from mmdet.models.task_modules.coders import BaseBBoxCoder
    MMROTATE_DistanceAnglePointCoder = BaseBBoxCoder
    MMROTATE_AVAILABLE = False


@TASK_UTILS.register_module()
class DistanceAnglePointCoder(MMROTATE_DistanceAnglePointCoder):
    """Distance Angle Point BBox coder.

    This coder encodes gt bboxes (x, y, w, h, theta) into (top, bottom, left,
    right, theta) and decode it back to the original.
    """

    def __init__(self, clip_border=True, angle_version='oc'):
        if not MMROTATE_AVAILABLE:
            raise ImportError(
                'Please run "mim install -r requirements/mmrotate.txt" '
                'to install mmrotate first for rotated detection.')

        super().__init__(clip_border=clip_border, angle_version=angle_version)

    def decode(
        self,
        points: torch.Tensor,
        pred_bboxes: torch.Tensor,
        stride: torch.Tensor,
        max_shape: Optional[Union[Sequence[int], torch.Tensor,
                                  Sequence[Sequence[int]]]] = None,
    ) -> torch.Tensor:
        """Decode distance prediction to bounding box.

        Args:
            points (Tensor): Shape (B, N, 2) or (N, 2).
            pred_bboxes (Tensor): Distance from the given point to 4
                boundaries and angle (left, top, right, bottom, angle).
                Shape (B, N, 5) or (N, 5)
            max_shape (Sequence[int] or torch.Tensor or Sequence[
                Sequence[int]],optional): Maximum bounds for boxes, specifies
                (H, W, C) or (H, W). If priors shape is (B, N, 4), then
                the max_shape should be a Sequence[Sequence[int]],
                and the length of max_shape should also be B.
                Default None.
        Returns:
            Tensor: Boxes with shape (N, 5) or (B, N, 5)
        """
        assert points.size(-2) == pred_bboxes.size(-2)
        assert points.size(-1) == 2
        assert pred_bboxes.size(-1) == 5
        if self.clip_border is False:
            max_shape = None

        if pred_bboxes.dim() == 2:
            stride = stride[:, None]
        else:
            stride = stride[None, :, None]
        pred_bboxes[..., :4] = pred_bboxes[..., :4] * stride

        return self.distance2obb(points, pred_bboxes, max_shape,
                                 self.angle_version)

    def encode(self,
               points: torch.Tensor,
               gt_bboxes: torch.Tensor,
               max_dis: float = 16.,
               eps: float = 0.01) -> torch.Tensor:
        """Encode bounding box to distances.

        Args:
            points (Tensor): Shape (N, 2), The format is [x, y].
            gt_bboxes (Tensor): Shape (N, 5), The format is "xywha"
            max_dis (float): Upper bound of the distance. Default None.
            eps (float): a small value to ensure target < max_dis, instead <=.
                Default 0.1.

        Returns:
            Tensor: Box transformation deltas. The shape is (N, 5).
        """

        assert points.size(-2) == gt_bboxes.size(-2)
        assert points.size(-1) == 2
        assert gt_bboxes.size(-1) == 5
        return self.obb2distance(points, gt_bboxes, max_dis, eps)
