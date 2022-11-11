# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor


def yolov5_bbox_decoder(priors: Tensor, bbox_preds: Tensor,
                        stride: Tensor) -> Tensor:
    bbox_preds = bbox_preds.sigmoid()

    x_center = (priors[..., 0] + priors[..., 2]) * 0.5
    y_center = (priors[..., 1] + priors[..., 3]) * 0.5
    w = priors[..., 2] - priors[..., 0]
    h = priors[..., 3] - priors[..., 1]

    x_center_pred = (bbox_preds[..., 0] - 0.5) * 2 * stride + x_center
    y_center_pred = (bbox_preds[..., 1] - 0.5) * 2 * stride + y_center
    w_pred = (bbox_preds[..., 2] * 2)**2 * w
    h_pred = (bbox_preds[..., 3] * 2)**2 * h

    decoded_bboxes = torch.stack(
        [x_center_pred, y_center_pred, w_pred, h_pred], dim=-1)

    return decoded_bboxes
