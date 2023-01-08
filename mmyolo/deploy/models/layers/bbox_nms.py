# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdeploy.core import mark
from torch import Tensor


def _efficient_nms(
    boxes: Tensor,
    scores: Tensor,
    max_output_boxes_per_class: int = 1000,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
    pre_top_k: int = -1,
    keep_top_k: int = 100,
    box_coding: int = 0,
):
    """Wrapper for `efficient_nms` with TensorRT.

    Args:
        boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4].
        scores (Tensor): The detection scores of shape
            [N, num_boxes, num_classes].
        max_output_boxes_per_class (int): Maximum number of output
            boxes per class of nms. Defaults to 1000.
        iou_threshold (float): IOU threshold of nms. Defaults to 0.5.
        score_threshold (float): score threshold of nms.
            Defaults to 0.05.
        pre_top_k (int): Number of top K boxes to keep before nms.
            Defaults to -1.
        keep_top_k (int): Number of top K boxes to keep after nms.
            Defaults to -1.
        box_coding (int): Bounding boxes format for nms.
            Defaults to 0 means [x, y, w, h].
            Set to 1 means [x1, y1 ,x2, y2].

    Returns:
        tuple[Tensor, Tensor]: (dets, labels), `dets` of shape [N, num_det, 5]
            and `labels` of shape [N, num_det].
    """
    boxes = boxes if boxes.dim() == 4 else boxes.unsqueeze(2)
    _, det_boxes, det_scores, labels = TRTEfficientNMSop.apply(
        boxes, scores, -1, box_coding, iou_threshold, keep_top_k, '1', 0,
        score_threshold)
    dets = torch.cat([det_boxes, det_scores.unsqueeze(2)], -1)

    # retain shape info
    batch_size = boxes.size(0)

    dets_shape = dets.shape
    label_shape = labels.shape
    dets = dets.reshape([batch_size, *dets_shape[1:]])
    labels = labels.reshape([batch_size, *label_shape[1:]])
    return dets, labels


@mark('efficient_nms', inputs=['boxes', 'scores'], outputs=['dets', 'labels'])
def efficient_nms(*args, **kwargs):
    """Wrapper function for `_efficient_nms`."""
    return _efficient_nms(*args, **kwargs)


class TRTEfficientNMSop(torch.autograd.Function):
    """Efficient NMS op for TensorRT."""

    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        background_class=-1,
        box_coding=0,
        iou_threshold=0.45,
        max_output_boxes=100,
        plugin_version='1',
        score_activation=0,
        score_threshold=0.25,
    ):
        """Forward function of TRTEfficientNMSop."""
        batch_size, num_boxes, num_classes = scores.shape
        num_det = torch.randint(
            0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 4)
        det_scores = torch.randn(batch_size, max_output_boxes)
        det_classes = torch.randint(
            0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32)
        return num_det, det_boxes, det_scores, det_classes

    @staticmethod
    def symbolic(g,
                 boxes,
                 scores,
                 background_class=-1,
                 box_coding=0,
                 iou_threshold=0.45,
                 max_output_boxes=100,
                 plugin_version='1',
                 score_activation=0,
                 score_threshold=0.25):
        """Symbolic function of TRTEfficientNMSop."""
        out = g.op(
            'TRT::EfficientNMS_TRT',
            boxes,
            scores,
            background_class_i=background_class,
            box_coding_i=box_coding,
            iou_threshold_f=iou_threshold,
            max_output_boxes_i=max_output_boxes,
            plugin_version_s=plugin_version,
            score_activation_i=score_activation,
            score_threshold_f=score_threshold,
            outputs=4)
        nums, boxes, scores, classes = out
        return nums, boxes, scores, classes
