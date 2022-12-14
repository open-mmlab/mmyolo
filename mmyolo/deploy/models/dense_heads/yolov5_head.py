# Copyright (c) OpenMMLab. All rights reserved.
import copy
from functools import partial
from typing import List, Optional, Tuple

import torch
from mmdeploy.codebase.mmdet import get_post_processing_params
from mmdeploy.codebase.mmdet.models.layers import multiclass_nms
from mmdeploy.core import FUNCTION_REWRITER
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmyolo.deploy.models.layers import efficient_nms
from mmyolo.models.dense_heads import YOLOv5Head


def yolov5_bbox_decoder(priors: Tensor, bbox_preds: Tensor,
                        stride: int) -> Tensor:
    """Decode YOLOv5 bounding boxes.

    Args:
        priors (Tensor): Prior boxes in center-offset form.
        bbox_preds (Tensor): Predicted bounding boxes.
        stride (int): Stride of the feature map.

    Returns:
        Tensor: Decoded bounding boxes.
    """
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


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmyolo.models.dense_heads.yolov5_head.'
    'YOLOv5Head.predict_by_feat')
def yolov5_head__predict_by_feat(self,
                                 cls_scores: List[Tensor],
                                 bbox_preds: List[Tensor],
                                 objectnesses: Optional[List[Tensor]] = None,
                                 batch_img_metas: Optional[List[dict]] = None,
                                 cfg: Optional[ConfigDict] = None,
                                 rescale: bool = False,
                                 with_nms: bool = True) -> Tuple[InstanceData]:
    """Transform a batch of output features extracted by the head into
    bbox results.
    Args:
        cls_scores (list[Tensor]): Classification scores for all
            scale levels, each is a 4D-tensor, has shape
            (batch_size, num_priors * num_classes, H, W).
        bbox_preds (list[Tensor]): Box energies / deltas for all
            scale levels, each is a 4D-tensor, has shape
            (batch_size, num_priors * 4, H, W).
        objectnesses (list[Tensor], Optional): Score factor for
            all scale level, each is a 4D-tensor, has shape
            (batch_size, 1, H, W).
        batch_img_metas (list[dict], Optional): Batch image meta info.
            Defaults to None.
        cfg (ConfigDict, optional): Test / postprocessing
            configuration, if None, test_cfg would be used.
            Defaults to None.
        rescale (bool): If True, return boxes in original image space.
            Defaults to False.
        with_nms (bool): If True, do nms before return boxes.
            Defaults to True.
    Returns:
        tuple[Tensor, Tensor]: The first item is an (N, num_box, 5) tensor,
            where 5 represent (tl_x, tl_y, br_x, br_y, score), N is batch
            size and the score between 0 and 1. The shape of the second
            tensor in the tuple is (N, num_box), and each element
            represents the class label of the corresponding box.
    """
    ctx = FUNCTION_REWRITER.get_context()
    detector_type = type(self)
    deploy_cfg = ctx.cfg
    use_efficientnms = deploy_cfg.get('use_efficientnms', False)
    dtype = cls_scores[0].dtype
    device = cls_scores[0].device
    bbox_decoder = self.bbox_coder.decode
    nms_func = multiclass_nms
    if use_efficientnms:
        if detector_type is YOLOv5Head:
            nms_func = partial(efficient_nms, box_coding=0)
            bbox_decoder = yolov5_bbox_decoder
        else:
            nms_func = efficient_nms

    assert len(cls_scores) == len(bbox_preds)
    cfg = self.test_cfg if cfg is None else cfg
    cfg = copy.deepcopy(cfg)

    num_imgs = cls_scores[0].shape[0]
    featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

    mlvl_priors = self.prior_generator.grid_priors(
        featmap_sizes, dtype=dtype, device=device)

    flatten_priors = torch.cat(mlvl_priors)

    mlvl_strides = [
        flatten_priors.new_full(
            (featmap_size[0] * featmap_size[1] * self.num_base_priors, ),
            stride)
        for featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
    ]
    flatten_stride = torch.cat(mlvl_strides)

    # flatten cls_scores, bbox_preds and objectness
    flatten_cls_scores = [
        cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_classes)
        for cls_score in cls_scores
    ]
    cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()

    flatten_bbox_preds = [
        bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
        for bbox_pred in bbox_preds
    ]
    flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)

    if objectnesses is not None:
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]
        flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        cls_scores = cls_scores * (flatten_objectness.unsqueeze(-1))

    scores = cls_scores

    bboxes = bbox_decoder(flatten_priors[None], flatten_bbox_preds,
                          flatten_stride)

    if not with_nms:
        return bboxes, scores

    post_params = get_post_processing_params(deploy_cfg)
    max_output_boxes_per_class = post_params.max_output_boxes_per_class
    iou_threshold = cfg.nms.get('iou_threshold', post_params.iou_threshold)
    score_threshold = cfg.get('score_thr', post_params.score_threshold)
    pre_top_k = post_params.pre_top_k
    keep_top_k = cfg.get('max_per_img', post_params.keep_top_k)

    return nms_func(bboxes, scores, max_output_boxes_per_class, iou_threshold,
                    score_threshold, pre_top_k, keep_top_k)


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmyolo.models.dense_heads.yolov5_head.'
    'YOLOv5Head.predict',
    backend='rknn')
def yolov5_head__predict__rknn(self, x: Tuple[Tensor], *args,
                               **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
    """Perform forward propagation of the detection head and predict detection
    results on the features of the upstream network.

    Args:
        x (tuple[Tensor]): Multi-level features from the
            upstream network, each is a 4D-tensor.
    """
    outs = self(x)
    return outs


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmyolo.models.dense_heads.yolov5_head.'
    'YOLOv5HeadModule.forward',
    backend='rknn')
def yolov5_head_module__forward__rknn(
        self, x: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
    """Forward feature of a single scale level."""
    out = []
    for i, feat in enumerate(x):
        out.append(self.convs_pred[i](feat))
    return out
