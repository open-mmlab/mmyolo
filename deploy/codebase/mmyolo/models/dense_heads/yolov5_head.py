# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Optional, Tuple

import torch
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmdeploy.codebase.mmdet import get_post_processing_params
from mmdeploy.codebase.mmdet.models.layers import multiclass_nms
from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmyolo.models.dense_heads.yolov5_head.'
    'YOLOv5Head.predict_by_feat')
def yolov5_head__predict_by_feat(ctx,
                                 self,
                                 cls_scores: List[Tensor],
                                 bbox_preds: List[Tensor],
                                 objectnesses: Optional[List[Tensor]],
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
    assert len(cls_scores) == len(bbox_preds)
    cfg = self.test_cfg if cfg is None else cfg
    cfg = copy.deepcopy(cfg)

    num_imgs = len(batch_img_metas)
    featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

    mlvl_priors = self.prior_generator.grid_priors(
        featmap_sizes, dtype=cls_scores[0].dtype, device=cls_scores[0].device)
    flatten_priors = torch.cat(mlvl_priors)

    mlvl_strides = [
        flatten_priors.new_full(
            (featmap_size.numel() * self.num_base_priors, ), stride)
        for featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
    ]
    flatten_stride = torch.cat(mlvl_strides)

    # flatten cls_scores, bbox_preds and objectness
    flatten_cls_scores = [
        cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_classes)
        for cls_score in cls_scores
    ]
    flatten_bbox_preds = [
        bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
        for bbox_pred in bbox_preds
    ]

    flatten_objectness = [
        objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
        for objectness in objectnesses
    ]

    cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
    flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
    flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
    bboxes = self.bbox_coder.decode(flatten_priors[None], flatten_bbox_preds,
                                    flatten_stride)

    # directly multiply score factor and feed to nms
    scores = cls_scores * (flatten_objectness.unsqueeze(-1))

    if not with_nms:
        return bboxes, scores
    deploy_cfg = ctx.cfg
    post_params = get_post_processing_params(deploy_cfg)
    max_output_boxes_per_class = post_params.max_output_boxes_per_class
    iou_threshold = cfg.nms.get('iou_threshold', post_params.iou_threshold)
    score_threshold = cfg.get('score_thr', post_params.score_threshold)
    pre_top_k = post_params.pre_top_k
    keep_top_k = cfg.get('max_per_img', post_params.keep_top_k)

    return multiclass_nms(bboxes, scores, max_output_boxes_per_class,
                          iou_threshold, score_threshold, pre_top_k,
                          keep_top_k)
