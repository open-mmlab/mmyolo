# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from typing import List, Optional

import torch
import torch.nn as nn
from mmdet.models.backbones.csp_darknet import Focus
from mmengine.config import ConfigDict
from torch import Tensor

from mmyolo.models import RepVGGBlock
from mmyolo.models.dense_heads import (RTMDetHead, YOLOv5Head, YOLOv7Head,
                                       YOLOXHead)
from mmyolo.models.layers import CSPLayerWithTwoConv
from ..backbone import DeployC2f, DeployFocus, GConvFocus, NcnnFocus
from ..bbox_code import (rtmdet_bbox_decoder, yolov5_bbox_decoder,
                         yolox_bbox_decoder)
from ..nms import batched_nms, efficient_nms, onnx_nms


class DeployModel(nn.Module):

    def __init__(self,
                 baseModel: nn.Module,
                 postprocess_cfg: Optional[ConfigDict] = None):
        super().__init__()
        self.baseModel = baseModel
        if postprocess_cfg is None:
            self.with_postprocess = False
        else:
            self.with_postprocess = True
            self.baseHead = baseModel.bbox_head
            self.__init_sub_attributes()
            self.detector_type = type(self.baseHead)
            self.pre_top_k = postprocess_cfg.get('pre_top_k', 1000)
            self.keep_top_k = postprocess_cfg.get('keep_top_k', 100)
            self.iou_threshold = postprocess_cfg.get('iou_threshold', 0.65)
            self.score_threshold = postprocess_cfg.get('score_threshold', 0.25)
            self.backend = postprocess_cfg.get('backend', 1)
        self.__switch_deploy()

    def __init_sub_attributes(self):
        self.bbox_decoder = self.baseHead.bbox_coder.decode
        self.prior_generate = self.baseHead.prior_generator.grid_priors
        self.num_base_priors = self.baseHead.num_base_priors
        self.featmap_strides = self.baseHead.featmap_strides
        self.num_classes = self.baseHead.num_classes

    def __switch_deploy(self):
        for layer in self.baseModel.modules():
            if isinstance(layer, RepVGGBlock):
                layer.switch_to_deploy()
            elif isinstance(layer, Focus):
                # onnxruntime tensorrt8 tensorrt7
                if self.backend in (1, 2, 3):
                    self.baseModel.backbone.stem = DeployFocus(layer)
                # ncnn
                elif self.backend == 4:
                    self.baseModel.backbone.stem = NcnnFocus(layer)
                # switch focus to group conv
                else:
                    self.baseModel.backbone.stem = GConvFocus(layer)
            elif isinstance(layer, CSPLayerWithTwoConv):
                setattr(layer, '__class__', DeployC2f)

    def pred_by_feat(self,
                     cls_scores: List[Tensor],
                     bbox_preds: List[Tensor],
                     objectnesses: Optional[List[Tensor]] = None,
                     **kwargs):
        assert len(cls_scores) == len(bbox_preds)
        dtype = cls_scores[0].dtype
        device = cls_scores[0].device

        nms_func = self.select_nms()
        if self.detector_type in (YOLOv5Head, YOLOv7Head):
            bbox_decoder = yolov5_bbox_decoder
        elif self.detector_type is RTMDetHead:
            bbox_decoder = rtmdet_bbox_decoder
        elif self.detector_type is YOLOXHead:
            bbox_decoder = yolox_bbox_decoder
        else:
            bbox_decoder = self.bbox_decoder

        num_imgs = cls_scores[0].shape[0]
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

        mlvl_priors = self.prior_generate(
            featmap_sizes, dtype=dtype, device=device)

        flatten_priors = torch.cat(mlvl_priors)

        mlvl_strides = [
            flatten_priors.new_full(
                (featmap_size[0] * featmap_size[1] * self.num_base_priors, ),
                stride) for featmap_size, stride in zip(
                    featmap_sizes, self.featmap_strides)
        ]
        flatten_stride = torch.cat(mlvl_strides)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.num_classes)
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

        return nms_func(bboxes, scores, self.keep_top_k, self.iou_threshold,
                        self.score_threshold, self.pre_top_k, self.keep_top_k)

    def select_nms(self):
        if self.backend == 1:
            nms_func = onnx_nms
        elif self.backend == 2:
            nms_func = efficient_nms
        elif self.backend == 3:
            nms_func = batched_nms
        else:
            raise NotImplementedError
        if type(self.baseHead) in (YOLOv5Head, YOLOv7Head, YOLOXHead):
            nms_func = partial(nms_func, box_coding=1)

        return nms_func

    def forward(self, inputs: Tensor):
        neck_outputs = self.baseModel(inputs)
        if self.with_postprocess:
            return self.pred_by_feat(*neck_outputs)
        else:
            return neck_outputs
