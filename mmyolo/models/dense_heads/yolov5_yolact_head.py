# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.models.utils import filter_scores_and_topk, multi_apply
from mmdet.utils import (ConfigType, OptConfigType, OptInstanceList,
                         OptMultiConfig)
from mmengine.config import ConfigDict
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor

from mmyolo.registry import MODELS
from ..utils import make_divisible
from .yolov5_head import YOLOv5Head, YOLOv5HeadModule


def crop_mask(masks, boxes):
    """"Crop" predicted masks by zeroing out everything not in the predicted
    bbox. Vectorized by Chong (thanks Chong).

    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox
        coords in relative point form
    """

    n, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
    r = torch.arange(
        w, device=masks.device, dtype=x1.dtype)[None,
                                                None, :]  # rows shape(1,w,1)
    c = torch.arange(
        h, device=masks.device, dtype=x1.dtype)[None, :,
                                                None]  # cols shape(h,1,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


class Proto(BaseModule):

    def __init__(self,
                 in_channels,
                 proto_channels=256,
                 num_protos=32,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg=None):
        super().__init__(init_cfg)
        self.cv1 = ConvModule(
            in_channels,
            proto_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = ConvModule(
            proto_channels,
            proto_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.cv3 = ConvModule(
            proto_channels,
            num_protos,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


@MODELS.register_module()
class YOLOv5YOLACTHeadModule(YOLOv5HeadModule):

    def __init__(self,
                 num_classes,
                 *args,
                 num_protos=32,
                 proto_channels=256,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 widen_factor=1.0,
                 **kwargs):
        self.num_protos = num_protos
        self.num_out_attrib_with_proto = 5 + num_classes + num_protos
        self.proto_channels = make_divisible(proto_channels, widen_factor)
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        super().__init__(
            *args,
            **kwargs,
            num_classes=num_classes,
            widen_factor=widen_factor)

    def _init_layers(self):
        """initialize conv layers in YOLOv5 head."""
        self.convs_pred = nn.ModuleList()
        for i in range(self.num_levels):
            conv_pred = nn.Conv2d(
                self.in_channels[i],
                self.num_base_priors * self.num_out_attrib_with_proto, 1)

            self.convs_pred.append(conv_pred)

        self.proto = Proto(
            self.in_channels[0],
            self.proto_channels,
            self.num_protos,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions, and objectnesses.
        """
        assert len(x) == self.num_levels
        cls_scores, bbox_preds, objectnesses, coeff_preds = multi_apply(
            self.forward_single, x, self.convs_pred)
        segm_preds = self.proto(x[0])
        return cls_scores, bbox_preds, objectnesses, coeff_preds, segm_preds

    def forward_single(
            self, x: Tensor,
            convs: nn.Module) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward feature of a single scale level."""

        pred_map = convs(x)
        bs, _, ny, nx = pred_map.shape
        pred_map = pred_map.view(bs, self.num_base_priors,
                                 self.num_out_attrib_with_proto, ny, nx)

        cls_score = pred_map[:, :, 5:self.num_classes + 5,
                             ...].reshape(bs, -1, ny, nx)
        bbox_pred = pred_map[:, :, :4, ...].reshape(bs, -1, ny, nx)
        objectness = pred_map[:, :, 4:5, ...].reshape(bs, -1, ny, nx)
        coeff_pred = pred_map[:, :, self.num_classes + 5:,
                              ...].reshape(bs, -1, ny, nx)

        return cls_score, bbox_pred, objectness, coeff_pred


@MODELS.register_module()
class YOLOv5YOLACTHead(YOLOv5Head):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_protos = self.head_module.num_protos

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        objectnesses: Optional[List[Tensor]] = None,
                        coeff_preds: Optional[List[Tensor]] = None,
                        segm_preds: Optional[Tensor] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = True,
                        with_nms: bool = True) -> List[InstanceData]:
        assert len(cls_scores) == len(bbox_preds) == len(coeff_preds)
        if objectnesses is None:
            with_objectnesses = False
        else:
            with_objectnesses = True
            assert len(cls_scores) == len(objectnesses)

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)

        multi_label = cfg.multi_label
        multi_label &= self.num_classes > 1
        cfg.multi_label = multi_label

        num_imgs = len(batch_img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

        # If the shape does not change, use the previous mlvl_priors
        if featmap_sizes != self.featmap_sizes:
            self.mlvl_priors = self.prior_generator.grid_priors(
                featmap_sizes,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device)
            self.featmap_sizes = featmap_sizes
        flatten_priors = torch.cat(self.mlvl_priors)

        mlvl_strides = [
            flatten_priors.new_full(
                (featmap_size.numel() * self.num_base_priors, ), stride) for
            featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
        ]
        flatten_stride = torch.cat(mlvl_strides)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.num_classes)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_coeff_preds = [
            coeff_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                   self.num_protos)
            for coeff_pred in coeff_preds
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_decoded_bboxes = self.bbox_coder.decode(
            flatten_priors[None], flatten_bbox_preds, flatten_stride)

        flatten_coeff_preds = torch.cat(flatten_coeff_preds, dim=1)

        if with_objectnesses:
            flatten_objectness = [
                objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
                for objectness in objectnesses
            ]
            flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        else:
            flatten_objectness = [None for _ in range(len(featmap_sizes))]

        results_list = []
        for (bboxes, scores, objectness, coeff_preds, segm_pred,
             img_meta) in zip(flatten_decoded_bboxes, flatten_cls_scores,
                              flatten_objectness, flatten_coeff_preds,
                              segm_preds, batch_img_metas):
            ori_shape = img_meta['ori_shape']
            scale_factor = img_meta['scale_factor']
            if 'pad_param' in img_meta:
                pad_param = img_meta['pad_param']
            else:
                pad_param = None

            score_thr = cfg.get('score_thr', -1)
            # yolox_style does not require the following operations
            if objectness is not None and score_thr > 0 and not cfg.get(
                    'yolox_style', False):
                conf_inds = objectness > score_thr
                bboxes = bboxes[conf_inds, :]
                scores = scores[conf_inds, :]
                objectness = objectness[conf_inds]
                coeff_preds = coeff_preds[conf_inds]

            if objectness is not None:
                # conf = obj_conf * cls_conf
                scores *= objectness[:, None]
                # NOTE: Important
                coeff_preds *= objectness[:, None]

            if scores.shape[0] == 0:
                empty_results = InstanceData()
                empty_results.bboxes = bboxes
                empty_results.scores = scores[:, 0]
                empty_results.labels = scores[:, 0].int()
                im_masks = torch.zeros(
                    0,
                    ori_shape[0],
                    ori_shape[1],
                    device=bboxes.device,
                    dtype=torch.bool)
                empty_results.masks = im_masks
                results_list.append(empty_results)
                continue

            nms_pre = cfg.get('nms_pre', 100000)
            if cfg.multi_label is False:
                scores, labels = scores.max(1, keepdim=True)
                scores, _, keep_idxs, results = filter_scores_and_topk(
                    scores,
                    score_thr,
                    nms_pre,
                    results=dict(labels=labels[:, 0]))
                labels = results['labels']
            else:
                scores, labels, keep_idxs, _ = filter_scores_and_topk(
                    scores, score_thr, nms_pre)

            results = InstanceData(
                scores=scores,
                labels=labels,
                bboxes=bboxes[keep_idxs],
                coeffs=coeff_preds[keep_idxs])

            if cfg.get('yolox_style', False):
                # do not need max_per_img
                cfg.max_per_img = len(results)

            results = self._bbox_post_process(
                results=results,
                cfg=cfg,
                rescale=False,
                with_nms=with_nms,
                img_meta=img_meta)

            input_shape_h, input_shape_w = img_meta['batch_input_shape'][:2]
            masks = self.process_mask(segm_pred, results.coeffs,
                                      results.bboxes,
                                      (input_shape_h, input_shape_w), True)

            if rescale:
                if pad_param is not None:
                    top_pad, bottom_pad, left_pad, right_pad = pad_param

                    results.bboxes -= results.bboxes.new_tensor(
                        [left_pad, top_pad, left_pad, top_pad])
                    top, left = int(top_pad), int(left_pad)
                    bottom, right = int(input_shape_h -
                                        top_pad), int(input_shape_w - left_pad)
                    masks = masks[:, top:bottom, left:right]

                results.bboxes /= results.bboxes.new_tensor(
                    scale_factor).repeat((1, 2))

                import cv2
                import numpy as np
                masks = masks.permute(1, 2, 0).contiguous().cpu().numpy()
                # astype(np.uint8) is very important
                masks = cv2.resize(
                    masks.astype(np.uint8), (ori_shape[1], ori_shape[0]))

                if len(masks.shape) == 2:
                    masks = masks[:, :, None]

                masks = torch.from_numpy(masks).permute(2, 0, 1)

                # masks = F.interpolate(
                #     masks[None],
                #     ori_shape[:2],
                #     mode='bilinear',
                #     align_corners=False)[0]

            results.bboxes[:, 0::2].clamp_(0, ori_shape[1])
            results.bboxes[:, 1::2].clamp_(0, ori_shape[0])

            results.masks = masks.bool()
            results_list.append(results)
        return results_list

    def process_mask(self, protos, masks_in, bboxes, shape, upsample=False):
        """
        Crop before upsample.
        proto_out: [mask_dim, mask_h, mask_w]
        out_masks: [n, mask_dim], n is number of masks after nms
        bboxes: [n, 4], n is number of masks after nms
        shape:input_image_size, (h, w)

        return: h, w, n
        """
        c, mh, mw = protos.shape  # CHW
        masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(
            -1, mh, mw)
        masks = F.interpolate(
            masks[None], shape, mode='bilinear', align_corners=False)[0]  # CHW
        masks = crop_mask(masks, bboxes)  # CHW
        return masks.gt_(0.5)

        # c, mh, mw = protos.shape  # CHW
        # ih, iw = shape
        # masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(
        #     -1, mh, mw)  # CHW
        #
        # downsampled_bboxes = bboxes.clone()
        # downsampled_bboxes[:, 0] *= mw / iw
        # downsampled_bboxes[:, 2] *= mw / iw
        # downsampled_bboxes[:, 3] *= mh / ih
        # downsampled_bboxes[:, 1] *= mh / ih
        #
        # masks = crop_mask(masks, downsampled_bboxes)  # CHW
        # if upsample:
        #     masks = F.interpolate(
        #         masks[None], shape, mode='bilinear',
        #         align_corners=False)[0]  # CHW
        # return masks.gt_(0.5)
