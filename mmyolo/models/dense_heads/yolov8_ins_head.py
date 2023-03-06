# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.ops import batched_nms
from mmdet.models.utils import filter_scores_and_topk, multi_apply
from mmdet.structures.bbox import get_box_tensor, get_box_wh, scale_boxes
from mmdet.utils import (ConfigType, OptConfigType, OptInstanceList,
                         OptMultiConfig)
from mmengine import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmyolo.registry import MODELS, TASK_UTILS
from ..utils import make_divisible
from . import YOLOv8Head, YOLOv8HeadModule


class Proto(nn.Module):
    """Mask Proto module for segmentation models of YOLOv8.

    Args:
        in_channels (int):
        middle_channels (int):
        masks_channels (int):
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to ``dict(type='BN', momentum=0.03, eps=0.001)``.
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Default: dict(type='SiLU', inplace=True).
    """

    def __init__(self,
                 in_channels: int,
                 middle_channels: int = 256,
                 masks_channels: int = 32,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True)):
        super().__init__()
        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=middle_channels,
            kernel_size=3,
            padding=1,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg)
        self.upsample = nn.ConvTranspose2d(
            middle_channels, middle_channels, 2, 2, 0, bias=True)
        self.conv2 = ConvModule(
            in_channels=middle_channels,
            out_channels=middle_channels,
            kernel_size=3,
            padding=1,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg)
        self.conv3 = ConvModule(
            in_channels=middle_channels,
            out_channels=masks_channels,
            kernel_size=1,
            padding=0,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg)

    def forward(self, x):
        return self.conv3(self.conv2(self.upsample(self.conv1(x))))


@MODELS.register_module()
class YOLOv8InsHeadModule(YOLOv8HeadModule):

    def __init__(self,
                 *args,
                 widen_factor: float = 1.0,
                 num_masks: int = 32,
                 num_protos: int = 256,
                 **kwargs):
        self.num_masks = num_masks
        self.num_protos = make_divisible(num_protos, widen_factor)

        super().__init__(*args, widen_factor=widen_factor, **kwargs)

    def init_weights(self, prior_prob=0.01):
        """Initialize the weight and bias of PPYOLOE head."""
        super().init_weights()
        for reg_pred, cls_pred, stride in zip(self.reg_preds, self.cls_preds,
                                              self.featmap_strides):
            reg_pred[-1].bias.data[:] = 1.0  # box
            # cls (.01 objects, 80 classes, 640 img)
            cls_pred[-1].bias.data[:self.num_classes] = math.log(
                5 / self.num_classes / (640 / stride)**2)

    def _init_layers(self):
        """initialize conv layers in YOLOv8 head."""
        # Init decouple head
        super()._init_layers()

        # Init proto preds net and mask coefficients preds net
        self.proto_preds = Proto(
            self.in_channels[0],
            self.num_protos,
            self.num_masks,
            act_cfg=self.act_cfg,
            norm_cfg=self.norm_cfg)

        middle_channels = max(self.in_channels[0] // 4, self.num_masks)
        self.mask_coe_preds = nn.ModuleList(  # mask coefficients preds
            nn.Sequential(
                ConvModule(
                    in_channels=in_c,
                    out_channels=middle_channels,
                    kernel_size=3,
                    padding=1,
                    act_cfg=self.act_cfg,
                    norm_cfg=self.norm_cfg),
                ConvModule(
                    in_channels=middle_channels,
                    out_channels=middle_channels,
                    kernel_size=3,
                    padding=1,
                    act_cfg=self.act_cfg,
                    norm_cfg=self.norm_cfg),
                ConvModule(
                    in_channels=middle_channels,
                    out_channels=self.num_masks,
                    kernel_size=1,
                    padding=0,
                    act_cfg=None,
                    norm_cfg=None)) for in_c in self.in_channels)

    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions
        """
        assert len(x) == self.num_levels

        p = self.proto_preds(x[0])

        return *multi_apply(self.forward_single, x, self.cls_preds,
                            self.reg_preds, self.cv4), p

    def forward_single(self, x: torch.Tensor, cls_pred: nn.ModuleList,
                       reg_pred: nn.ModuleList,
                       mask_pred: nn.ModuleList) -> Tuple:
        """Forward feature of a single scale level."""

        # detect prediction
        det_output = super().forward_single(x, cls_pred, reg_pred)
        # mask prediction
        mc = mask_pred(x)
        return *det_output, mc


@MODELS.register_module()
class YOLOv8InsHead(YOLOv8Head):

    def __init__(self,
                 head_module: ConfigType,
                 prior_generator: ConfigType = dict(
                     type='mmdet.MlvlPointGenerator',
                     offset=0.5,
                     strides=[8, 16, 32]),
                 bbox_coder: ConfigType = dict(type='DistancePointBBoxCoder'),
                 loss_cls: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='none',
                     loss_weight=0.5),
                 loss_bbox: ConfigType = dict(
                     type='IoULoss',
                     iou_mode='ciou',
                     bbox_format='xyxy',
                     reduction='sum',
                     loss_weight=7.5,
                     return_iou=False),
                 loss_dfl=dict(
                     type='mmdet.DistributionFocalLoss',
                     reduction='mean',
                     loss_weight=1.5 / 4),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            head_module=head_module,
            prior_generator=prior_generator,
            bbox_coder=bbox_coder,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_dfl=loss_dfl,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)

    def special_init(self):
        """Since YOLO series algorithms will inherit from YOLOv5Head, but
        different algorithms have special initialization process.

        The special_init function is designed to deal with this situation.
        """
        if self.train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg.assigner)

            # Add common attributes to reduce calculation
            self.featmap_sizes_train = None
            self.num_level_priors = None
            self.flatten_priors_train = None
            self.stride_tensor = None

    def predict_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            ####
            mcs: List[Tensor],
            p: Tensor,
            ####
            batch_img_metas: Optional[List[dict]] = None,
            cfg: Optional[ConfigDict] = None,
            rescale: bool = True,
            with_nms: bool = True) -> List[InstanceData]:
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
                device=cls_scores[0].device,
                with_stride=True)
            self.featmap_sizes = featmap_sizes
        flatten_priors = torch.cat(self.mlvl_priors)

        mlvl_strides = [
            flatten_priors.new_full(
                (featmap_size.numel() * self.num_base_priors, ), stride) for
            featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
        ]
        flatten_stride = torch.cat(mlvl_strides)

        # flatten cls_scores, bbox_preds
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.num_classes)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        # flatten_kernel_preds = [
        #     kernel_pred.permute(0, 2, 3,
        #                         1).reshape(num_imgs, -1,
        #                                    self.head_module.num_gen_params)
        #     for kernel_pred in kernel_preds
        # ]
        flatten_mcs_preds = [
            kernel_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                    self.head_module.num_masks)
            for kernel_pred in mcs
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_decoded_bboxes = self.bbox_coder.decode(
            flatten_priors[..., :2].unsqueeze(0), flatten_bbox_preds,
            flatten_stride)

        flatten_mcs_preds = torch.cat(flatten_mcs_preds, dim=1)

        results_list = []
        for (bboxes, scores, mcs_pred, p_single,
             img_meta) in zip(flatten_decoded_bboxes, flatten_cls_scores,
                              flatten_mcs_preds, p, batch_img_metas):
            ori_shape = img_meta['ori_shape']
            scale_factor = img_meta['scale_factor']
            if 'pad_param' in img_meta:
                pad_param = img_meta['pad_param']
            else:
                pad_param = None

            score_thr = cfg.get('score_thr', -1)
            if scores.shape[0] == 0:
                empty_results = InstanceData()
                empty_results.bboxes = bboxes
                empty_results.scores = scores[:, 0]
                empty_results.labels = scores[:, 0].int()
                h, w = ori_shape[:2] if rescale else img_meta['img_shape'][:2]
                empty_results.masks = torch.zeros(
                    size=(0, h, w), dtype=torch.bool, device=bboxes.device)
                results_list.append(empty_results)
                continue

            nms_pre = cfg.get('nms_pre', 100000)
            if cfg.multi_label is False:
                scores, labels = scores.max(1, keepdim=True)
                scores, _, keep_idxs, results = filter_scores_and_topk(
                    scores,
                    score_thr,
                    nms_pre,
                    results=dict(
                        labels=labels[:, 0],
                        mcs_pred=mcs_pred,
                        priors=flatten_priors))
                labels = results['labels']
                mcs_pred = results['mcs_pred']
                priors = results['priors']
            else:
                out = filter_scores_and_topk(
                    scores,
                    score_thr,
                    nms_pre,
                    results=dict(mcs_pred=mcs_pred, priors=flatten_priors))
                scores, labels, keep_idxs, filtered_results = out
                mcs_pred = filtered_results['mcs_pred']
                priors = filtered_results['priors']

            results = InstanceData(
                scores=scores,
                labels=labels,
                bboxes=bboxes[keep_idxs],
                mcs_pred=mcs_pred,
                priors=priors)

            if rescale:
                if pad_param is not None:
                    results.bboxes -= results.bboxes.new_tensor([
                        pad_param[2], pad_param[0], pad_param[2], pad_param[0]
                    ])
                results.bboxes /= results.bboxes.new_tensor(
                    scale_factor).repeat((1, 2))

            if cfg.get('yolox_style', False):
                # do not need max_per_img
                cfg.max_per_img = len(results)

            results = self._bbox_mask_post_process(
                results=results,
                mask_feat=p_single,
                cfg=cfg,
                rescale_bbox=False,
                rescale_mask=rescale,
                with_nms=with_nms,
                pad_param=pad_param,
                img_meta=img_meta)
            results.bboxes[:, 0::2].clamp_(0, ori_shape[1])
            results.bboxes[:, 1::2].clamp_(0, ori_shape[0])

            results_list.append(results)
        return results_list

    def _bbox_mask_post_process(
            self,
            results: InstanceData,
            mask_feat: Tensor,
            cfg: ConfigDict,
            rescale_bbox: bool = False,
            rescale_mask: bool = True,
            with_nms: bool = True,
            pad_param: Optional[np.ndarray] = None,
            img_meta: Optional[dict] = None) -> InstanceData:
        if rescale_bbox:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            results.bboxes = scale_boxes(results.bboxes, scale_factor)

        if hasattr(results, 'score_factors'):
            # TODO： Add sqrt operation in order to be consistent with
            #  the paper.
            score_factors = results.pop('score_factors')
            results.scores = results.scores * score_factors

        # filter small size bboxes
        if cfg.get('min_bbox_size', -1) >= 0:
            w, h = get_box_wh(results.bboxes)
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                results = results[valid_mask]

        # TODO: deal with `with_nms` and `nms_cfg=None` in test_cfg
        assert with_nms, 'with_nms must be True for RTMDet-Ins'
        if results.bboxes.numel() > 0:
            bboxes = get_box_tensor(results.bboxes)
            det_bboxes, keep_idxs = batched_nms(bboxes, results.scores,
                                                results.labels, cfg.nms)
            results = results[keep_idxs]
            # some nms would reweight the score, such as softnms
            results.scores = det_bboxes[:, -1]
            results = results[:cfg.max_per_img]

            # process masks
            mask_logits = self._mask_predict_by_feat(mask_feat,
                                                     results.mcs_pred,
                                                     results.priors)
            stride = self.prior_generator.strides[0][0]
            mask_logits = F.interpolate(
                mask_logits.unsqueeze(0),
                scale_factor=stride // 2,
                mode='bilinear')

            if rescale_mask:
                # TODO: When use mmdet.Resize or mmdet.Pad, will meet bug
                # Use img_meta to crop and resize
                ori_h, ori_w = img_meta['ori_shape'][:2]
                if isinstance(pad_param, np.ndarray):
                    pad_param = pad_param.astype(np.int32)
                    crop_y1, crop_y2 = pad_param[
                        0], mask_logits.shape[-2] - pad_param[1]
                    crop_x1, crop_x2 = pad_param[
                        2], mask_logits.shape[-1] - pad_param[3]
                    mask_logits = mask_logits[..., crop_y1:crop_y2,
                                              crop_x1:crop_x2]
                mask_logits = F.interpolate(
                    mask_logits,
                    size=[ori_h, ori_w],
                    mode='bilinear',
                    align_corners=False)
            mask_logits = self.crop_mask(mask_logits, results.bboxes)
            masks = mask_logits.squeeze(0)
            masks = masks > cfg.mask_thr_binary
            results.masks = masks
            # mask_logits.gt_(cfg.mask_thr_binary)
            # mask_logits = mask_logits.to(torch.uint8)
            #
            # if rescale_mask:
            #     # TODO: When use mmdet.Resize or mmdet.Pad, will meet bug
            #     # Use img_meta to crop and resize
            #     ori_h, ori_w = img_meta['ori_shape'][:2]
            #     if isinstance(pad_param, np.ndarray):
            #         pad_param = pad_param.astype(np.int32)
            #         crop_y1, crop_y2 = pad_param[
            #                                0], mask_logits.shape[-2] - pad_param[1]
            #         crop_x1, crop_x2 = pad_param[
            #                                2], mask_logits.shape[-1] - pad_param[3]
            #         mask_logits = mask_logits[..., crop_y1:crop_y2,
            #                       crop_x1:crop_x2]
            #     mask_logits = F.interpolate(
            #         mask_logits,
            #         size=[ori_h, ori_w],
            #         mode='bilinear',
            #         align_corners=False)
            #
            # results.masks = mask_logits.squeeze(0).to(torch.bool)
        else:
            h, w = img_meta['ori_shape'][:2] if rescale_mask else img_meta[
                'img_shape'][:2]
            results.masks = torch.zeros(
                size=(results.bboxes.shape[0], h, w),
                dtype=torch.bool,
                device=results.bboxes.device)
        return results

    def _mask_predict_by_feat(self, mask_feat: Tensor, kernels: Tensor,
                              priors: Tensor) -> Tensor:
        c, mh, mw = mask_feat.shape
        masks = (kernels @ mask_feat.float().view(c, -1)).sigmoid().view(
            -1, mh, mw)
        return masks

    def crop_mask(self, masks, boxes):
        _, n, h, w = masks.shape
        x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4,
                                     1)  # x1 shape(1,1,n)
        r = torch.arange(
            w, device=masks.device,
            dtype=x1.dtype)[None, None, None, :]  # rows shape(1,w,1)
        c = torch.arange(
            h, device=masks.device, dtype=x1.dtype)[None, None, :,
                                                    None]  # cols shape(h,1,1)

        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    def loss_by_feat(
            self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            bbox_dist_preds: Sequence[Tensor],
            batch_gt_instances: Sequence[InstanceData],
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        raise NotImplementedError
