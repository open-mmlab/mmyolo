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
from mmdet.utils import ConfigType, OptInstanceList
from mmengine import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmyolo.registry import MODELS, TASK_UTILS
from ..utils import make_divisible
from . import YOLOv8Head, YOLOv8HeadModule


class ProtoModule(nn.Module):
    """Mask Proto module for segmentation models of YOLOv8.

    Args:
        in_channels (int): Number of channels in the input feature map.
        middle_channels (int): Number of channels in the middle feature map.
        masks_channels (int): Number of channels in the output mask feature
            map. This is the channel count of the mask.
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
        """Forward features."""
        return self.conv3(self.conv2(self.upsample(self.conv1(x))))


@MODELS.register_module()
class YOLOv8InsHeadModule(YOLOv8HeadModule):
    """Detection and Instance Segmentation Head of YOLOv8.

    Args:
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        masks_channels (int): Number of channels in the mask feature map.
            This is the channel count of the mask.
        protos_channels (int): Number of channels in the proto feature map.
    """

    def __init__(self,
                 *args,
                 widen_factor: float = 1.0,
                 masks_channels: int = 32,
                 protos_channels: int = 256,
                 **kwargs):
        self.masks_channels = masks_channels
        self.protos_channels = make_divisible(protos_channels, widen_factor)

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
        # Init class and regression head branch.
        super()._init_layers()

        # Init proto preds branch and mask coefficients preds branch.
        self.proto_preds = ProtoModule(
            self.in_channels[0],
            self.protos_channels,
            self.masks_channels,
            act_cfg=self.act_cfg,
            norm_cfg=self.norm_cfg)

        middle_channels = max(self.in_channels[0] // 4, self.masks_channels)
        # mask coefficients preds
        self.mask_coe_preds = nn.ModuleList(
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
                    out_channels=self.masks_channels,
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
    """YOLOv8 Instance Segmentation and Detection head."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        mask_coefficients: List[Tensor],
                        mask_protos: Tensor,
                        score_factors: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = True,
                        with_nms: bool = True) -> List[InstanceData]:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Note: When score_factors is not None, the cls_scores are
        usually multiplied by it then obtain the real score used in NMS.

        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            mask_coefficients (list[Tensor]): Mask coefficients predictions
                for all scale levels, each is a 4D-tensor, has shape
                (batch_size, mask_channels, H, W).
            mask_protos (Tensor): Mask prototype features extracted from the
                mask head, has shape (batch_size, mask_channels, H, W).
            score_factors (list[Tensor], optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, num_priors * 1, H, W). Defaults to None.
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
            list[:obj:`InstanceData`]: Object detection and instance
            segmentation results of each image after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, h, w).
        """
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
        flatten_mcs_preds = [
            kernel_pred.permute(0, 2, 3,
                                1).reshape(num_imgs, -1,
                                           self.head_module.masks_channels)
            for kernel_pred in mask_coefficients
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_decoded_bboxes = self.bbox_coder.decode(
            flatten_priors[..., :2].unsqueeze(0), flatten_bbox_preds,
            flatten_stride)

        flatten_mcs_preds = torch.cat(flatten_mcs_preds, dim=1)

        results_list = []
        for (bboxes, scores, mcs_pred, mask_proto,
             img_meta) in zip(flatten_decoded_bboxes, flatten_cls_scores,
                              flatten_mcs_preds, mask_protos, batch_img_metas):
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
                    results=dict(labels=labels[:, 0], mcs_pred=mcs_pred))
                labels = results['labels']
                mcs_pred = results['mcs_pred']
            else:
                out = filter_scores_and_topk(
                    scores,
                    score_thr,
                    nms_pre,
                    results=dict(mcs_pred=mcs_pred))
                scores, labels, keep_idxs, filtered_results = out
                mcs_pred = filtered_results['mcs_pred']

            results = InstanceData(
                scores=scores,
                labels=labels,
                bboxes=bboxes[keep_idxs],
                mcs_pred=mcs_pred)

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
                mask_feat=mask_proto,
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
        """bbox and mask post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually `with_nms` is False is used for aug test.

        Args:
            results (:obj:`InstaceData`): Detection instance results,
                each item has shape (num_bboxes, ).
            mask_feat (Tensor): Mask prototype features extracted from the
                mask head, has shape (batch_size, num_prototypes, H, W).
            cfg (ConfigDict): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale_bbox (bool): If True, return boxes in original image space.
                Default to False.
            rescale_mask (bool): If True, return masks in original image space.
                Default to True.
            with_nms (bool): If True, do nms before return boxes.
                Default to True.
            img_meta (dict, optional): Image meta info. Defaults to None.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, h, w).
        """
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
        assert with_nms, 'with_nms must be True for YOLOv8-Ins'
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
                                                     results.mcs_pred)
            stride = self.prior_generator.strides[0][0]
            mask_logits = F.interpolate(
                mask_logits.unsqueeze(0),
                scale_factor=stride // 2,
                mode='bilinear')

            # TODO: logic is diffenent from the official.
            if rescale_mask:
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
            # Remove the prediction points on the outside of the box
            mask_logits = self.crop_mask(mask_logits, results.bboxes)
            masks = mask_logits.squeeze(0)
            masks = masks > cfg.mask_thr_binary
            results.masks = masks
        else:
            h, w = img_meta['ori_shape'][:2] if rescale_mask else img_meta[
                'img_shape'][:2]
            results.masks = torch.zeros(
                size=(results.bboxes.shape[0], h, w),
                dtype=torch.bool,
                device=results.bboxes.device)
        return results

    def _mask_predict_by_feat(self, mask_feat: Tensor,
                              kernels: Tensor) -> Tensor:
        """Generate mask logits from mask features with dynamic convs.

        Args:
            mask_feat (Tensor): Mask prototype features.
                Has shape (mask_channels, H, W).
            kernels (Tensor): Kernel parameters for each instance.
                Has shape (num_instance, mask_channels)
        Returns:
            Tensor: Instance segmentation masks for each instance.
                Has shape (num_instance, H, W).
        """
        c, mh, mw = mask_feat.shape
        masks = (kernels @ mask_feat.float().view(c, -1)).sigmoid().view(
            -1, mh, mw)
        return masks

    def crop_mask(self, masks: Tensor, boxes: Tensor):
        """Crop mask by the bounding box.

        Args:
          masks (Tensor): Predicted mask results. Has shape
              (1, num_instance, H, W).
          boxes (Tensor): Tensor of the bbox. Has shape (num_instance, 4).

        Returns:
          (torch.Tensor): The masks are being cropped to the bounding box.
        """
        _, n, h, w = masks.shape
        x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)
        r = torch.arange(
            w, device=masks.device,
            dtype=x1.dtype)[None, None, None, :]  # rows shape(1, 1, w, 1)
        c = torch.arange(
            h, device=masks.device,
            dtype=x1.dtype)[None, None, :, None]  # cols shape(1, h, 1, 1)

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
