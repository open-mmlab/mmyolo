# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.models.utils import filter_scores_and_topk, multi_apply
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
        self.mask_coeff_preds = nn.ModuleList(
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

    def forward(self, x: Tuple[Tensor]) -> Tuple:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions
        """
        assert len(x) == self.num_levels

        mask_protos = self.proto_preds(x[0])
        output = multi_apply(self.forward_single, x, self.cls_preds,
                             self.reg_preds, self.mask_coeff_preds)
        output = *output, mask_protos

        return output

    def forward_single(self, x: torch.Tensor, cls_pred: nn.ModuleList,
                       reg_pred: nn.ModuleList,
                       mask_coeff_pred: nn.ModuleList) -> Tuple:
        """Forward feature of a single scale level."""

        # detect prediction
        det_output = super().forward_single(x, cls_pred, reg_pred)
        # mask prediction
        mask_coefficient = mask_coeff_pred(x)
        output = *det_output, mask_coefficient
        return output


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

        flatten_stride = flatten_priors[:, -1]

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
        flatten_mask_coeff_preds = [
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

        flatten_mask_coeff_preds = torch.cat(flatten_mask_coeff_preds, dim=1)

        results_list = []
        for (bboxes, scores, mask_coeff_pred, mask_proto,
             img_meta) in zip(flatten_decoded_bboxes, flatten_cls_scores,
                              flatten_mask_coeff_preds, mask_protos,
                              batch_img_metas):
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
                        labels=labels[:, 0], mask_coeff_pred=mask_coeff_pred))
                labels = results['labels']
                mask_coeff_pred = results['mask_coeff_pred']
            else:
                out = filter_scores_and_topk(
                    scores,
                    score_thr,
                    nms_pre,
                    results=dict(mask_coeff_pred=mask_coeff_pred))
                scores, labels, keep_idxs, filtered_results = out
                mask_coeff_pred = filtered_results['mask_coeff_pred']

            results = InstanceData(
                scores=scores,
                labels=labels,
                bboxes=bboxes[keep_idxs],
                mask_coeff_pred=mask_coeff_pred)

            results = self._bbox_post_process(
                results=results,
                cfg=cfg,
                rescale=False,
                with_nms=with_nms,
                img_meta=img_meta)

            input_shape_h, input_shape_w = img_meta['batch_input_shape'][:2]
            masks = self.process_mask(
                mask_proto,
                results.mask_coeff_pred,
                results.bboxes, (input_shape_h, input_shape_w),
                rescale,
                mask_thr_binary=cfg.mask_thr_binary)[0]

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
                # TODO: Running speed is very slow and needs to be optimized
                # Now, the logic remains the same as official
                masks = masks.permute(1, 2, 0).contiguous().cpu().numpy()
                # astype(np.uint8) is very important
                masks = cv2.resize(
                    masks.astype(np.uint8), (ori_shape[1], ori_shape[0]))

                if len(masks.shape) == 2:
                    masks = masks[:, :, None]

                masks = torch.from_numpy(masks).permute(2, 0, 1)

            results.bboxes[:, 0::2].clamp_(0, ori_shape[1])
            results.bboxes[:, 1::2].clamp_(0, ori_shape[0])

            results.masks = masks.bool()
            results_list.append(results)
        return results_list

    def process_mask(self,
                     mask_proto: Tensor,
                     mask_coeff_pred: Tensor,
                     bboxes: Tensor,
                     shape: Tuple[int, int],
                     upsample: bool = False,
                     mask_thr_binary: float = 0.5):
        """Generate mask logits results.

        Args:

            mask_proto (Tensor): Mask prototype features.
                Has shape (num_instance, masks_channels).
            mask_coeff_pred (Tensor): Mask coefficients prediction for
                single image. Has shape (masks_channels, H, W)
            bboxes (Tensor): Tensor of the bbox. Has shape (num_instance, 4).
            shape (Tuple): Batch input shape of image.
            upsample (bool): Whether upsample masks results to batch input
                shape. Default to False.
            mask_thr_binary (float): Threshold of mask prediction. Default
                to 0.5.

        Return:
            Tensor: Instance segmentation masks for each instance.
                Has shape (num_instance, H, W).
        """

        c, mh, mw = mask_proto.shape  # CHW
        masks = (
            mask_coeff_pred @ mask_proto.float().view(c, -1)).sigmoid().view(
                -1, mh, mw)
        if upsample:
            masks = F.interpolate(
                masks[None], shape, mode='bilinear',
                align_corners=False)  # CHW
        masks = self.crop_mask(masks, bboxes)  # CHW
        return masks.gt_(mask_thr_binary)

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
