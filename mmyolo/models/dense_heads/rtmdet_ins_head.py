# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, is_norm
from mmdet.models.utils import filter_scores_and_topk, multi_apply
from mmdet.structures.bbox import distance2bbox
from mmdet.utils import ConfigType, InstanceList, OptInstanceList, reduce_mean
from mmengine.model import (BaseModule, bias_init_with_prob, constant_init,
                            normal_init)
from mmengine.structures import InstanceData
from torch import Tensor

from mmyolo.registry import MODELS
from .rtmdet_head import RTMDetHead, RTMDetSepBNHeadModule


@MODELS.register_module()
class RTMDetInsSepBNHeadModule(RTMDetSepBNHeadModule):
    """Detection Head of RTMDet-Ins with sep-bn layers.

    Args:
        num_prototypes (int): Number of mask prototype features extracted
            from the mask head. Defaults to 8.
        dyconv_channels (int): Channel of the dynamic conv layers.
            Defaults to 8.
        num_dyconvs (int): Number of the dynamic convolution layers.
            Defaults to 3.
        share_conv (bool): Whether to share conv layers between stages.
            Defaults to True.
    """

    def __init__(self,
                 *args,
                 num_prototypes: int = 8,
                 dyconv_channels: int = 8,
                 num_dyconvs: int = 3,
                 share_conv: bool = True,
                 **kwargs) -> None:
        self.share_conv = share_conv
        self.num_prototypes = num_prototypes
        self.num_dyconvs = num_dyconvs
        self.dyconv_channels = dyconv_channels
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.kernel_convs = nn.ModuleList()

        self.rtm_cls = nn.ModuleList()
        self.rtm_reg = nn.ModuleList()
        self.rtm_kernel = nn.ModuleList()
        self.rtm_obj = nn.ModuleList()

        # calculate num dynamic parameters
        weight_nums, bias_nums = [], []
        for i in range(self.num_dyconvs):
            if i == 0:
                weight_nums.append(
                    (self.num_prototypes + 2) * self.dyconv_channels)
                bias_nums.append(self.dyconv_channels)
            elif i == self.num_dyconvs - 1:
                weight_nums.append(self.dyconv_channels)
                bias_nums.append(1)
            else:
                weight_nums.append(self.dyconv_channels * self.dyconv_channels)
                bias_nums.append(self.dyconv_channels)
        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)
        pred_pad_size = self.pred_kernel_size // 2

        for n in range(len(self.featmap_strides)):
            cls_convs = nn.ModuleList()
            reg_convs = nn.ModuleList()
            kernel_convs = nn.ModuleList()
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                cls_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                reg_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                kernel_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(cls_convs)
            self.kernel_convs.append(kernel_convs)

            self.rtm_cls.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * self.num_classes,
                    self.pred_kernel_size,
                    padding=pred_pad_size))
            self.rtm_reg.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * 4,
                    self.pred_kernel_size,
                    padding=pred_pad_size))
            self.rtm_kernel.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_gen_params,
                    self.pred_kernel_size,
                    padding=pred_pad_size))

        if self.share_conv:
            for n in range(len(self.featmap_strides)):
                for i in range(self.stacked_convs):
                    self.cls_convs[n][i].conv = self.cls_convs[0][i].conv
                    self.reg_convs[n][i].conv = self.reg_convs[0][i].conv

        self.mask_head = MaskFeatModule(
            in_channels=self.in_channels,
            feat_channels=self.feat_channels,
            stacked_convs=4,
            num_levels=len(self.featmap_strides),
            num_prototypes=self.num_prototypes,
            act_cfg=self.act_cfg,
            norm_cfg=self.norm_cfg)

    def init_weights(self) -> None:
        """Initialize weights of the head."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)
            if is_norm(m):
                constant_init(m, 1)
        bias_cls = bias_init_with_prob(0.01)
        for rtm_cls, rtm_reg, rtm_kernel in zip(self.rtm_cls, self.rtm_reg,
                                                self.rtm_kernel):
            normal_init(rtm_cls, std=0.01, bias=bias_cls)
            normal_init(rtm_reg, std=0.01, bias=1)

    def forward(self, feats: Tuple[Tensor, ...]) -> tuple:
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
            - cls_scores (list[Tensor]): Classification scores for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * num_classes.
            - bbox_preds (list[Tensor]): Box energies / deltas for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * 4.
            - kernel_preds (list[Tensor]): Dynamic conv kernels for all scale
              levels, each is a 4D-tensor, the channels number is
              num_gen_params.
            - mask_feat (Tensor): Output feature of the mask head. Each is a
              4D-tensor, the channels number is num_prototypes.
        """
        mask_feat = self.mask_head(feats)

        cls_scores = []
        bbox_preds = []
        kernel_preds = []
        for idx, x in enumerate(feats):
            cls_feat = x
            reg_feat = x
            kernel_feat = x

            for cls_layer in self.cls_convs[idx]:
                cls_feat = cls_layer(cls_feat)
            cls_score = self.rtm_cls[idx](cls_feat)

            for kernel_layer in self.kernel_convs[idx]:
                kernel_feat = kernel_layer(kernel_feat)
            kernel_pred = self.rtm_kernel[idx](kernel_feat)

            for reg_layer in self.reg_convs[idx]:
                reg_feat = reg_layer(reg_feat)

            reg_dist = F.relu(self.rtm_reg[idx](reg_feat))

            cls_scores.append(cls_score)
            bbox_preds.append(reg_dist)
            kernel_preds.append(kernel_pred)
        return tuple(cls_scores), tuple(bbox_preds), tuple(
            kernel_preds), mask_feat


@MODELS.register_module()
class RTMDetInsHead(RTMDetHead):
    """Detection Head of RTMDet-Ins.

    Args:
        mask_loss_stride (int): Down sample stride of the masks for loss
            computation. Defaults to 4.
        loss_mask (:obj:`ConfigDict` or dict): Config dict for mask loss.
    """

    def __init__(self,
                 *args,
                 mask_loss_stride: int = 4,
                 loss_mask=dict(
                     type='mmdet.DiceLoss',
                     loss_weight=2.0,
                     eps=5e-6,
                     reduction='mean'),
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_gen_params = self.head_module.num_gen_params
        self.weight_nums = self.head_module.weight_nums
        self.bias_nums = self.head_module.bias_nums
        self.dyconv_channels = self.head_module.dyconv_channels

        self.mask_loss_stride = mask_loss_stride
        self.loss_mask = MODELS.build(loss_mask)

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        kernel_preds: List[Tensor],
                        mask_feat: Tensor,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigType] = None,
                        rescale: bool = True,
                        with_nms: bool = True) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Note: When score_factors is not None, the cls_scores are
        usually multiplied by it then obtain the real score used in NMS,
        such as CenterNess in FCOS, IoU branch in ATSS.

        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            kernel_preds (list[Tensor]): Kernel predictions of dynamic
                convs for all scale levels, each is a 4D-tensor, has shape
                (batch_size, num_params, H, W).
            mask_feat (Tensor): Mask prototype features extracted from the
                mask head, has shape (batch_size, num_prototypes, H, W).
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
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, h, w).
        """
        assert len(cls_scores) == len(bbox_preds)

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)

        multi_label = cfg.get('multi_label', True)
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

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_decoded_bboxes = self.bbox_coder.decode(
            flatten_priors[None], flatten_bbox_preds, flatten_stride)

        flatten_kernel_preds = [
            kernel_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                    self.num_gen_params)
            for kernel_pred in kernel_preds
        ]
        flatten_kernel_preds = torch.cat(flatten_kernel_preds, dim=1)

        nms_pre = cfg.get('nms_pre', 100000)
        score_thr = cfg.get('score_thr', -1)
        stride = self.prior_generator.strides[0][0]

        results_list = []
        for (bboxes, scores, kernel_preds, mask_feat_pre_batch,
             img_meta) in zip(flatten_decoded_bboxes, flatten_cls_scores,
                              flatten_kernel_preds, mask_feat,
                              batch_img_metas):
            ori_shape = img_meta['ori_shape']
            scale_factor = img_meta['scale_factor']
            input_shape_h, input_shape_w = img_meta['batch_input_shape'][:2]
            if 'pad_param' in img_meta:
                pad_param = img_meta['pad_param']
            else:
                pad_param = None

            if cfg.multi_label is False:
                scores, labels = scores.max(1, keepdim=True)
                scores, _, keep_idxs, results = filter_scores_and_topk(
                    scores,
                    score_thr,
                    nms_pre,
                    results=dict(
                        labels=labels[:, 0],
                        kernel_preds=kernel_preds,
                        strides=flatten_stride,
                        priors=flatten_priors))
                labels = results['labels']
            else:
                scores, labels, keep_idxs, results = filter_scores_and_topk(
                    scores,
                    score_thr,
                    nms_pre,
                    results=dict(
                        kernel_preds=kernel_preds,
                        strides=flatten_stride,
                        priors=flatten_priors))

            priors = results['priors']
            kernel_preds = results['kernel_preds']
            strides = results['strides']

            results = InstanceData(
                scores=scores,
                labels=labels,
                bboxes=bboxes[keep_idxs],
                priors=priors,
                strides=strides,
                kernel_preds=kernel_preds)

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

            results = self._bbox_post_process(
                results=results,
                cfg=cfg,
                rescale=False,
                with_nms=with_nms,
                img_meta=img_meta)

            if results.bboxes.numel() > 0:
                results.bboxes[:, 0::2].clamp_(0, ori_shape[1])
                results.bboxes[:, 1::2].clamp_(0, ori_shape[0])

                # process mask
                mask_logits = self._mask_predict_by_feat_single(
                    mask_feat_pre_batch, results.kernel_preds, results.priors,
                    results.strides)

                mask_logits = F.interpolate(
                    mask_logits.unsqueeze(0),
                    scale_factor=stride,
                    mode='bilinear')

                if rescale:
                    if pad_param is not None:
                        top_pad, bottom_pad, left_pad, right_pad = pad_param
                        top, left = int(top_pad), int(left_pad)
                        bottom, right = int(input_shape_h -
                                            top_pad), int(input_shape_w -
                                                          left_pad)
                        mask_logits = mask_logits[..., top:bottom, left:right]

                    mask_logits = F.interpolate(
                        mask_logits,
                        size=ori_shape[:2],
                        mode='bilinear',
                        align_corners=False)

                masks = mask_logits.sigmoid().squeeze(0)
                masks = masks > cfg.mask_thr_binary
                results.masks = masks
            else:
                h, w = ori_shape[:2] if rescale else img_meta[
                    'batch_input_shape'][:2]
                results.masks = torch.zeros(
                    size=(results.bboxes.shape[0], h, w),
                    dtype=torch.bool,
                    device=results.bboxes.device)

            results_list.append(results)
        return results_list

    def parse_dynamic_params(self, flatten_kernels: Tensor) -> tuple:
        """split kernel head prediction to conv weight and bias."""
        n_inst = flatten_kernels.size(0)
        n_layers = len(self.weight_nums)
        params_splits = list(
            torch.split_with_sizes(
                flatten_kernels, self.weight_nums + self.bias_nums, dim=1))
        weight_splits = params_splits[:n_layers]
        bias_splits = params_splits[n_layers:]
        for i in range(n_layers):
            if i < n_layers - 1:
                weight_splits[i] = weight_splits[i].reshape(
                    n_inst * self.dyconv_channels, -1, 1, 1)
                bias_splits[i] = bias_splits[i].reshape(n_inst *
                                                        self.dyconv_channels)
            else:
                weight_splits[i] = weight_splits[i].reshape(n_inst, -1, 1, 1)
                bias_splits[i] = bias_splits[i].reshape(n_inst)

        return weight_splits, bias_splits

    def _mask_predict_by_feat_single(
            self,
            mask_feat: Tensor,
            kernels: Tensor,
            priors: Tensor,
            strides: Optional[Tensor] = None) -> Tensor:
        """Generate mask logits from mask features with dynamic convs.

        Args:
            mask_feat (Tensor): Mask prototype features.
                Has shape (num_prototypes, H, W).
            kernels (Tensor): Kernel parameters for each instance.
                Has shape (num_instance, num_params)
            priors (Tensor): Center priors for each instance.
                Has shape (num_instance, 4).
        Returns:
            Tensor: Instance segmentation masks for each instance.
                Has shape (num_instance, H, W).
        """
        num_inst = priors.shape[0]
        h, w = mask_feat.size()[-2:]
        if num_inst < 1:
            return torch.empty(
                size=(num_inst, h, w),
                dtype=mask_feat.dtype,
                device=mask_feat.device)
        if len(mask_feat.shape) < 4:
            mask_feat.unsqueeze(0)

        coord = self.prior_generator.single_level_grid_priors(
            (h, w), level_idx=0).reshape(1, -1, 2)
        num_inst = priors.shape[0]
        points = priors[:, :2].reshape(-1, 1, 2)
        if strides is None:
            strides = priors[:, 2:].reshape(-1, 1, 2)
        else:
            strides = strides[:, None].repeat(1, 2).reshape(-1, 1, 2)
        relative_coord = (points - coord).permute(0, 2, 1) / (
            strides[..., 0].reshape(-1, 1, 1) * 8)
        relative_coord = relative_coord.reshape(num_inst, 2, h, w)

        mask_feat = torch.cat(
            [relative_coord,
             mask_feat.repeat(num_inst, 1, 1, 1)], dim=1)
        weights, biases = self.parse_dynamic_params(kernels)

        n_layers = len(weights)
        x = mask_feat.reshape(1, -1, h, w)
        for i, (weight, bias) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, weight, bias=bias, stride=1, padding=0, groups=num_inst)
            if i < n_layers - 1:
                x = F.relu(x)
        x = x.reshape(num_inst, h, w)
        return x

    def loss_mask_by_feat(self, mask_feats: Tensor, flatten_kernels: Tensor,
                          sampling_results_list: list,
                          batch_gt_instances: InstanceList) -> Tensor:
        """Compute instance segmentation loss.

        Args:
            mask_feats (list[Tensor]): Mask prototype features extracted from
                the mask head. Has shape (N, num_prototypes, H, W)
            flatten_kernels (list[Tensor]): Kernels of the dynamic conv layers.
                Has shape (N, num_instances, num_params)
            sampling_results_list (list[:obj:`SamplingResults`]) Batch of
                assignment results.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.

        Returns:
            Tensor: The mask loss tensor.
        """
        batch_pos_mask_logits = []
        pos_gt_masks = []
        for idx, (mask_feat, kernels, sampling_results,
                  gt_instances) in enumerate(
                      zip(mask_feats, flatten_kernels, sampling_results_list,
                          batch_gt_instances)):
            pos_priors = sampling_results.pos_priors
            pos_inds = sampling_results.pos_inds
            pos_kernels = kernels[pos_inds]  # n_pos, num_gen_params
            pos_mask_logits = self._mask_predict_by_feat_single(
                mask_feat, pos_kernels, pos_priors)
            if gt_instances.masks.numel() == 0:
                gt_masks = torch.empty_like(gt_instances.masks)
            else:
                gt_masks = gt_instances.masks[
                    sampling_results.pos_assigned_gt_inds, :]
            batch_pos_mask_logits.append(pos_mask_logits)
            pos_gt_masks.append(gt_masks)

        pos_gt_masks = torch.cat(pos_gt_masks, 0)
        batch_pos_mask_logits = torch.cat(batch_pos_mask_logits, 0)

        # avg_factor
        num_pos = batch_pos_mask_logits.shape[0]
        num_pos = reduce_mean(mask_feats.new_tensor([num_pos
                                                     ])).clamp_(min=1).item()

        if batch_pos_mask_logits.shape[0] == 0:
            return mask_feats.sum() * 0

        scale = self.prior_generator.strides[0][0] // self.mask_loss_stride
        # upsample pred masks
        batch_pos_mask_logits = F.interpolate(
            batch_pos_mask_logits.unsqueeze(0),
            scale_factor=scale,
            mode='bilinear',
            align_corners=False).squeeze(0)
        # downsample gt masks
        pos_gt_masks = pos_gt_masks[:, self.mask_loss_stride //
                                    2::self.mask_loss_stride,
                                    self.mask_loss_stride //
                                    2::self.mask_loss_stride]

        loss_mask = self.loss_mask(
            batch_pos_mask_logits,
            pos_gt_masks,
            weight=None,
            avg_factor=num_pos)

        return loss_mask

    def loss_by_feat(self,
                     cls_scores: List[Tensor],
                     bbox_preds: List[Tensor],
                     kernel_preds: List[Tensor],
                     mask_feat: Tensor,
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Decoded box for each scale
                level with shape (N, num_anchors * 4, H, W) in
                [tl_x, tl_y, br_x, br_y] format.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_imgs = len(batch_img_metas)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        # If the shape does not equal, generate new one
        if featmap_sizes != self.featmap_sizes:
            self.featmap_sizes = featmap_sizes
            self.multi_level_anchors = self.prior_generator.grid_priors(
                featmap_sizes, device=device, with_stride=True)
        anchor_list = [self.multi_level_anchors for _ in range(num_imgs)]

        flatten_cls_scores = torch.cat([
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.cls_out_channels)
            for cls_score in cls_scores
        ], 1)
        flatten_kernels = torch.cat([
            kernel_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                    self.num_gen_params)
            for kernel_pred in kernel_preds
        ], 1)

        decoded_bboxes = []
        for anchor, bbox_pred, stride in zip(anchor_list[0], bbox_preds,
                                             self.featmap_strides):
            anchor = anchor.reshape(-1, 4)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            bbox_pred = distance2bbox(anchor, bbox_pred * stride)
            decoded_bboxes.append(bbox_pred)

        flatten_bboxes = torch.cat(decoded_bboxes, 1)

        batch_instance_list = []
        bboxes_labels, masks = batch_gt_instances
        for i in range(num_imgs):
            single_bboxes_labels = bboxes_labels[bboxes_labels[:, 0] == i, :]
            single_masks = masks[bboxes_labels[:, 0] == i, :]
            single_gt_instance = InstanceData(
                bboxes=single_bboxes_labels[:, 2:],
                labels=single_bboxes_labels[:, 1],
                masks=single_masks)
            batch_instance_list.append(single_gt_instance)
        batch_gt_instances = batch_instance_list

        cls_reg_targets = self.get_targets(
            flatten_cls_scores,
            flatten_bboxes,
            anchor_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)
        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         assign_metrics_list, sampling_results_list) = cls_reg_targets

        losses_cls, losses_bbox, \
            cls_avg_factors, bbox_avg_factors = multi_apply(
                self.loss_by_feat_single,
                cls_scores,
                decoded_bboxes,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                assign_metrics_list,
                self.prior_generator.strides)

        cls_avg_factor = reduce_mean(sum(cls_avg_factors)).clamp_(min=1).item()
        losses_cls = list(map(lambda x: x / cls_avg_factor, losses_cls))

        bbox_avg_factor = reduce_mean(
            sum(bbox_avg_factors)).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))

        loss_mask = self.loss_mask_by_feat(mask_feat, flatten_kernels,
                                           sampling_results_list,
                                           batch_gt_instances)
        loss = dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox, loss_mask=loss_mask)
        return loss


class MaskFeatModule(BaseModule):
    """Mask feature head used in RTMDet-Ins.

    Args:
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels of the mask feature
             map branch.
        num_levels (int): The starting feature map level from RPN that
             will be used to predict the mask feature map.
        num_prototypes (int): Number of output channel of the mask feature
             map branch. This is the channel count of the mask
             feature map that to be dynamically convolved with the predicted
             kernel.
        stacked_convs (int): Number of convs in mask feature branch.
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Default: dict(type='ReLU', inplace=True)
        norm_cfg (dict): Config dict for normalization layer. Default: None.
    """

    def __init__(
        self,
        in_channels: int,
        feat_channels: int = 256,
        stacked_convs: int = 4,
        num_levels: int = 3,
        num_prototypes: int = 8,
        act_cfg: ConfigType = dict(type='ReLU', inplace=True),
        norm_cfg: ConfigType = dict(type='BN')
    ) -> None:
        super().__init__(init_cfg=None)
        self.num_levels = num_levels
        self.fusion_conv = nn.Conv2d(num_levels * in_channels, in_channels, 1)
        convs = []
        for i in range(stacked_convs):
            in_c = in_channels if i == 0 else feat_channels
            convs.append(
                ConvModule(
                    in_c,
                    feat_channels,
                    3,
                    padding=1,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg))
        self.stacked_convs = nn.Sequential(*convs)
        self.projection = nn.Conv2d(
            feat_channels, num_prototypes, kernel_size=1)

    def forward(self, features: Tuple[Tensor, ...]) -> Tensor:
        # multi-level feature fusion
        fusion_feats = [features[0]]
        size = features[0].shape[-2:]
        for i in range(1, self.num_levels):
            f = F.interpolate(features[i], size=size, mode='bilinear')
            fusion_feats.append(f)
        fusion_feats = torch.cat(fusion_feats, dim=1)
        fusion_feats = self.fusion_conv(fusion_feats)
        # pred mask feats
        mask_features = self.stacked_convs(fusion_feats)
        mask_features = self.projection(mask_features)
        return mask_features
