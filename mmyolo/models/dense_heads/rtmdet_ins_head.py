# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, is_norm
from mmcv.ops import batched_nms
from mmdet.models.utils import filter_scores_and_topk
from mmdet.structures.bbox import get_box_tensor, get_box_wh, scale_boxes
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, OptMultiConfig)
from mmengine import ConfigDict
from mmengine.model import (BaseModule, bias_init_with_prob, constant_init,
                            normal_init)
from mmengine.structures import InstanceData
from torch import Tensor

from mmyolo.registry import MODELS
from .rtmdet_head import RTMDetHead, RTMDetSepBNHeadModule


class MaskFeatModule(BaseModule):
    """Mask feature head used in RTMDet-Ins. Copy from mmdet.

    Args:
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels of the mask feature
             map branch.
        stacked_convs (int): Number of convs in mask feature branch.
        num_levels (int): The starting feature map level from RPN that
             will be used to predict the mask feature map.
        num_prototypes (int): Number of output channel of the mask feature
             map branch. This is the channel count of the mask
             feature map that to be dynamically convolved with the predicted
             kernel.
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


@MODELS.register_module()
class RTMDetInsSepBNHeadModule(RTMDetSepBNHeadModule):
    """Detection and Instance Segmentation Head of RTMDet.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        num_prototypes (int): Number of mask prototype features extracted
            from the mask head. Defaults to 8.
        dyconv_channels (int): Channel of the dynamic conv layers.
            Defaults to 8.
        num_dyconvs (int): Number of the dynamic convolution layers.
            Defaults to 3.
        use_sigmoid_cls (bool): Use sigmoid for class prediction.
            Defaults to True.
    """

    def __init__(self,
                 num_classes: int,
                 *args,
                 num_prototypes: int = 8,
                 dyconv_channels: int = 8,
                 num_dyconvs: int = 3,
                 use_sigmoid_cls: bool = True,
                 **kwargs):
        self.num_prototypes = num_prototypes
        self.num_dyconvs = num_dyconvs
        self.dyconv_channels = dyconv_channels
        self.use_sigmoid_cls = use_sigmoid_cls
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        super().__init__(num_classes=num_classes, *args, **kwargs)

    def _init_layers(self):
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
                    self.num_base_priors * self.cls_out_channels,
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
            - mask_feat (Tensor): Mask prototype features.
                Has shape (batch_size, num_prototypes, H, W).
        """
        mask_feat = self.mask_head(feats)

        cls_scores = []
        bbox_preds = []
        kernel_preds = []
        for idx, (x, stride) in enumerate(zip(feats, self.featmap_strides)):
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
            reg_dist = self.rtm_reg[idx](reg_feat)

            cls_scores.append(cls_score)
            bbox_preds.append(reg_dist)
            kernel_preds.append(kernel_pred)
        return tuple(cls_scores), tuple(bbox_preds), tuple(
            kernel_preds), mask_feat


@MODELS.register_module()
class RTMDetInsSepBNHead(RTMDetHead):
    """RTMDet Instance Segmentation head.

    Args:
        head_module(ConfigType): Base module used for RTMDetInsSepBNHead
        prior_generator: Points generator feature maps in
            2D points-based detectors.
        bbox_coder (:obj:`ConfigDict` or dict): Config of bbox coder.
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
        loss_bbox (:obj:`ConfigDict` or dict): Config of localization loss.
        loss_mask (:obj:`ConfigDict` or dict): Config of mask loss.
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config of
            anchor head. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config of
            anchor head. Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 head_module: ConfigType,
                 prior_generator: ConfigType = dict(
                     type='mmdet.MlvlPointGenerator',
                     offset=0,
                     strides=[8, 16, 32]),
                 bbox_coder: ConfigType = dict(type='DistancePointBBoxCoder'),
                 loss_cls: ConfigType = dict(
                     type='mmdet.QualityFocalLoss',
                     use_sigmoid=True,
                     beta=2.0,
                     loss_weight=1.0),
                 loss_bbox: ConfigType = dict(
                     type='mmdet.GIoULoss', loss_weight=2.0),
                 loss_mask=dict(
                     type='mmdet.DiceLoss',
                     loss_weight=2.0,
                     eps=5e-6,
                     reduction='mean'),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):

        super().__init__(
            head_module=head_module,
            prior_generator=prior_generator,
            bbox_coder=bbox_coder,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if isinstance(self.head_module, RTMDetInsSepBNHeadModule):
            assert self.use_sigmoid_cls == self.head_module.use_sigmoid_cls
        self.loss_mask = MODELS.build(loss_mask)

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        kernel_preds: List[Tensor],
                        mask_feats: Tensor,
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
            kernel_preds (list[Tensor]): Kernel predictions of dynamic
                convs for all scale levels, each is a 4D-tensor, has shape
                (batch_size, num_params, H, W).
            mask_feats (Tensor): Mask prototype features extracted from the
                mask head, has shape (batch_size, num_prototypes, H, W).
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
        flatten_kernel_preds = [
            kernel_pred.permute(0, 2, 3,
                                1).reshape(num_imgs, -1,
                                           self.head_module.num_gen_params)
            for kernel_pred in kernel_preds
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_decoded_bboxes = self.bbox_coder.decode(
            flatten_priors[..., :2].unsqueeze(0), flatten_bbox_preds,
            flatten_stride)

        flatten_kernel_preds = torch.cat(flatten_kernel_preds, dim=1)

        results_list = []
        for (bboxes, scores, kernel_pred, mask_feat,
             img_meta) in zip(flatten_decoded_bboxes, flatten_cls_scores,
                              flatten_kernel_preds, mask_feats,
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
                        labels=labels[:, 0],
                        kernel_pred=kernel_pred,
                        priors=flatten_priors))
                labels = results['labels']
                kernel_pred = results['kernel_pred']
                priors = results['priors']
            else:
                out = filter_scores_and_topk(
                    scores,
                    score_thr,
                    nms_pre,
                    results=dict(
                        kernel_pred=kernel_pred, priors=flatten_priors))
                scores, labels, keep_idxs, filtered_results = out
                kernel_pred = filtered_results['kernel_pred']
                priors = filtered_results['priors']

            results = InstanceData(
                scores=scores,
                labels=labels,
                bboxes=bboxes[keep_idxs],
                kernels=kernel_pred,
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
                mask_feat=mask_feat,
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
                                                     results.kernels,
                                                     results.priors)

            stride = self.prior_generator.strides[0][0]
            mask_logits = F.interpolate(
                mask_logits.unsqueeze(0), scale_factor=stride, mode='bilinear')
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

            masks = mask_logits.sigmoid().squeeze(0)
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

    def _mask_predict_by_feat(self, mask_feat: Tensor, kernels: Tensor,
                              priors: Tensor) -> Tensor:
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
        num_inst = kernels.shape[0]
        h, w = mask_feat.size()[-2:]
        if num_inst < 1:
            return torch.empty(
                size=(num_inst, h, w),
                dtype=mask_feat.dtype,
                device=mask_feat.device)
        if len(mask_feat.shape) < 4:
            mask_feat.unsqueeze(0)

        coord = self.prior_generator.single_level_grid_priors(
            (h, w), level_idx=0, device=mask_feat.device).reshape(1, -1, 2)
        num_inst = priors.shape[0]
        points = priors[:, :2].reshape(-1, 1, 2)
        strides = priors[:, 2:].reshape(-1, 1, 2)
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

    def parse_dynamic_params(self, flatten_kernels: Tensor) -> tuple:
        """split kernel head prediction to conv weight and bias."""
        n_inst = flatten_kernels.size(0)
        n_layers = len(self.head_module.weight_nums)
        params_splits = list(
            torch.split_with_sizes(
                flatten_kernels,
                self.head_module.weight_nums + self.head_module.bias_nums,
                dim=1))
        weight_splits = params_splits[:n_layers]
        bias_splits = params_splits[n_layers:]
        for i in range(n_layers):
            if i < n_layers - 1:
                weight_splits[i] = weight_splits[i].reshape(
                    n_inst * self.head_module.dyconv_channels, -1, 1, 1)
                bias_splits[i] = bias_splits[i].reshape(
                    n_inst * self.head_module.dyconv_channels)
            else:
                weight_splits[i] = weight_splits[i].reshape(n_inst, -1, 1, 1)
                bias_splits[i] = bias_splits[i].reshape(n_inst)

        return weight_splits, bias_splits

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        raise NotImplementedError
