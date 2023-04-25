# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Optional, Sequence, Tuple, Union

import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.models.utils import filter_scores_and_topk, multi_apply
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy
from mmdet.utils import ConfigType, OptInstanceList
from mmengine.config import ConfigDict
from mmengine.dist import get_dist_info
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor

from mmyolo.registry import MODELS
from ..utils import make_divisible
from .yolov5_head import YOLOv5Head, YOLOv5HeadModule


class ProtoModule(BaseModule):
    """Mask Proto module for segmentation models of YOLOv5.

    Args:
        in_channels (int): Number of channels in the input feature map.
        middle_channels (int): Number of channels in the middle feature map.
        mask_channels (int): Number of channels in the output mask feature
            map. This is the channel count of the mask.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to ``dict(type='BN', momentum=0.03, eps=0.001)``.
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Default: dict(type='SiLU', inplace=True).
    """

    def __init__(self,
                 *args,
                 in_channels: int = 32,
                 middle_channels: int = 256,
                 mask_channels: int = 32,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = ConvModule(
            in_channels,
            middle_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = ConvModule(
            middle_channels,
            middle_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv3 = ConvModule(
            middle_channels,
            mask_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv3(self.conv2(self.upsample(self.conv1(x))))


@MODELS.register_module()
class YOLOv5InsHeadModule(YOLOv5HeadModule):
    """Detection and Instance Segmentation Head of YOLOv5.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        mask_channels (int): Number of channels in the mask feature map.
            This is the channel count of the mask.
        proto_channels (int): Number of channels in the proto feature map.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to ``dict(type='BN', momentum=0.03, eps=0.001)``.
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Default: dict(type='SiLU', inplace=True).
    """

    def __init__(self,
                 *args,
                 num_classes: int,
                 mask_channels: int = 32,
                 proto_channels: int = 256,
                 widen_factor: float = 1.0,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 **kwargs):
        self.mask_channels = mask_channels
        self.num_out_attrib_with_proto = 5 + num_classes + mask_channels
        self.proto_channels = make_divisible(proto_channels, widen_factor)
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        super().__init__(
            *args,
            num_classes=num_classes,
            widen_factor=widen_factor,
            **kwargs)

    def _init_layers(self):
        """initialize conv layers in YOLOv5 Ins head."""
        self.convs_pred = nn.ModuleList()
        for i in range(self.num_levels):
            conv_pred = nn.Conv2d(
                self.in_channels[i],
                self.num_base_priors * self.num_out_attrib_with_proto, 1)
            self.convs_pred.append(conv_pred)

        self.proto_pred = ProtoModule(
            in_channels=self.in_channels[0],
            middle_channels=self.proto_channels,
            mask_channels=self.mask_channels,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions, objectnesses, and mask predictions.
        """
        assert len(x) == self.num_levels
        cls_scores, bbox_preds, objectnesses, coeff_preds = multi_apply(
            self.forward_single, x, self.convs_pred)
        mask_protos = self.proto_pred(x[0])
        return cls_scores, bbox_preds, objectnesses, coeff_preds, mask_protos

    def forward_single(
            self, x: Tensor,
            convs_pred: nn.Module) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward feature of a single scale level."""

        pred_map = convs_pred(x)
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
class YOLOv5InsHead(YOLOv5Head):
    """YOLOv5 Instance Segmentation and Detection head.

    Args:
        mask_overlap(bool): Defaults to True.
        loss_mask (:obj:`ConfigDict` or dict): Config of mask loss.
        loss_mask_weight (float): The weight of mask loss.
    """

    def __init__(self,
                 *args,
                 mask_overlap: bool = True,
                 loss_mask: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='none'),
                 loss_mask_weight=0.05,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_overlap = mask_overlap
        self.loss_mask: nn.Module = MODELS.build(loss_mask)
        self.loss_mask_weight = loss_mask_weight

    def loss(self, x: Tuple[Tensor], batch_data_samples: Union[list,
                                                               dict]) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`], dict): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """

        if isinstance(batch_data_samples, list):
            # TODO: support non-fast version ins segmention
            raise NotImplementedError
        else:
            outs = self(x)
            # Fast version
            loss_inputs = outs + (batch_data_samples['bboxes_labels'],
                                  batch_data_samples['masks'],
                                  batch_data_samples['img_metas'])
            losses = self.loss_by_feat(*loss_inputs)

        return losses

    def loss_by_feat(
            self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            objectnesses: Sequence[Tensor],
            coeff_preds: Sequence[Tensor],
            proto_preds: Tensor,
            batch_gt_instances: Sequence[InstanceData],
            batch_gt_masks: Sequence[Tensor],
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (Sequence[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (Sequence[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            objectnesses (Sequence[Tensor]): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            coeff_preds (Sequence[Tensor]): Mask coefficient for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * mask_channels.
            proto_preds (Tensor): Mask prototype features extracted from the
                mask head, has shape (batch_size, mask_channels, H, W).
            batch_gt_instances (Sequence[InstanceData]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_gt_masks (Sequence[Tensor]): Batch of gt_mask.
            batch_img_metas (Sequence[dict]): Meta information of each image,
                e.g., image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
        Returns:
            dict[str, Tensor]: A dictionary of losses.
        """
        # 1. Convert gt to norm format
        batch_targets_normed = self._convert_gt_to_norm_format(
            batch_gt_instances, batch_img_metas)

        device = cls_scores[0].device
        loss_cls = torch.zeros(1, device=device)
        loss_box = torch.zeros(1, device=device)
        loss_obj = torch.zeros(1, device=device)
        loss_mask = torch.zeros(1, device=device)
        scaled_factor = torch.ones(8, device=device)

        for i in range(self.num_levels):
            batch_size, _, h, w = bbox_preds[i].shape
            target_obj = torch.zeros_like(objectnesses[i])

            # empty gt bboxes
            if batch_targets_normed.shape[1] == 0:
                loss_box += bbox_preds[i].sum() * 0
                loss_cls += cls_scores[i].sum() * 0
                loss_obj += self.loss_obj(
                    objectnesses[i], target_obj) * self.obj_level_weights[i]
                loss_mask += coeff_preds[i].sum() * 0
                continue

            priors_base_sizes_i = self.priors_base_sizes[i]
            # feature map scale whwh
            scaled_factor[2:6] = torch.tensor(
                bbox_preds[i].shape)[[3, 2, 3, 2]]
            # Scale batch_targets from range 0-1 to range 0-features_maps size.
            # (num_base_priors, num_bboxes, 8)
            batch_targets_scaled = batch_targets_normed * scaled_factor

            # 2. Shape match
            wh_ratio = batch_targets_scaled[...,
                                            4:6] / priors_base_sizes_i[:, None]
            match_inds = torch.max(
                wh_ratio, 1 / wh_ratio).max(2)[0] < self.prior_match_thr
            batch_targets_scaled = batch_targets_scaled[match_inds]

            # no gt bbox matches anchor
            if batch_targets_scaled.shape[0] == 0:
                loss_box += bbox_preds[i].sum() * 0
                loss_cls += cls_scores[i].sum() * 0
                loss_obj += self.loss_obj(
                    objectnesses[i], target_obj) * self.obj_level_weights[i]
                loss_mask += coeff_preds[i].sum() * 0
                continue

            # 3. Positive samples with additional neighbors

            # check the left, up, right, bottom sides of the
            # targets grid, and determine whether assigned
            # them as positive samples as well.
            batch_targets_cxcy = batch_targets_scaled[:, 2:4]
            grid_xy = scaled_factor[[2, 3]] - batch_targets_cxcy
            left, up = ((batch_targets_cxcy % 1 < self.near_neighbor_thr) &
                        (batch_targets_cxcy > 1)).T
            right, bottom = ((grid_xy % 1 < self.near_neighbor_thr) &
                             (grid_xy > 1)).T
            offset_inds = torch.stack(
                (torch.ones_like(left), left, up, right, bottom))

            batch_targets_scaled = batch_targets_scaled.repeat(
                (5, 1, 1))[offset_inds]
            retained_offsets = self.grid_offset.repeat(1, offset_inds.shape[1],
                                                       1)[offset_inds]

            # prepare pred results and positive sample indexes to
            # calculate class loss and bbox lo
            _chunk_targets = batch_targets_scaled.chunk(4, 1)
            img_class_inds, grid_xy, grid_wh,\
                priors_targets_inds = _chunk_targets
            (priors_inds, targets_inds) = priors_targets_inds.long().T
            (img_inds, class_inds) = img_class_inds.long().T

            grid_xy_long = (grid_xy -
                            retained_offsets * self.near_neighbor_thr).long()
            grid_x_inds, grid_y_inds = grid_xy_long.T
            bboxes_targets = torch.cat((grid_xy - grid_xy_long, grid_wh), 1)

            # 4. Calculate loss
            # bbox loss
            retained_bbox_pred = bbox_preds[i].reshape(
                batch_size, self.num_base_priors, -1, h,
                w)[img_inds, priors_inds, :, grid_y_inds, grid_x_inds]
            priors_base_sizes_i = priors_base_sizes_i[priors_inds]
            decoded_bbox_pred = self._decode_bbox_to_xywh(
                retained_bbox_pred, priors_base_sizes_i)
            loss_box_i, iou = self.loss_bbox(decoded_bbox_pred, bboxes_targets)
            loss_box += loss_box_i

            # obj loss
            iou = iou.detach().clamp(0)
            target_obj[img_inds, priors_inds, grid_y_inds,
                       grid_x_inds] = iou.type(target_obj.dtype)
            loss_obj += self.loss_obj(objectnesses[i],
                                      target_obj) * self.obj_level_weights[i]

            # cls loss
            if self.num_classes > 1:
                pred_cls_scores = cls_scores[i].reshape(
                    batch_size, self.num_base_priors, -1, h,
                    w)[img_inds, priors_inds, :, grid_y_inds, grid_x_inds]

                target_class = torch.full_like(pred_cls_scores, 0.)
                target_class[range(batch_targets_scaled.shape[0]),
                             class_inds] = 1.
                loss_cls += self.loss_cls(pred_cls_scores, target_class)
            else:
                loss_cls += cls_scores[i].sum() * 0

            # mask regression
            retained_coeff_preds = coeff_preds[i].reshape(
                batch_size, self.num_base_priors, -1, h,
                w)[img_inds, priors_inds, :, grid_y_inds, grid_x_inds]

            _, c, mask_h, mask_w = proto_preds.shape
            if batch_gt_masks.shape[-2:] != (mask_h, mask_w):
                batch_gt_masks = F.interpolate(
                    batch_gt_masks[None], (mask_h, mask_w), mode='nearest')[0]

            xywh_normed = batch_targets_scaled[:, 2:6] / scaled_factor[2:6]
            area_normed = xywh_normed[:, 2:].prod(1)
            xywh_scaled = xywh_normed * torch.tensor(
                proto_preds.shape, device=device)[[3, 2, 3, 2]]
            xyxy_scaled = bbox_cxcywh_to_xyxy(xywh_scaled)

            for bs in range(batch_size):
                match_inds = (img_inds == bs)  # matching index
                if not match_inds.any():
                    continue

                if self.mask_overlap:
                    mask_gti = torch.where(
                        batch_gt_masks[bs][None] ==
                        targets_inds[match_inds].view(-1, 1, 1), 1.0, 0.0)
                else:
                    mask_gti = batch_gt_masks[targets_inds][match_inds]

                mask_preds = (retained_coeff_preds[match_inds]
                              @ proto_preds[bs].view(c, -1)).view(
                                  -1, mask_h, mask_w)
                loss_mask_full = self.loss_mask(mask_preds, mask_gti)
                loss_mask += (
                    self.crop_mask(loss_mask_full[None],
                                   xyxy_scaled[match_inds]).mean(dim=(2, 3)) /
                    area_normed[match_inds]).mean()

        _, world_size = get_dist_info()
        return dict(
            loss_cls=loss_cls * batch_size * world_size,
            loss_obj=loss_obj * batch_size * world_size,
            loss_bbox=loss_box * batch_size * world_size,
            loss_mask=loss_mask * self.loss_mask_weight * world_size)

    def _convert_gt_to_norm_format(self,
                                   batch_gt_instances: Sequence[InstanceData],
                                   batch_img_metas: Sequence[dict]) -> Tensor:
        """Add target_inds for instance segmentation."""
        batch_targets_normed = super()._convert_gt_to_norm_format(
            batch_gt_instances, batch_img_metas)

        if self.mask_overlap:
            batch_size = len(batch_img_metas)
            target_inds = []
            for i in range(batch_size):
                # find number of targets of each image
                num_gts = (batch_gt_instances[:, 0] == i).sum()
                # (num_anchor, num_gts)
                target_inds.append(
                    torch.arange(num_gts, device=batch_gt_instances.device).
                    float().view(1, num_gts).repeat(self.num_base_priors, 1) +
                    1)
            target_inds = torch.cat(target_inds, 1)
        else:
            num_gts = batch_gt_instances.shape[0]
            target_inds = torch.arange(
                num_gts, device=batch_gt_instances.device).float().view(
                    1, num_gts).repeat(self.num_base_priors, 1)
        batch_targets_normed = torch.cat(
            [batch_targets_normed, target_inds[..., None]], 2)
        return batch_targets_normed

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        objectnesses: Optional[List[Tensor]] = None,
                        coeff_preds: Optional[List[Tensor]] = None,
                        proto_preds: Optional[Tensor] = None,
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
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            coeff_preds (list[Tensor]): Mask coefficients predictions
                for all scale levels, each is a 4D-tensor, has shape
                (batch_size, mask_channels, H, W).
            proto_preds (Tensor): Mask prototype features extracted from the
                mask head, has shape (batch_size, mask_channels, H, W).
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
            coeff_pred.permute(0, 2, 3,
                               1).reshape(num_imgs, -1,
                                          self.head_module.mask_channels)
            for coeff_pred in coeff_preds
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_decoded_bboxes = self.bbox_coder.decode(
            flatten_priors.unsqueeze(0), flatten_bbox_preds, flatten_stride)

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
        for (bboxes, scores, objectness, coeffs, mask_proto,
             img_meta) in zip(flatten_decoded_bboxes, flatten_cls_scores,
                              flatten_objectness, flatten_coeff_preds,
                              proto_preds, batch_img_metas):
            ori_shape = img_meta['ori_shape']
            batch_input_shape = img_meta['batch_input_shape']
            input_shape_h, input_shape_w = batch_input_shape
            if 'pad_param' in img_meta:
                pad_param = img_meta['pad_param']
                input_shape_withoutpad = (input_shape_h - pad_param[0] -
                                          pad_param[1], input_shape_w -
                                          pad_param[2] - pad_param[3])
            else:
                pad_param = None
                input_shape_withoutpad = batch_input_shape
            scale_factor = (input_shape_withoutpad[1] / ori_shape[1],
                            input_shape_withoutpad[0] / ori_shape[0])

            score_thr = cfg.get('score_thr', -1)
            # yolox_style does not require the following operations
            if objectness is not None and score_thr > 0 and not cfg.get(
                    'yolox_style', False):
                conf_inds = objectness > score_thr
                bboxes = bboxes[conf_inds, :]
                scores = scores[conf_inds, :]
                objectness = objectness[conf_inds]
                coeffs = coeffs[conf_inds]

            if objectness is not None:
                # conf = obj_conf * cls_conf
                scores *= objectness[:, None]
                # NOTE: Important
                coeffs *= objectness[:, None]

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
                    results=dict(labels=labels[:, 0], coeffs=coeffs))
                labels = results['labels']
                coeffs = results['coeffs']
            else:
                out = filter_scores_and_topk(
                    scores, score_thr, nms_pre, results=dict(coeffs=coeffs))
                scores, labels, keep_idxs, filtered_results = out
                coeffs = filtered_results['coeffs']

            results = InstanceData(
                scores=scores,
                labels=labels,
                bboxes=bboxes[keep_idxs],
                coeffs=coeffs)

            if cfg.get('yolox_style', False):
                # do not need max_per_img
                cfg.max_per_img = len(results)

            results = self._bbox_post_process(
                results=results,
                cfg=cfg,
                rescale=False,
                with_nms=with_nms,
                img_meta=img_meta)

            if len(results.bboxes):
                masks = self.process_mask(mask_proto, results.coeffs,
                                          results.bboxes,
                                          (input_shape_h, input_shape_w), True)
                if rescale:
                    if pad_param is not None:
                        # bbox minus pad param
                        top_pad, _, left_pad, _ = pad_param
                        results.bboxes -= results.bboxes.new_tensor(
                            [left_pad, top_pad, left_pad, top_pad])
                        # mask crop pad param
                        top, left = int(top_pad), int(left_pad)
                        bottom, right = int(input_shape_h -
                                            top_pad), int(input_shape_w -
                                                          left_pad)
                        masks = masks[:, :, top:bottom, left:right]
                    results.bboxes /= results.bboxes.new_tensor(
                        scale_factor).repeat((1, 2))

                    fast_test = cfg.get('fast_test', False)
                    if fast_test:
                        masks = F.interpolate(
                            masks,
                            size=ori_shape,
                            mode='bilinear',
                            align_corners=False)
                        masks = masks.squeeze(0)
                        masks = masks > cfg.mask_thr_binary
                    else:
                        masks.gt_(cfg.mask_thr_binary)
                        masks = torch.as_tensor(masks, dtype=torch.uint8)
                        masks = masks[0].permute(1, 2,
                                                 0).contiguous().cpu().numpy()
                        masks = mmcv.imresize(masks,
                                              (ori_shape[1], ori_shape[0]))

                        if len(masks.shape) == 2:
                            masks = masks[:, :, None]
                        masks = torch.from_numpy(masks).permute(2, 0, 1)

                results.bboxes[:, 0::2].clamp_(0, ori_shape[1])
                results.bboxes[:, 1::2].clamp_(0, ori_shape[0])

                results.masks = masks.bool()
                results_list.append(results)
            else:
                h, w = ori_shape[:2] if rescale else img_meta['img_shape'][:2]
                results.masks = torch.zeros(
                    size=(0, h, w), dtype=torch.bool, device=bboxes.device)
                results_list.append(results)
        return results_list

    def process_mask(self,
                     mask_proto: Tensor,
                     mask_coeff_pred: Tensor,
                     bboxes: Tensor,
                     shape: Tuple[int, int],
                     upsample: bool = False) -> Tensor:
        """Generate mask logits results.

        Args:
            mask_proto (Tensor): Mask prototype features.
                Has shape (num_instance, mask_channels).
            mask_coeff_pred (Tensor): Mask coefficients prediction for
                single image. Has shape (mask_channels, H, W)
            bboxes (Tensor): Tensor of the bbox. Has shape (num_instance, 4).
            shape (Tuple): Batch input shape of image.
            upsample (bool): Whether upsample masks results to batch input
                shape. Default to False.
        Return:
            Tensor: Instance segmentation masks for each instance.
                Has shape (num_instance, H, W).
        """
        c, mh, mw = mask_proto.shape  # CHW
        masks = (
            mask_coeff_pred @ mask_proto.float().view(c, -1)).sigmoid().view(
                -1, mh, mw)[None]
        if upsample:
            masks = F.interpolate(
                masks, shape, mode='bilinear', align_corners=False)  # 1CHW
        masks = self.crop_mask(masks, bboxes)
        return masks

    def crop_mask(self, masks: Tensor, boxes: Tensor) -> Tensor:
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
