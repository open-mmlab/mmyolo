# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy
from mmdet.models.utils import filter_scores_and_topk, multi_apply
from mmdet.utils import (ConfigType, OptConfigType, OptInstanceList,
                         OptMultiConfig)
from mmengine.config import ConfigDict
from mmengine.dist import get_dist_info
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
class YOLOv5InsHeadModule(YOLOv5HeadModule):

    def __init__(self,
                 *args,
                 num_classes,
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
class YOLOv5InsHead(YOLOv5Head):

    def __init__(self,
                 *args,
                 overlap=True,
                 loss_mask=dict(
                    type='mmdet.CrossEntropyLoss',
                    use_sigmoid=True,
                    loss_weight=0.05,
                    reduction='none'),
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.num_protos = self.head_module.num_protos
        self.overlap = overlap
        self.loss_mask: nn.Module = MODELS.build(loss_mask)
    
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
            losses = super().loss(x, batch_data_samples)
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
            segm_preds: Tensor,
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
            batch_gt_instances (Sequence[InstanceData]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (Sequence[dict]): Meta information of each image,
                e.g., image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
        Returns:
            dict[str, Tensor]: A dictionary of losses.
        """
        if self.ignore_iof_thr != -1:
            # TODO: Support fast version
            # convert ignore gt
            batch_target_ignore_list = []
            for i, gt_instances_ignore in enumerate(batch_gt_instances_ignore):
                bboxes = gt_instances_ignore.bboxes
                labels = gt_instances_ignore.labels
                index = bboxes.new_full((len(bboxes), 1), i)
                # (batch_idx, label, bboxes)
                target = torch.cat((index, labels[:, None].float(), bboxes),
                                   dim=1)
                batch_target_ignore_list.append(target)

            # (num_bboxes, 6)
            batch_gt_targets_ignore = torch.cat(
                batch_target_ignore_list, dim=0)
            if batch_gt_targets_ignore.shape[0] != 0:
                # Consider regions with ignore in annotations
                return self._loss_by_feat_with_ignore(
                    cls_scores,
                    bbox_preds,
                    objectnesses,
                    batch_gt_instances=batch_gt_instances,
                    batch_img_metas=batch_img_metas,
                    batch_gt_instances_ignore=batch_gt_targets_ignore)

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
            img_class_inds, grid_xy, grid_wh, priors_targets_inds = _chunk_targets
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

            _, c , mask_h, mask_w = segm_preds.shape
            if batch_gt_masks.shape[-2:] != (mask_h, mask_w):
                batch_gt_masks = F.interpolate(
                    batch_gt_masks[None], (mask_h, mask_w), mode='nearest')[0]
            
            nxywh = batch_targets_scaled[:, 2:6] / scaled_factor[2:6]
            marea = nxywh[:, 2:].prod(1)
            mxyxy = bbox_cxcywh_to_xyxy(nxywh * scaled_factor[2:6] * 2) # TODO

            for bs in range(batch_size):
                j = (img_inds == bs)  # matching index
                if self.overlap:
                    mask_gti = torch.where(batch_gt_masks[bs][None] == targets_inds[j].view(-1, 1, 1), 1.0, 0.0)
                else:
                    mask_gti = batch_gt_masks[targets_inds][j]
                
                mask_preds = (retained_coeff_preds[j] @ segm_preds[bs].float().view(c, -1)).view(
                    -1, mask_h, mask_w)
                loss_mask_full = self.loss_mask(mask_preds, mask_gti)
                loss_mask += (crop_mask(loss_mask_full, mxyxy[j]).mean(dim=(1, 2)) / marea[j]).mean()

        _, world_size = get_dist_info()
        return dict(
            loss_cls=loss_cls * batch_size * world_size,
            loss_obj=loss_obj * batch_size * world_size,
            loss_bbox=loss_box * batch_size * world_size,
            loss_mask=loss_mask * world_size)

    def _convert_gt_to_norm_format(self,
                                   batch_gt_instances: Sequence[InstanceData],
                                   batch_img_metas: Sequence[dict]) -> Tensor:
        batch_targets_normed = super()._convert_gt_to_norm_format(
            batch_gt_instances, batch_img_metas)

        if self.overlap:
            batch_size = len(batch_img_metas)
            target_inds = []
            for i in range(batch_size):
                # find number of targets of each image
                num = (batch_gt_instances[:, 0] == i).sum() 
                # (num_anchor, num_gts)
                target_inds.append(
                    torch.arange(num, device=batch_gt_instances.device).float().view(
                    1, num).repeat(self.num_base_priors, 1) + 1)
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