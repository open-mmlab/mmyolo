# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Sequence, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.models.utils import multi_apply
from mmdet.utils import (ConfigType, OptConfigType, OptInstanceList,
                         OptMultiConfig)
from mmengine import MessageHub
from mmengine.dist import get_dist_info
from mmengine.model import BaseModule, bias_init_with_prob
from mmengine.structures import InstanceData
from torch import Tensor

from mmyolo.registry import MODELS, TASK_UTILS
from ..utils import gt_instances_preprocess
from .yolov5_head import YOLOv5Head


@MODELS.register_module()
class YOLOv6HeadModule(BaseModule):
    """YOLOv6Head head module used in `YOLOv6.

    <https://arxiv.org/pdf/2209.02976>`_.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (Union[int, Sequence]): Number of channels in the input
            feature map.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        num_base_priors: (int): The number of priors (points) at a point
            on the feature grid.
        featmap_strides (Sequence[int]): Downsample factor of each feature map.
             Defaults to [8, 16, 32].
            None, otherwise False. Defaults to "auto".
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: Union[int, Sequence],
                 widen_factor: float = 1.0,
                 num_base_priors: int = 1,
                 reg_max=0,
                 featmap_strides: Sequence[int] = (8, 16, 32),
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)

        self.num_classes = num_classes
        self.featmap_strides = featmap_strides
        self.num_levels = len(self.featmap_strides)
        self.num_base_priors = num_base_priors
        self.reg_max = reg_max
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        if isinstance(in_channels, int):
            self.in_channels = [int(in_channels * widen_factor)
                                ] * self.num_levels
        else:
            self.in_channels = [int(i * widen_factor) for i in in_channels]

        self._init_layers()

    def _init_layers(self):
        """initialize conv layers in YOLOv6 head."""
        # Init decouple head
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        if self.reg_max > 1:
            proj = torch.arange(
                self.reg_max + self.num_base_priors, dtype=torch.float)
            self.register_buffer('proj', proj, persistent=False)

        for i in range(self.num_levels):
            self.stems.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.in_channels[i],
                    kernel_size=1,
                    stride=1,
                    padding=1 // 2,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
            self.cls_convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.in_channels[i],
                    kernel_size=3,
                    stride=1,
                    padding=3 // 2,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
            self.reg_convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.in_channels[i],
                    kernel_size=3,
                    stride=1,
                    padding=3 // 2,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=self.in_channels[i],
                    out_channels=self.num_base_priors * self.num_classes,
                    kernel_size=1))
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=self.in_channels[i],
                    out_channels=(self.num_base_priors + self.reg_max) * 4,
                    kernel_size=1))

    def init_weights(self):
        super().init_weights()
        bias_init = bias_init_with_prob(0.01)
        for conv in self.cls_preds:
            conv.bias.data.fill_(bias_init)
            conv.weight.data.fill_(0.)

        for conv in self.reg_preds:
            conv.bias.data.fill_(1.0)
            conv.weight.data.fill_(0.)

    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions.
        """
        assert len(x) == self.num_levels
        return multi_apply(self.forward_single, x, self.stems, self.cls_convs,
                           self.cls_preds, self.reg_convs, self.reg_preds)

    def forward_single(self, x: Tensor, stem: nn.Module, cls_conv: nn.Module,
                       cls_pred: nn.Module, reg_conv: nn.Module,
                       reg_pred: nn.Module) -> Tuple[Tensor, Tensor]:
        """Forward feature of a single scale level."""
        b, _, h, w = x.shape
        y = stem(x)
        cls_x = y
        reg_x = y
        cls_feat = cls_conv(cls_x)
        reg_feat = reg_conv(reg_x)

        cls_score = cls_pred(cls_feat)
        bbox_dist_preds = reg_pred(reg_feat)

        if self.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, self.reg_max + self.num_base_priors,
                 h * w]).permute(0, 3, 1, 2)

            # TODO: The get_flops script cannot handle the situation of
            #  matmul, and needs to be fixed later
            # bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj)
            bbox_preds = bbox_dist_preds.softmax(3).matmul(
                self.proj.view([-1, 1])).squeeze(-1)
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = bbox_dist_preds

        if self.training:
            return cls_score, bbox_preds, bbox_dist_preds
        else:
            return cls_score, bbox_preds


@MODELS.register_module()
class YOLOv6Head(YOLOv5Head):
    """YOLOv6Head head used in `YOLOv6 <https://arxiv.org/pdf/2209.02976>`_.

    Args:
        head_module(ConfigType): Base module used for YOLOv6Head
        prior_generator(dict): Points generator feature maps
            in 2D points-based detectors.
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
        loss_bbox (:obj:`ConfigDict` or dict): Config of localization loss.
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
                     offset=0.5,
                     strides=[8, 16, 32]),
                 bbox_coder: ConfigType = dict(type='DistancePointBBoxCoder'),
                 loss_cls: ConfigType = dict(
                     type='mmdet.VarifocalLoss',
                     use_sigmoid=True,
                     alpha=0.75,
                     gamma=2.0,
                     iou_weighted=True,
                     reduction='sum',
                     loss_weight=1.0),
                 loss_bbox: ConfigType = dict(
                     type='IoULoss',
                     iou_mode='giou',
                     bbox_format='xyxy',
                     reduction='mean',
                     loss_weight=2.5,
                     return_iou=False),
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
        # yolov6 doesn't need loss_obj
        self.loss_obj = None

    def special_init(self):
        """Since YOLO series algorithms will inherit from YOLOv5Head, but
        different algorithms have special initialization process.

        The special_init function is designed to deal with this situation.
        """
        if self.train_cfg:
            self.initial_epoch = self.train_cfg['initial_epoch']
            self.initial_assigner = TASK_UTILS.build(
                self.train_cfg.initial_assigner)
            self.assigner = TASK_UTILS.build(self.train_cfg.assigner)

            # Add common attributes to reduce calculation
            self.featmap_sizes_train = None
            self.num_level_priors = None
            self.flatten_priors_train = None
            self.stride_tensor = None

    def loss_by_feat(
            self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            bbox_dist_preds: Sequence[Tensor],
            batch_gt_instances: Sequence[InstanceData],
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
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
        Returns:
            dict[str, Tensor]: A dictionary of losses.
        """

        # get epoch information from message hub
        message_hub = MessageHub.get_current_instance()
        current_epoch = message_hub.get_info('epoch')

        num_imgs = len(batch_img_metas)
        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs

        current_featmap_sizes = [
            cls_score.shape[2:] for cls_score in cls_scores
        ]
        # If the shape does not equal, generate new one
        if current_featmap_sizes != self.featmap_sizes_train:
            self.featmap_sizes_train = current_featmap_sizes

            mlvl_priors_with_stride = self.prior_generator.grid_priors(
                self.featmap_sizes_train,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device,
                with_stride=True)

            self.num_level_priors = [len(n) for n in mlvl_priors_with_stride]
            self.flatten_priors_train = torch.cat(
                mlvl_priors_with_stride, dim=0)
            self.stride_tensor = self.flatten_priors_train[..., [2]]

        # gt info
        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels = gt_info[:, :, :1]
        gt_bboxes = gt_info[:, :, 1:]  # xyxy
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        # pred info
        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.num_classes)
            for cls_pred in cls_scores
        ]

        flatten_pred_bboxes = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]

        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_pred_bboxes = torch.cat(flatten_pred_bboxes, dim=1)
        flatten_pred_bboxes = self.bbox_coder.decode(
            self.flatten_priors_train[..., :2], flatten_pred_bboxes,
            self.stride_tensor[:, 0])
        pred_scores = torch.sigmoid(flatten_cls_preds)

        if current_epoch < self.initial_epoch:
            assigned_result = self.initial_assigner(
                flatten_pred_bboxes.detach(), self.flatten_priors_train,
                self.num_level_priors, gt_labels, gt_bboxes, pad_bbox_flag)
        else:
            assigned_result = self.assigner(flatten_pred_bboxes.detach(),
                                            pred_scores.detach(),
                                            self.flatten_priors_train,
                                            gt_labels, gt_bboxes,
                                            pad_bbox_flag)

        assigned_bboxes = assigned_result['assigned_bboxes']
        assigned_scores = assigned_result['assigned_scores']
        fg_mask_pre_prior = assigned_result['fg_mask_pre_prior']

        # cls loss
        with torch.cuda.amp.autocast(enabled=False):
            loss_cls = self.loss_cls(flatten_cls_preds, assigned_scores)

        # rescale bbox
        assigned_bboxes /= self.stride_tensor
        flatten_pred_bboxes /= self.stride_tensor

        # TODO: Add all_reduce makes training more stable
        assigned_scores_sum = assigned_scores.sum()
        if assigned_scores_sum > 0:
            loss_cls /= assigned_scores_sum

        # select positive samples mask
        num_pos = fg_mask_pre_prior.sum()
        if num_pos > 0:
            # when num_pos > 0, assigned_scores_sum will >0, so the loss_bbox
            # will not report an error
            # iou loss
            prior_bbox_mask = fg_mask_pre_prior.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(
                flatten_pred_bboxes, prior_bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(
                assigned_bboxes, prior_bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(
                assigned_scores.sum(-1), fg_mask_pre_prior).unsqueeze(-1)
            loss_bbox = self.loss_bbox(
                pred_bboxes_pos,
                assigned_bboxes_pos,
                weight=bbox_weight,
                avg_factor=assigned_scores_sum)
        else:
            loss_bbox = flatten_pred_bboxes.sum() * 0

        _, world_size = get_dist_info()
        return dict(
            loss_cls=loss_cls * world_size, loss_bbox=loss_bbox * world_size)
