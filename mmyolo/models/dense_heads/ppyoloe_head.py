# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.utils import multi_apply
from mmdet.utils import (ConfigType, OptConfigType, OptInstanceList,
                         OptMultiConfig, reduce_mean)
from mmengine import MessageHub
from mmengine.dist import get_dist_info
from mmengine.model import BaseModule, bias_init_with_prob
from mmengine.structures import InstanceData
from torch import Tensor

from mmyolo.registry import MODELS, TASK_UTILS
from ..layers.yolo_bricks import PPYOLOESELayer
from .yolov6_head import YOLOv6Head


@MODELS.register_module()
class PPYOLOEHeadModule(BaseModule):
    """PPYOLOEHead head module used in `PPYOLOE`

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Default: 1.0.
        num_base_priors:int: The number of priors (points) at a point
            on the feature grid.
        featmap_strides (Sequence[int]): Downsample factor of each feature map.
             Defaults to (8, 16, 32).
        reg_max (int): TOOD reg_max param.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: Union[int, Sequence],
                 widen_factor: float = 1.0,
                 num_base_priors: int = 1,
                 featmap_strides: Sequence[int] = (8, 16, 32),
                 reg_max: int = 16,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.1, eps=1e-5),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)

        self.num_classes = num_classes
        self.featmap_strides = featmap_strides
        self.num_levels = len(self.featmap_strides)
        self.num_base_priors = num_base_priors
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.reg_max = reg_max

        if isinstance(in_channels, int):
            self.in_channels = [int(in_channels * widen_factor)
                                ] * self.num_levels
        else:
            self.in_channels = [int(i * widen_factor) for i in in_channels]

        self._init_layers()

    def init_weights(self, prior_prob=0.01):
        """Initialize the weight and bias of PPYOLOE head."""
        super().init_weights()
        for conv in self.cls_preds:
            conv.bias.data.fill_(bias_init_with_prob(prior_prob))
            conv.weight.data.fill_(0.)

        for conv in self.reg_preds:
            conv.bias.data.fill_(1.0)
            conv.weight.data.fill_(0.)

    def _init_layers(self):
        """initialize conv layers in PPYOLOE head."""
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.cls_stems = nn.ModuleList()
        self.reg_stems = nn.ModuleList()

        for in_channel in self.in_channels:
            self.cls_stems.append(
                PPYOLOESELayer(
                    in_channel, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))
            self.reg_stems.append(
                PPYOLOESELayer(
                    in_channel, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))

        for in_channel in self.in_channels:
            self.cls_preds.append(
                nn.Conv2d(in_channel, self.num_classes, 3, padding=1))
            self.reg_preds.append(
                nn.Conv2d(in_channel, 4 * (self.reg_max + 1), 3, padding=1))

        self.proj_conv = nn.Conv2d(self.reg_max + 1, 1, 1, bias=False)
        self.proj = nn.Parameter(
            torch.linspace(0, self.reg_max, self.reg_max + 1),
            requires_grad=False)
        self.proj_conv.weight = nn.Parameter(
            self.proj.view([1, self.reg_max + 1, 1, 1]).clone().detach(),
            requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions.
        """
        assert len(x) == self.num_levels

        return multi_apply(self.forward_single, x, self.cls_stems,
                           self.cls_preds, self.reg_stems, self.reg_preds)

    def forward_single(self, x: Tensor, cls_stem: nn.ModuleList,
                       cls_pred: nn.ModuleList, reg_stem: nn.ModuleList,
                       reg_pred: nn.ModuleList) -> Tensor:
        """Forward feature of a single scale level."""
        b, _, h, w = x.shape
        hw = h * w
        avg_feat = F.adaptive_avg_pool2d(x, (1, 1))
        cls_logit = cls_pred(cls_stem(x, avg_feat) + x)
        reg_dist = reg_pred(reg_stem(x, avg_feat))
        reg_dist = reg_dist.reshape([-1, 4, self.reg_max + 1,
                                     hw]).permute(0, 2, 3, 1)
        # TODO: 命名要改
        reg_dist_proj = self.proj_conv(F.softmax(reg_dist, dim=1))

        if self.training:
            return cls_logit, reg_dist_proj, reg_dist
        else:
            return cls_logit, reg_dist_proj


@MODELS.register_module()
class PPYOLOEHead(YOLOv6Head):
    """PPYOLOEHead head used in `PPYOLOE`.

    Args:
        head_module(nn.Module): Base module used for YOLOv5Head
        prior_generator(dict): Points generator feature maps in
            2D points-based detectors.
        bbox_coder (:obj:`ConfigDict` or dict): Config of bbox coder.
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
        loss_bbox (:obj:`ConfigDict` or dict): Config of localization loss.
        loss_obj (:obj:`ConfigDict` or dict): Config of objectness loss.
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config of
            anchor head. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config of
            anchor head. Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 head_module: nn.Module,
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
                     type='mmdet.GIoULoss', reduction='sum', loss_weight=2.5),
                 loss_dfl: ConfigType = dict(
                     type='DfLoss', reduction='mean', loss_weight=0.5),
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
        loss_dfl['reg_max'] = self.head_module.reg_max
        self.loss_dfl = MODELS.build(loss_dfl)
        # ppyoloe doesn't need loss_obj
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
            self.featmap_sizes = None
            self.mlvl_priors = None
            self.num_level_priors = None
            self.flatten_priors = None
            self.stride_tensor = None

    # TODO: 命名要改
    def loss_by_feat(
            self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            bbox_preds_org: Sequence[Tensor],
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
        if current_featmap_sizes != self.featmap_sizes:
            self.featmap_sizes = current_featmap_sizes

            self.mlvl_priors = self.prior_generator.grid_priors(
                self.featmap_sizes,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device,
                with_stride=True)

            self.num_level_priors = [len(n) for n in self.mlvl_priors]
            self.flatten_priors = torch.cat(self.mlvl_priors, dim=0)
            self.stride_tensor = self.flatten_priors[..., [2]]

        # gt info
        gt_info = self.gt_instances_preprocess(batch_gt_instances, num_imgs)
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

        # 原尺寸bs, 4, 17, h*w
        # 目标尺寸 [32, 8400, 68]
        pred_dists = [
            bbox_pred_org.permute(0, 3, 1, 2).reshape(
                num_imgs, -1, (self.head_module.reg_max + 1) * 4)
            for bbox_pred_org in bbox_preds_org
        ]

        dist_preds = torch.cat(pred_dists, dim=1)
        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_pred_bboxes = torch.cat(flatten_pred_bboxes, dim=1)
        flatten_pred_bboxes = self.bbox_coder.decode(
            self.flatten_priors[..., :2], flatten_pred_bboxes,
            self.flatten_priors[..., 2])
        pred_scores = torch.sigmoid(flatten_cls_preds)

        if current_epoch < self.initial_epoch:
            assigned_result = self.initial_assigner(
                flatten_pred_bboxes.detach(), self.flatten_priors,
                self.num_level_priors, gt_labels, gt_bboxes, pad_bbox_flag)
        else:
            assigned_result = self.assigner(flatten_pred_bboxes.detach(),
                                            pred_scores.detach(),
                                            self.flatten_priors, gt_labels,
                                            gt_bboxes, pad_bbox_flag)

        # [32, 8400, 4]
        assigned_bboxes = assigned_result['assigned_bboxes']
        # [32, 8400, 80]
        assigned_scores = assigned_result['assigned_scores']
        # [32, 8400]
        fg_mask_pre_prior = assigned_result['fg_mask_pre_prior']

        # cls loss
        with torch.cuda.amp.autocast(enabled=False):
            loss_cls = self.loss_cls(flatten_cls_preds, assigned_scores)

        # rescale bbox
        assigned_bboxes /= self.stride_tensor
        flatten_pred_bboxes /= self.stride_tensor

        assigned_scores_sum = assigned_scores.sum()
        assigned_scores_sum = torch.clamp(
            reduce_mean(assigned_scores_sum), min=1)
        loss_cls /= assigned_scores_sum

        # select positive samples mask
        num_pos = fg_mask_pre_prior.sum()
        if num_pos > 0:
            # when num_pos > 0, assigned_scores_sum will >0, so the loss_bbox
            # will not report an error
            # iou loss
            prior_bbox_mask = fg_mask_pre_prior.unsqueeze(-1).repeat(
                [1, 1, 4])  # [32, 8400, 4]
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

            # dfl loss
            dist_mask = fg_mask_pre_prior.unsqueeze(
                -1).repeat(  # [32, 8400, 68]
                    [1, 1, (self.head_module.reg_max + 1) * 4])

            # 这里其实需要的是decode之前的
            pred_dist_pos = torch.masked_select(dist_preds, dist_mask).reshape(
                [-1, 4, self.head_module.reg_max + 1])  # [n_pos, 4, 17]
            # 这里需要的都是除以了stride的变量
            assigned_ltrb = self.bbox_coder.encode(
                self.flatten_priors[..., :2],
                assigned_bboxes,
                max_dis=self.head_module.reg_max,
                eps=0.01)
            assigned_ltrb_pos = torch.masked_select(
                assigned_ltrb, prior_bbox_mask).reshape([-1, 4])
            loss_dfl = self.loss_dfl(
                pred_dist_pos,
                assigned_ltrb_pos,
                weight=bbox_weight,
                avg_factor=assigned_scores_sum)

            loss_l1 = F.l1_loss(pred_bboxes_pos, assigned_bboxes_pos)

        else:
            loss_bbox = flatten_pred_bboxes.sum() * 0
            loss_dfl = flatten_pred_bboxes.sum() * 0
            loss_l1 = (flatten_pred_bboxes.sum() * 0)

        _, world_size = get_dist_info()
        return dict(
            loss_cls=loss_cls * world_size,
            loss_bbox=loss_bbox * world_size,
            loss_dfl=loss_dfl * world_size,
            # 不参与反向传播
            loss_l1=loss_l1.detach())
