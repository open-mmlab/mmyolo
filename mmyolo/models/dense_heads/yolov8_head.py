# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.models.utils import multi_apply
from mmdet.utils import (ConfigType, OptConfigType, OptInstanceList,
                         OptMultiConfig)
from mmengine.model import BaseModule, bias_init_with_prob
from mmengine.structures import InstanceData
from torch import Tensor

from mmyolo.registry import MODELS, TASK_UTILS
from ..utils import make_divisible
from .yolov5_head import YOLOv5Head


@MODELS.register_module()
class YOLOv8HeadModule(BaseModule):
    """YOLOv8HeadModule head module used in `YOLOv8`.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (Union[int, Sequence]): Number of channels in the input
            feature map.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        num_base_priors (int): The number of priors (points) at a point
            on the feature grid.
        featmap_strides (Sequence[int]): Downsample factor of each feature map.
             Defaults to [8, 16, 32].
        reg_max (int): Max value of integral set :math: ``{0, ..., reg_max-1}``
            in QFL setting. Defaults to 16.
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
                 featmap_strides: Sequence[int] = (8, 16, 32),
                 reg_max: int = 16,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.featmap_strides = featmap_strides
        self.num_levels = len(self.featmap_strides)
        self.num_base_priors = num_base_priors
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_channels = in_channels
        self.reg_max = reg_max

        in_channels = []
        for channel in self.in_channels:
            channel = make_divisible(channel, widen_factor)
            in_channels.append(channel)
        self.in_channels = in_channels

        self._init_layers()

    def init_weights(self):
        """Initialize weights of the head."""
        # Use prior in model initialization to improve stability
        super().init_weights()

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.reset_parameters()

        bias_init = bias_init_with_prob(0.01)
        for conv_cls in self.cls_preds:
            conv_cls.bias.data.fill_(bias_init)

    def _init_layers(self):
        """initialize conv layers in YOLOv8 head."""
        # Init decouple head
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()

        reg_out_channels = max(
            (16, self.in_channels[0] // 4, self.reg_max * 4))
        cls_out_channels = max(self.in_channels[0], self.num_classes)

        for i in range(self.num_levels):
            self.reg_preds.append(
                nn.Sequential(
                    ConvModule(
                        in_channels=self.in_channels[i],
                        out_channels=reg_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg),
                    ConvModule(
                        in_channels=reg_out_channels,
                        out_channels=reg_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg),
                    nn.Conv2d(
                        in_channels=reg_out_channels,
                        out_channels=4 * self.reg_max,
                        kernel_size=1)))
            self.cls_preds.append(
                nn.Sequential(
                    ConvModule(
                        in_channels=self.in_channels[i],
                        out_channels=cls_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg),
                    ConvModule(
                        in_channels=cls_out_channels,
                        out_channels=cls_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg),
                    nn.Conv2d(
                        in_channels=cls_out_channels,
                        out_channels=self.num_classes,
                        kernel_size=1)))

        proj = torch.linspace(0, self.reg_max - 1,
                              self.reg_max).view([1, self.reg_max, 1, 1])
        self.register_buffer('proj', proj, persistent=False)

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
        return multi_apply(self.forward_single, x, self.cls_preds,
                           self.reg_preds)

    def forward_single(self, x: torch.Tensor, cls_pred: nn.ModuleList,
                       reg_pred: nn.ModuleList) -> Tuple:
        """Forward feature of a single scale level."""
        b, _, h, w = x.shape
        cls_logit = cls_pred(x)
        bbox_dist_preds = reg_pred(x)
        if self.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, self.reg_max, h * w]).permute(0, 2, 3, 1)
            # TODO: Test whether use matmul instead of conv can
            #  speed up training.
            bbox_preds = F.conv2d(F.softmax(bbox_dist_preds, dim=1), self.proj)
        else:
            bbox_preds = bbox_dist_preds
        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds
        else:
            return cls_logit, bbox_preds


# TODO Training mode is currently not supported
@MODELS.register_module()
class YOLOv8Head(YOLOv5Head):
    """YOLOv8Head head used in `YOLOv8`.

    Args:
        head_module(:obj:`ConfigDict` or dict): Base module used for YOLOv8Head
        prior_generator(dict): Points generator feature maps
            in 2D points-based detectors.
        bbox_coder (:obj:`ConfigDict` or dict): Config of bbox coder.
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
        loss_bbox (:obj:`ConfigDict` or dict): Config of localization loss.
        loss_dfl (:obj:`ConfigDict` or dict): Config of Distribution Focal
            Loss.
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
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)
        self.loss_dfl = MODELS.build(loss_dfl)
        # yolov8 doesn't need loss_obj
        self.loss_obj = None

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
            bbox_dist_preds (Sequence[Tensor]): Box distribution logits for
                each scale level with shape (bs, reg_max + 1, H*W, 4).
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

        # 进行结果对齐时候使用
        # o_preds = torch.load('/data/gulingrui/code/ultralytics/feats.pth')
        # o_targets, o_pred_distri, o_pred_bboxes, o_pred_scores = torch.load(
        #     '/data/gulingrui/code/ultralytics/inputs.pth')
        # o_loss = torch.load('/data/gulingrui/code/ultralytics/loss.pth')
        # print(o_loss)
        #
        # # 准备数据
        # cls_scores = [i[:, -80:] for i in o_preds]
        # # (16, 64, h, w) -> (16, 64, h*w) -> (16, h*w, 64)
        # -> (16, h*w, 4, 16)
        # bbox_dist_preds = [
        #     i[:, :64].view(16, 64, -1).permute(0, 2,
        #     1).view(16, -1, 4,
        #     16).permute(0, 3, 1, 2)
        #     for i in o_preds
        # ]
        # # -> (16, 16, h*w, 4)
        # assert bbox_dist_preds[
        # 0].shape[-1] == 4 and bbox_dist_preds[0].shape[
        #     1] == 16
        # bbox_preds = []
        # for i in bbox_dist_preds:
        #     tempres = F.conv2d(
        #         F.softmax(i.float(), dim=1), self.head_module.proj)
        #     bbox_preds.append(tempres)

        num_imgs = len(batch_img_metas)
        # num_imgs = 16

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
        gt_info = self.gt_instances_preprocess(batch_gt_instances, num_imgs)
        # 对齐时候使用
        # gt_info = o_targets
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
        # (bs, reg_max+1, n, 4) -> (bs, n, 4, reg_max+1)
        flatten_pred_dists = [
            bbox_pred_org.permute(0, 2, 3,
                                  1).reshape(num_imgs, -1,
                                             self.head_module.reg_max * 4)
            for bbox_pred_org in bbox_dist_preds
        ]

        flatten_dist_preds = torch.cat(flatten_pred_dists, dim=1)
        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_pred_bboxes = torch.cat(flatten_pred_bboxes, dim=1)
        flatten_pred_bboxes = self.bbox_coder.decode(
            self.flatten_priors_train[..., :2], flatten_pred_bboxes,
            self.stride_tensor[..., 0])
        pred_scores = torch.sigmoid(flatten_cls_preds)

        o_1, o_2, o_3, o_4, o_5, o_6 = torch.load(
            '/data/gulingrui/code/ultralytics/assigner.pth')
        o_assigned_bboxes, o_assigned_scores, o_fg_mask_pre_prior = torch.load(
            '/data/gulingrui/code/ultralytics/assignerres.pth')

        assigned_result = self.assigner(flatten_pred_bboxes.detach(),
                                        pred_scores.detach(),
                                        self.flatten_priors_train, gt_labels,
                                        gt_bboxes, pad_bbox_flag)

        assigned_bboxes = assigned_result['assigned_bboxes']
        assigned_scores = assigned_result['assigned_scores']
        fg_mask_pre_prior = assigned_result['fg_mask_pre_prior']

        # # cls loss
        # with torch.cuda.amp.autocast(enabled=False):
        #     loss_cls = self.loss_cls(flatten_cls_preds, assigned_scores)
        assigned_scores_sum = assigned_scores.sum()

        loss_cls = self.loss_cls(flatten_cls_preds, assigned_scores).sum()
        loss_cls /= assigned_scores_sum

        # rescale bbox
        assigned_bboxes /= self.stride_tensor
        flatten_pred_bboxes /= self.stride_tensor

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
                pred_bboxes_pos, assigned_bboxes_pos,
                weight=bbox_weight) / assigned_scores_sum

            # dfl loss
            dist_mask = fg_mask_pre_prior.unsqueeze(-1).repeat(
                [1, 1, (self.head_module.reg_max) * 4])

            pred_dist_pos = torch.masked_select(
                flatten_dist_preds,
                dist_mask).reshape([-1, 4, self.head_module.reg_max])
            assigned_ltrb = self.bbox_coder.encode(
                self.flatten_priors_train[..., :2] / self.stride_tensor,
                assigned_bboxes,
                max_dis=self.head_module.reg_max - 1,
                eps=0.01)
            assigned_ltrb_pos = torch.masked_select(
                assigned_ltrb, prior_bbox_mask).reshape([-1, 4])
            loss_dfl = self.loss_dfl(
                pred_dist_pos.reshape(-1, self.head_module.reg_max),
                assigned_ltrb_pos.reshape(-1),
                weight=bbox_weight.expand(-1, 4).reshape(-1),
                avg_factor=assigned_scores_sum)
        else:
            loss_bbox = flatten_pred_bboxes.sum() * 0
            loss_dfl = flatten_pred_bboxes.sum() * 0

        return dict(
            loss_cls=loss_cls * num_imgs,
            loss_bbox=loss_bbox * num_imgs,
            loss_dfl=loss_dfl * num_imgs)

    @staticmethod
    def gt_instances_preprocess(batch_gt_instances: Union[Tensor, Sequence],
                                batch_size: int) -> Tensor:
        """Split batch_gt_instances with batch size, from [all_gt_bboxes, 6]
        to.

        [batch_size, number_gt, 5]. If some shape of single batch smaller than
        gt bbox len, then using [-1., 0., 0., 0., 0.] to fill.

        Args:
            batch_gt_instances (Sequence[Tensor]): Ground truth
                instances for whole batch, shape [all_gt_bboxes, 6]
            batch_size (int): Batch size.

        Returns:
            Tensor: batch gt instances data, shape [batch_size, number_gt, 5]
        """
        if isinstance(batch_gt_instances, Sequence):
            max_gt_bbox_len = max(
                [len(gt_instances) for gt_instances in batch_gt_instances])
            # fill [-1., 0., 0., 0., 0.] if some shape of
            # single batch not equal max_gt_bbox_len
            batch_instance_list = []
            for index, gt_instance in enumerate(batch_gt_instances):
                bboxes = gt_instance.bboxes
                labels = gt_instance.labels
                batch_instance_list.append(
                    torch.cat((labels[:, None], bboxes), dim=-1))

                if bboxes.shape[0] >= max_gt_bbox_len:
                    continue

                fill_tensor = bboxes.new_full(
                    [max_gt_bbox_len - bboxes.shape[0], 5], 0)
                fill_tensor[:, 0] = -1.
                batch_instance_list[index] = torch.cat(
                    (batch_instance_list[-1], fill_tensor), dim=0)

            return torch.stack(batch_instance_list)
        else:
            # faster version
            # sqlit batch gt instance [all_gt_bboxes, 6] ->
            # [batch_size, number_gt_each_batch, 5]
            batch_instance_list = []
            max_gt_bbox_len = 0
            for i in range(batch_size):
                single_batch_instance = \
                    batch_gt_instances[batch_gt_instances[:, 0] == i, :]
                single_batch_instance = single_batch_instance[:, 1:]
                batch_instance_list.append(single_batch_instance)
                if len(single_batch_instance) > max_gt_bbox_len:
                    max_gt_bbox_len = len(single_batch_instance)

            # fill [-1., 0., 0., 0., 0.] if some shape of
            # single batch not equal max_gt_bbox_len
            for index, gt_instance in enumerate(batch_instance_list):
                if gt_instance.shape[0] >= max_gt_bbox_len:
                    continue
                fill_tensor = batch_gt_instances.new_full(
                    [max_gt_bbox_len - gt_instance.shape[0], 5], 0)
                fill_tensor[:, 0] = -1.
                batch_instance_list[index] = torch.cat(
                    (batch_instance_list[index], fill_tensor), dim=0)

            return torch.stack(batch_instance_list)
