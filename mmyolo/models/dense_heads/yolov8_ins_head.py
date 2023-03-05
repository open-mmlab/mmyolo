# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import List, Sequence, Tuple, Union, Optional

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.models.utils import multi_apply
from mmdet.utils import (ConfigType, OptConfigType, OptInstanceList,
                         OptMultiConfig)
from mmengine import ConfigDict
from mmengine.dist import get_dist_info
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor

from mmyolo.registry import MODELS, TASK_UTILS
from .. import YOLOv8HeadModule, YOLOv8Head
from ..utils import gt_instances_preprocess, make_divisible
from .yolov5_head import YOLOv5Head


# class Segment(Detect):
#     # YOLOv8 Segment head for segmentation models
#     def __init__(self, nc=80, nm=32, npr=256, ch=()):
#         super().__init__(nc, ch)
#         self.nm = nm  # number of masks
#         self.npr = npr  # number of protos
#         self.proto = Proto(ch[0], self.npr, self.nm)  # protos
#         self.detect = Detect.forward
#
#         c4 = max(ch[0] // 4, self.nm)
#         self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)
#
#     def forward(self, x):
#         p = self.proto(x[0])  # mask protos
#         bs = p.shape[0]  # batch size
#
#         mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
#         x = self.detect(self, x)
#         if self.training:
#             return x, mc, p
#         return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class Proto(nn.Module):
    # YOLOv8 mask Proto module for segmentation models
    def __init__(self, c1, c_=256, c2=32, act_cfg=None, norm_cfg=None):  # ch_in, number of protos, number of masks
        super().__init__()
        self.cv1 = ConvModule(in_channels=c1, out_channels=c_, kernel_size=3, act_cfg=act_cfg, norm_cfg=norm_cfg)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = ConvModule(in_channels=c_, out_channels=c_, kernel_size=3, act_cfg=act_cfg, norm_cfg=norm_cfg)
        self.cv2 = ConvModule(in_channels=c_, out_channels=c2, kernel_size=1, act_cfg=act_cfg, norm_cfg=norm_cfg)

    def forward(self, x):
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


@MODELS.register_module()
class YOLOv8InsHeadModule(YOLOv8HeadModule):
    def __init__(self,
                 *args,
                 #####
                 num_masks=32,
                 num_protos=256,
                 ch=(),
                 #####
                 **kwargs):
        ###
        self.num_masks = num_masks
        self.num_protos = num_protos
        self.ch = ch

        super().__init__(*args, **kwargs)

    def init_weights(self, prior_prob=0.01):
        """Initialize the weight and bias of PPYOLOE head."""
        super().init_weights()
        for reg_pred, cls_pred, stride in zip(self.reg_preds, self.cls_preds,
                                              self.featmap_strides):
            reg_pred[-1].bias.data[:] = 1.0  # box
            # cls (.01 objects, 80 classes, 640 img)
            cls_pred[-1].bias.data[:self.num_classes] = math.log(
                5 / self.num_classes / (640 / stride) ** 2)

    def _init_layers(self):
        """initialize conv layers in YOLOv8 head."""
        # Init decouple head
        # 原来的层不用动
        super()._init_layers()
        self.proto = Proto(self.ch[0], self.num_protos, self.num_masks, act_cfg=self.act_cfg, norm_cfg=self.norm_cfg)
        c4 = max(self.ch[0], self.num_protos, self.num_masks)
        self.cv4 = nn.ModuleList(nn.Sequential(
            ConvModule(in_channels=x, out_channels=c4, kernel_size=3, act_cfg=self.act_cfg, norm_cfg=self.norm_cfg),
            ConvModule(in_channels=c4, out_channels=c4, kernel_size=3, act_cfg=self.act_cfg, norm_cfg=self.norm_cfg),
            ConvModule(in_channels=c4, out_channels=self.num_masks, kernel_size=1, act_cfg=None, norm_cfg=None))
                                 for x in self.ch)

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

        p = self.proto(x[0])

        return multi_apply(self.forward_single, x, self.cls_preds,
                           self.reg_preds, self.cv4), p

    def forward_single(self, x: torch.Tensor, cls_pred: nn.ModuleList,
                       reg_pred: nn.ModuleList, mask_pred: nn.ModuleList) -> Tuple:
        """Forward feature of a single scale level."""

        det_output = super().forward_single(x, cls_pred, reg_pred)
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

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        mcs: List[Tensor],
                        p: Tensor,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = True,
                        with_nms: bool = True) -> List[InstanceData]:
        pass

    def loss_by_feat(
            self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            bbox_dist_preds: Sequence[Tensor],
            batch_gt_instances: Sequence[InstanceData],
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        raise NotImplementedError
