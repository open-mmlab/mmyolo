from typing import Union, Sequence, Tuple, List, Optional
from mmdet.utils import (OptMultiConfig, ConfigType,
                         OptConfigType, OptInstanceList)

import torch
from torch import Tensor
import torch.nn as nn

from mmyolo.models.utils import make_divisible
from mmyolo.registry import MODELS

from mmdet.models.utils import multi_apply
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead

from mmengine.structures import InstanceData
from mmengine.model import BaseModule

class YOLO6DHeadModule(BaseModule):
    """YOLO6DHead head module used in 'YOLO6D'.
    
    """
    def __init__(self,
                 num_classes: int,
                 in_channels: Union[int, Sequence],
                 widen_factor: float = 1.0,
                 num_base_priors: int = 1,
                 featmap_strides: Sequence[int] = (32),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)
        
        self.num_classes = num_classes
        self.widen_factor = widen_factor
        
        self.featmap_strides = featmap_strides
        self.num_out_attrib = 19 + self.num_classes

        self.num_levels = len(self.featmap_strides)
        
        # todo: what's this?
        self.num_base_priors = num_base_priors
        
        # 通过卷积将feature map通道数转为: num_keypoints(9)*2 + conf(1) + cls(c)
        self._init_layer()
        
    def _init_layer(self):
        """initialize conv layers in YOLO6D head"""
        self.convs_pred = nn.ModuleList()
        for i in range(self.num_levels):
            conv_pred = nn.Conv2d(self.in_channels[i],
                                  self.num_base_priors * self.num_out_attrib,
                                  1)
            self.convs_pred.append(conv_pred)
        
    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        """Forward features from the upstream network"""
        
        assert len(x) == self.num_levels
        return multi_apply(self.forward_single, self.convs_pred)
        
    def forward_single(self, x: Tensor,
                       convs: nn.Module) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward feature of a single scale level"""
        
        pred_map = convs(x)
        bs, _, ny, nx = pred_map.shape
        pred_map = pred_map.view(bs, self.num_base_priors, self.num_out_attrib,
                                 ny, nx)
        
        cls_score = pred_map[:, :, :self.num_classes, ...].reshape(bs, -1, ny, nx)
        bbox_pred = pred_map

  
class YOLO6DHead(BaseDenseHead):
    def __init__(self,
                 head_module: ConfigType,
                 prior_generator: ConfigType = dict(
                     type='mmdet.YOLOAnchorGenerator',
                     base_sizes=[[(13,13)]],
                     strides=[32])
                 bbox_coder: ConfigType = dict(type='YOLOv5BBoxCoder')
                 loss_cls,
                 loss_obj,
                 loss_x,
                 loss_y,
                 thresh,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptConfigType = None,
                 ):
        super().__init__(init_cfg=init_cfg)
        
        self.head_module = MODELS.build(head_module)
        self.num_classes = head_module.num_classes
        self.featmap_strides = self.head_module.featmap_strides
        self.num_levels = len(self.featmap_strides)
    
    def init():
        self.num_base_priors

    def forward(self, x: Tuple[Tensor]):
        """Forward features from the upstream network
        
        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
            a 4D-tensor
        
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions, and objectnesses.
        """
        return self.head_module(x)
    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        """Forward features from the upstream network"""
        return self.head_module(x)
        
    def predict_by_feat(
                        )-> List[InstanceData]:
        pass
    
    def loss(self, x, batch_data_samples):
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.
        """
        
        if isinstance(batch_data_samples, list):
            outs = super().loss(x, batch_data_samples)
        else:
            outs = self(x)
            # Fast version
            loss_inputs = outs + (batch_data_samples['bboxes_labels'],
                                  batch_data_samples['img_metas'])
            losses = self.loss_by_feat(*loss_inputs)

        return losses
        
    def loss_by_feat(self,
                     cls_scores: Sequence[Tensor],
                     bbox_preds: Sequence[Tensor],
                     objectnesses: Sequence[Tensor],
                     batch_gt_instances: Sequence[InstanceData],
                     batch_img_metas: Sequence[dict],
                     batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features
        extracted by the detection head
        
        """
        
        if self.ignore_iof_thr != -1:
            pass
        
        # 1. Convert gt to norm format
        batch_targets_normed = self._convert_gt_to_norm_format(
            batch_gt_instances, batch_img_metas
        )
            
    def _convert_gt_to_norm_format(self,
                                   batch_gt_instances: Sequence[InstanceData],
                                   batch_img_metas: Sequence[dict]) -> Tensor:
        if isinstance(batch_gt_instances, torch.Tensor):
            # fast version
            pass
        else:
            batch_target_list = []
            # convert xyxy bbox to yolo format
            for i, gt_instances in enumerate(batch_gt_instances):
                img_shape = batch_img_metas[i]['batch_input_shape']
                # todo: 修改json文件的bbox和label名称
                bboxes = gt_instances.bboxes
                labels = gt_instances.labels
                control_points = gt_instances.control_points
                
                xy1, xy2 = bboxes.split((2,2), dim=-1)
                bboxes = torch.cat([(xy1+xy2)/2, (xy2-xy1)], dim=-1)
                bboxes[:, 1::2] /= img_shape[0]
                bboxes[:, 0::2] /= img_shape[1]
                
                control_points[:, 0] /= img_shape[0]
                control_points[:, 1] /= img_shape[1]
                
                index = bboxes.new_full((len(bboxes), 1), i)
                
                # (batch_idx, label, normed_bbox, normed_control_points)
                target = torch.cat((index, labels[:, None].float(), bboxes,
                                    control_points), dim=1)
                batch_target_list.append(target)
            
            # (num_base_priors, num_bboxes, 6+18)
            batch_targets_normed = torch.cat(
                batch_target_list, dim=0).repeat(self.num_base_priors, 1, 1)
        
        # (num_base_priors, num_bboxes, 1)
        batch_targets_prior_inds = self.prior_inds.repeat(
            1, batch_targets_normed.shape[1])[..., None]
        # (num_base_priors, num_bboxes, 6+18+1)
        batch_targets_normed = torch.cat(
            (batch_targets_normed, batch_targets_prior_inds), 2)
        
        return batch_targets_normed
            
    
    def predict_by_feat():