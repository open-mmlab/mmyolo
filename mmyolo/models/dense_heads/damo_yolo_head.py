import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
import numpy as np
from mmcv.cnn import Scale
from ..task_modules.assigners.ota_assigner import AlignOTAAssigner
from ..utils import BoxList
from mmdet.models.utils import multi_apply
from mmdet.utils import reduce_mean
from mmdet.models.losses import (DistributionFocalLoss, GIoULoss, weighted_loss)
from typing import List, Optional, Tuple
from torch import Tensor
import torchvision
from mmengine.config import ConfigDict
from mmdet.structures import SampleList
from mmengine.structures import InstanceData
from mmdet.utils import InstanceList, OptMultiConfig

@weighted_loss
def quality_focal_loss(pred, target, beta=2.0, use_sigmoid=True):
    r"""Quality Focal Loss (QFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.
    Args:
        pred (torch.Tensor): Predicted joint representation of classification
            and quality (IoU) estimation with shape (N, C), C is the number of
            classes.
        target (tuple([torch.Tensor])): Target category label with shape (N,)
            and target quality label with shape (N,).
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.
    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    assert len(target) == 2, """target for QFL must be a tuple of two elements,
        including category label and quality label, respectively"""
    # label denotes the category id, score denotes the quality score
    label, score = target
    if use_sigmoid:
        func = F.binary_cross_entropy_with_logits
    else:
        func = F.binary_cross_entropy
    # negatives are supervised by 0 quality score
    pred_sigmoid = pred.sigmoid() if use_sigmoid else pred
    scale_factor = pred_sigmoid  # 8400, 81
    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = func(pred, zerolabel, reduction='none') * scale_factor.pow(beta)

    bg_class_ind = pred.size(1)
    pos = ((label >= 0) &
           (label < bg_class_ind)).nonzero(as_tuple=False).squeeze(1)
    pos_label = label[pos].long()
    # positives are supervised by bbox quality (IoU) score
    scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
    loss[pos,
         pos_label] = func(pred[pos, pos_label], score[pos],
                           reduction='none') * scale_factor.abs().pow(beta)

    loss = loss.sum(dim=1, keepdim=False)
    return loss

class QualityFocalLoss(nn.Module):
    r"""Quality Focal Loss (QFL) is a variant of `Generalized Focal Loss:
    Learning Qualified and Distributed Bounding Boxes for Dense Object
    Detection <https://arxiv.org/abs/2006.04388>`_.
    Args:
        use_sigmoid (bool): Whether sigmoid operation is conducted in QFL.
            Defaults to True.
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """
    def __init__(self,
                 use_sigmoid=True,
                 beta=2.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(QualityFocalLoss, self).__init__()
        # assert use_sigmoid is True, 'Only sigmoid in QFL supported now.'
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.
        Args:
            pred (torch.Tensor): Predicted joint representation of
                classification and quality (IoU) estimation with shape (N, C),
                C is the number of classes.
            target (tuple([torch.Tensor])): Target category label with shape
                (N,) and target quality label with shape (N,).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override
                     if reduction_override else self.reduction)
        loss_cls = self.loss_weight * quality_focal_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            use_sigmoid=self.use_sigmoid,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_cls


def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   iou_thr,
                   max_num=100,
                   score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels \
            are 0-based.
    """
    num_classes = multi_scores.size(1)
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(multi_scores.size(0),
                                              num_classes, 4)
    scores = multi_scores
    # filter out boxes with low scores
    valid_mask = scores > score_thr  # 1000 * 80 bool

    # We use masked_select for ONNX exporting purpose,
    # which is equivalent to bboxes = bboxes[valid_mask]
    # (TODO): as ONNX does not support repeat now,
    # we have to use this ugly code
    # bboxes -> 1000, 4
    bboxes = torch.masked_select(
        bboxes,
        torch.stack((valid_mask, valid_mask, valid_mask, valid_mask),
                    -1)).view(-1, 4)  # mask->  1000*80*4, 80000*4
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = torch.masked_select(scores, valid_mask)
    labels = valid_mask.nonzero(as_tuple=False)[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
        scores = multi_bboxes.new_zeros((0, ))

        return bboxes, scores, labels

    keep = torchvision.ops.batched_nms(bboxes, scores, labels, iou_thr)

    if max_num > 0:
        keep = keep[:max_num]

    return bboxes[keep], scores[keep], labels[keep]


def postprocess(cls_scores: object,
                bbox_preds: object,
                num_classes: object,
                conf_thre: object = 0.7,
                nms_thre: object = 0.45,
                imgs: object = None) -> object:
    batch_size = bbox_preds.size(0)
    output = [None for _ in range(batch_size)]
    for i in range(batch_size):
        # If none are remaining => process next image
        if not bbox_preds[i].size(0):
            continue
        detections, scores, labels = multiclass_nms(bbox_preds[i],
                                                    cls_scores[i], conf_thre,
                                                    nms_thre, 500)
        detections = torch.cat((detections, torch.ones_like(
            scores[:, None]), scores[:, None], labels[:, None]),
                               dim=1)

        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    # transfer to BoxList
    for i in range(len(output)):
        res = output[i]
        if res is None or imgs is None:
            boxlist = BoxList(torch.zeros(0, 4), (0, 0), mode='xyxy')
            boxlist.add_field('objectness', 0)
            boxlist.add_field('scores', 0)
            boxlist.add_field('labels', -1)
        else:
            img_h, img_w = imgs[i].ori_shape
            boxlist = BoxList(res[:, :4], (img_w, img_h), mode='xyxy')
            boxlist.add_field('objectness', res[:, 4])
            boxlist.add_field('scores', res[:, 5])
            boxlist.add_field('labels', res[:, 6] + 1)
        output[i] = boxlist
    result_list = []
    for boxlist in output:
        result = InstanceData()
        result.bboxes = boxlist.bbox
        result.labels = boxlist.extra_fields['labels'].to(int)-1
        result.scores = boxlist.extra_fields['scores']
        result_list.append(result)
    return result_list


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def bias_init_with_prob(prior_prob):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.
    """
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], -1)


def bbox2distance(points, bbox, max_dis=None, eps=0.1):
    """Decode bounding box based on distances.
    """
    left = points[:, 0] - bbox[:, 0]
    top = points[:, 1] - bbox[:, 1]
    right = bbox[:, 2] - points[:, 0]
    bottom = bbox[:, 3] - points[:, 1]
    if max_dis is not None:
        left = left.clamp(min=0, max=max_dis - eps)
        top = top.clamp(min=0, max=max_dis - eps)
        right = right.clamp(min=0, max=max_dis - eps)
        bottom = bottom.clamp(min=0, max=max_dis - eps)
    return torch.stack([left, top, right, bottom], -1)


class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.
    """
    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project',
                             torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.
        """
        b, hw, _, _ = x.size()
        x = x.reshape(b * hw * 4, self.reg_max + 1)
        y = self.project.type_as(x).unsqueeze(1)
        x = torch.matmul(x, y).reshape(b, hw, 4)
        return x

from mmyolo.registry import MODELS

@MODELS.register_module()
class ZeroHead(BaseDenseHead):
    """Ref to Generalized Focal Loss V2: Learning Reliable Localization Quality
    Estimation for Dense Object Detection.
    """
    def __init__(
            self,
            num_classes=80,
            in_channels=[128,256,512],
            stacked_convs=0,
            feat_channels=256,
            reg_max=16,
            strides=[8, 16, 32],
            norm='gn',
            act='silu',
            nms_conf_thre=0.05,
            nms_iou_thre=0.7,
            nms=True,
            init_cfg=None,
            **kwargs):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.stacked_convs = stacked_convs
        self.act = act
        self.strides = strides
        if stacked_convs == 0:
            feat_channels = in_channels
        if isinstance(feat_channels, list):
            self.feat_channels = feat_channels
        else:
            self.feat_channels = [feat_channels] * len(self.strides)
        # add 1 for keep consistance with former models
        self.cls_out_channels = num_classes + 1
        self.reg_max = reg_max

        self.nms = nms
        self.nms_conf_thre = nms_conf_thre
        self.nms_iou_thre = nms_iou_thre

        self.assigner = AlignOTAAssigner(center_radius=2.5,
                                         cls_weight=1.0,
                                         iou_weight=3.0)

        self.feat_size = [torch.zeros(4) for _ in strides]

        super(ZeroHead, self).__init__()
        self.integral = Integral(self.reg_max)
        self.loss_dfl = DistributionFocalLoss(loss_weight=0.25)
        self.loss_cls = QualityFocalLoss(use_sigmoid=False,
                                         beta=2.0,
                                         loss_weight=1.0)
        self.loss_bbox = GIoULoss(loss_weight=2.0)

        self._init_layers()

    def _build_not_shared_convs(self, in_channel, feat_channels):
        cls_convs = nn.ModuleList()
        reg_convs = nn.ModuleList()

        for i in range(self.stacked_convs):
            chn = feat_channels if i > 0 else in_channel
            kernel_size = 3 if i > 0 else 1
            # cls_convs.append(
            #     ConvBNAct(chn,
            #               feat_channels,
            #               kernel_size,
            #               stride=1,
            #               groups=1,
            #               norm='bn',
            #               act=self.act))
            # reg_convs.append(
            #     ConvBNAct(chn,
            #               feat_channels,
            #               kernel_size,
            #               stride=1,
            #               groups=1,
            #               norm='bn',
            #               act=self.act))

        return cls_convs, reg_convs

    def _init_layers(self):
        """Initialize layers of the head."""
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        for i in range(len(self.strides)):
            cls_convs, reg_convs = self._build_not_shared_convs(
                self.in_channels[i], self.feat_channels[i])
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)

        self.gfl_cls = nn.ModuleList([
            nn.Conv2d(self.feat_channels[i],
                      self.cls_out_channels,
                      3,
                      padding=1) for i in range(len(self.strides))
        ])

        self.gfl_reg = nn.ModuleList([
            nn.Conv2d(self.feat_channels[i],
                      4 * (self.reg_max + 1),
                      3,
                      padding=1) for i in range(len(self.strides))
        ])

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        """Initialize weights of the head."""
        for cls_conv in self.cls_convs:
            for m in cls_conv:
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
        for reg_conv in self.reg_convs:
            for m in reg_conv:
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        for i in range(len(self.strides)):
            normal_init(self.gfl_cls[i], std=0.01, bias=bias_cls)
            normal_init(self.gfl_reg[i], std=0.01)

    def forward(self, xin, labels=None, imgs=None, aux_targets=None):
        b, c, h, w = xin[0].shape
        # forward for bboxes and classification prediction
        cls_scores, bbox_preds, bbox_before_softmax = multi_apply(
            self.forward_single,
            xin,
            self.cls_convs,
            self.reg_convs,
            self.gfl_cls,
            self.gfl_reg,
            self.scales,
        )

        if self.training:
            return cls_scores, bbox_preds, bbox_before_softmax
        else:
            return cls_scores, bbox_preds

    def loss(self, xin, labels=None, imgs=None, aux_targets=None):

        # prepare labels during training
        b, c, h, w = xin[0].shape
        if labels is not None:
            gt_bbox_list = []
            gt_cls_list = []
            for label in labels:
                gt_bbox_list.append(label.bbox)
                gt_cls_list.append((label.get_field('labels') -
                                    1).long())  # labels starts from 1

        # prepare priors for label assignment and bbox decode
        mlvl_priors_list = [
            self.get_single_level_center_priors(xin[i].shape[0],
                                                xin[i].shape[-2:],
                                                stride,
                                                dtype=torch.float32,
                                                device=xin[0].device)
            for i, stride in enumerate(self.strides)
        ]
        mlvl_priors = torch.cat(mlvl_priors_list, dim=1)

        # forward for bboxes and classification prediction
        cls_scores, bbox_preds, bbox_before_softmax = multi_apply(
            self.forward_single,
            xin,
            self.cls_convs,
            self.reg_convs,
            self.gfl_cls,
            self.gfl_reg,
            self.scales,
        )
        cls_scores = torch.cat(cls_scores, dim=1)
        bbox_preds = torch.cat(bbox_preds, dim=1)
        bbox_before_softmax = torch.cat(bbox_before_softmax, dim=1)

        # calculating losses
        loss = self._loss(
            cls_scores,
            bbox_preds,
            bbox_before_softmax,
            gt_bbox_list,
            gt_cls_list,
            mlvl_priors,
        )
        return loss


    def loss_by_feat(self, **kwargs) -> dict:
        pass


    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        # prepare priors for label assignment and bbox decode
        if self.feat_size[0] != x[0].shape:
            mlvl_priors_list = [
                self.get_single_level_center_priors(x[i].shape[0],
                                                    x[i].shape[-2:],
                                                    stride,
                                                    dtype=torch.float32,
                                                    device=x[0].device)
                for i, stride in enumerate(self.strides)
            ]
            self.mlvl_priors = torch.cat(mlvl_priors_list, dim=1)
            self.feat_size[0] = x[0].shape
        # x = (torch.ones((20,128,80,80),device='cuda'),
        #      torch.ones((20,256,40,40),device='cuda'),
        #      torch.ones((20,512,20,20),device='cuda'))
        # forward for bboxes and classification prediction
        cls_scores, bbox_preds,bbox_before_softmax = multi_apply(
            self.forward_single,
            x,
            self.cls_convs,
            self.reg_convs,
            self.gfl_cls,
            self.gfl_reg,
            self.scales,
        )
        cls_scores = torch.cat(cls_scores, dim=1)[:, :, :self.num_classes]
        bbox_preds = torch.cat(bbox_preds, dim=1)
        # batch bbox decode
        bbox_preds = self.integral(bbox_preds) * self.mlvl_priors[..., 2, None]
        bbox_preds = distance2bbox(self.mlvl_priors[..., :2], bbox_preds)
        ## TODO:discuss with teacher about the postprocess postion
        if self.nms:
            output = postprocess(cls_scores, bbox_preds, self.num_classes,
                                 self.nms_conf_thre, self.nms_iou_thre, imgs=batch_data_samples)
            return output
        return cls_scores, bbox_preds


    def forward_single(self, x, cls_convs, reg_convs, gfl_cls, gfl_reg, scale):
        """Forward feature of a single scale level.

        """
        cls_feat = x
        reg_feat = x

        for cls_conv, reg_conv in zip(cls_convs, reg_convs):
            cls_feat = cls_conv(cls_feat)
            reg_feat = reg_conv(reg_feat)

        bbox_pred = scale(gfl_reg(reg_feat)).float()
        N, C, H, W = bbox_pred.size()
        ##
        bbox_before_softmax = bbox_pred.reshape(N, 4, self.reg_max + 1, H,
                                                W)
        bbox_before_softmax = bbox_before_softmax.flatten(
            start_dim=3).permute(0, 3, 1, 2)
        bbox_pred = F.softmax(bbox_pred.reshape(N, 4, self.reg_max + 1, H, W),
                              dim=2)

        cls_score = gfl_cls(cls_feat).sigmoid()

        cls_score = cls_score.flatten(start_dim=2).permute(
            0, 2, 1)  # N, h*w, self.num_classes+1
        bbox_pred = bbox_pred.flatten(start_dim=3).permute(
            0, 3, 1, 2)  # N, h*w, 4, self.reg_max+1
        return cls_score, bbox_pred, bbox_before_softmax

    def get_single_level_center_priors(self, batch_size, featmap_size, stride,
                                       dtype, device):

        h, w = featmap_size
        x_range = (torch.arange(0, int(w), dtype=dtype,
                                device=device)) * stride
        y_range = (torch.arange(0, int(h), dtype=dtype,
                                device=device)) * stride

        x = x_range.repeat(h, 1)
        y = y_range.unsqueeze(-1).repeat(1, w)

        y = y.flatten()
        x = x.flatten()
        strides = x.new_full((x.shape[0], ), stride)
        priors = torch.stack([x, y, strides, strides], dim=-1)

        return priors.unsqueeze(0).repeat(batch_size, 1, 1)

    def _loss(
        self,
        cls_scores,
        bbox_preds,
        bbox_before_softmax,
        gt_bboxes,
        gt_labels,
        mlvl_center_priors,
        gt_bboxes_ignore=None,
    ):
        """Compute losses of the head.

        """
        device = cls_scores[0].device

        # get decoded bboxes for label assignment
        dis_preds = self.integral(bbox_preds) * mlvl_center_priors[..., 2,
                                                                   None]
        decoded_bboxes = distance2bbox(mlvl_center_priors[..., :2], dis_preds)
        cls_reg_targets = self.get_targets(cls_scores,
                                           decoded_bboxes,
                                           gt_bboxes,
                                           mlvl_center_priors,
                                           gt_labels_list=gt_labels)

        if cls_reg_targets is None:
            return None

        (labels_list, label_scores_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, dfl_targets_list, num_pos) = cls_reg_targets

        num_total_pos = max(
            reduce_mean(torch.tensor(num_pos).type(
                torch.float).to(device)).item(), 1.0)

        labels = torch.cat(labels_list, dim=0)
        label_scores = torch.cat(label_scores_list, dim=0)
        bbox_targets = torch.cat(bbox_targets_list, dim=0)
        dfl_targets = torch.cat(dfl_targets_list, dim=0)

        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # bbox_preds = bbox_preds.reshape(-1, 4 * (self.reg_max + 1))
        bbox_before_softmax = bbox_before_softmax.reshape(
            -1, 4 * (self.reg_max + 1))
        decoded_bboxes = decoded_bboxes.reshape(-1, 4)

        loss_qfl = self.loss_cls(cls_scores, (labels, label_scores),
                                 avg_factor=num_total_pos)

        pos_inds = torch.nonzero((labels >= 0) & (labels < self.num_classes),
                                 as_tuple=False).squeeze(1)

        if len(pos_inds) > 0:
            weight_targets = cls_scores.detach()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]
            norm_factor = max(reduce_mean(weight_targets.sum()).item(), 1.0)
            loss_bbox = self.loss_bbox(
                decoded_bboxes[pos_inds],
                bbox_targets[pos_inds],
                weight=weight_targets,
                avg_factor=1.0 * norm_factor,
            )
            loss_dfl = self.loss_dfl(
                bbox_before_softmax[pos_inds].reshape(-1, self.reg_max + 1),
                dfl_targets[pos_inds].reshape(-1),
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0 * norm_factor,
            )

        else:
            loss_bbox = bbox_preds.sum() * 0.0
            loss_dfl = bbox_preds.sum() * 0.0

        total_loss = loss_qfl + loss_bbox + loss_dfl

        return dict(
            total_loss=total_loss,
            loss_cls=loss_qfl,
            loss_bbox=loss_bbox,
            loss_dfl=loss_dfl,
        )

    def get_targets(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    mlvl_center_priors,
                    gt_labels_list=None,
                    unmap_outputs=True):
        """Get targets for GFL head.

        """
        num_imgs = mlvl_center_priors.shape[0]

        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]

        (all_labels, all_label_scores, all_label_weights, all_bbox_targets,
         all_bbox_weights, all_dfl_targets, all_pos_num) = multi_apply(
             self.get_target_single,
             mlvl_center_priors,
             cls_scores,
             bbox_preds,
             gt_bboxes_list,
             gt_labels_list,
         )
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        all_pos_num = sum(all_pos_num)

        return (all_labels, all_label_scores, all_label_weights,
                all_bbox_targets, all_bbox_weights, all_dfl_targets,
                all_pos_num)

    def get_target_single(self,
                          center_priors,
                          cls_scores,
                          bbox_preds,
                          gt_bboxes,
                          gt_labels,
                          unmap_outputs=True,
                          gt_bboxes_ignore=None):
        """Compute regression, classification targets for anchors in a single
        image.

        """
        # assign gt and sample anchors

        num_valid_center = center_priors.shape[0]

        labels = center_priors.new_full((num_valid_center, ),
                                        self.num_classes,
                                        dtype=torch.long)
        label_weights = center_priors.new_zeros(num_valid_center,
                                                dtype=torch.float)
        label_scores = center_priors.new_zeros(num_valid_center,
                                               dtype=torch.float)

        bbox_targets = torch.zeros_like(center_priors)
        bbox_weights = torch.zeros_like(center_priors)
        dfl_targets = torch.zeros_like(center_priors)

        if gt_labels.size(0) == 0:

            return (labels, label_scores, label_weights, bbox_targets,
                    bbox_weights, dfl_targets, 0)

        assign_result = self.assigner.assign(cls_scores.detach(),
                                             center_priors,
                                             bbox_preds.detach(), gt_bboxes,
                                             gt_labels)

        pos_inds, neg_inds, pos_bbox_targets, pos_assign_gt_inds = self.sample(
            assign_result, gt_bboxes)
        pos_ious = assign_result.max_overlaps[pos_inds]

        if len(pos_inds) > 0:
            labels[pos_inds] = gt_labels[pos_assign_gt_inds]
            label_scores[pos_inds] = pos_ious
            label_weights[pos_inds] = 1.0

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            dfl_targets[pos_inds, :] = (bbox2distance(
                center_priors[pos_inds, :2] / center_priors[pos_inds, None, 2],
                pos_bbox_targets / center_priors[pos_inds, None, 2],
                self.reg_max))
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        # map up to original set of anchors

        return (labels, label_scores, label_weights, bbox_targets,
                bbox_weights, dfl_targets, pos_inds.size(0))

    def sample(self, assign_result, gt_bboxes):
        pos_inds = torch.nonzero(assign_result.gt_inds > 0,
                                 as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(assign_result.gt_inds == 0,
                                 as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert pos_assigned_gt_inds.numel() == 0
            pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)
            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds, :]

        return pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds