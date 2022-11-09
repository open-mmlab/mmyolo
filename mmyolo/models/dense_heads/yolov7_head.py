# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.utils import OptInstanceList
from mmengine.dist import get_dist_info
from mmengine.structures import InstanceData
from torch import Tensor

from mmyolo.registry import MODELS
from .yolov5_head import YOLOv5Head, YOLOv5HeadModule


class ImplicitA(nn.Module):

    def __init__(self, channel, mean=0., std=.02):
        super().__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit + x


class ImplicitM(nn.Module):

    def __init__(self, channel, mean=1., std=.02):
        super().__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit * x


@MODELS.register_module()
class YOLOv7HeadModule(YOLOv5HeadModule):

    def _init_layers(self):
        """initialize conv layers in YOLOv5 head."""
        self.convs_pred = nn.ModuleList()
        for i in range(self.num_levels):
            conv_pred = nn.Sequential(
                ImplicitA(self.in_channels[i]),
                nn.Conv2d(self.in_channels[i],
                          self.num_base_priors * self.num_out_attrib, 1),
                ImplicitM(self.num_base_priors * self.num_out_attrib),
            )
            self.convs_pred.append(conv_pred)

    def init_weights(self):
        """Initialize the bias of YOLOv5 head."""
        super(YOLOv5HeadModule, self).init_weights()
        for mi, s in zip(self.convs_pred, self.featmap_strides):  # from
            mi = mi[1]  # nn.Conv2d

            b = mi.bias.data.view(3, -1)
            # obj (8 objects per 640 image)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:] += math.log(0.6 / (self.num_classes - 0.99))

            mi.bias.data = b.view(-1)


@MODELS.register_module()
class YOLOv7Head(YOLOv5Head):
    """YOLOv7Head head used in `YOLOv7 <https://arxiv.org/abs/2207.02696>`_.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def loss_by_feat(
            self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            objectnesses: Sequence[Tensor],
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
            objectnesses (Sequence[Tensor]): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
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

        # x = torch.load('a.pth')
        # cls_scores = x['cls_scores']
        # bbox_preds = x['bbox_preds']
        # objectnesses = x['objectnesses']
        # batch_gt_instances = x['batch_gt_instances']

        # 1. Convert gt to norm format (num_bboxes, 6)
        batch_targets_normed = self._convert_gt_to_norm_format(
            batch_gt_instances, batch_img_metas)

        device = cls_scores[0].device
        loss_cls = torch.zeros(1, device=device)
        loss_box = torch.zeros(1, device=device)
        loss_obj = torch.zeros(1, device=device)

        indices, anch = [], []
        na, nt = self.num_base_priors, batch_targets_normed.shape[0]  # number of anchors, targets
        gain = torch.ones(7, device=device).long()  # normalized to gridspace gain
        ai = self.prior_inds.repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((batch_targets_normed.repeat(na, 1, 1), ai[:, :, None]),
                            2)  # append anchor indices (3, n, 7)
        for i in range(self.num_levels):
            anchors = self.priors_base_sizes[i]
            gain[2:6] = torch.tensor(bbox_preds[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors featmap scales
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(
                    r, 1. / r).max(2)[0] < self.prior_match_thr  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < 0.5) & (gxy > 1.)).T
                l, m = ((gxi % 1. < 0.5) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + self.grid_offset)[j]
            else:
                t = targets[0]
                offsets = 0

            # Define batch_indx,cls_index
            batch_idx, _ = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            anchor_indx = t[:, 6].long()  # anchor indices m
            indices.append(
                (batch_idx, anchor_indx, gj.clamp_(0, gain[3] - 1),
                 gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            anch.append(anchors[anchor_indx])  # anchors

        matching_bs = [[] for _ in bbox_preds]
        matching_as = [[] for _ in bbox_preds]
        matching_gjs = [[] for _ in bbox_preds]
        matching_gis = [[] for _ in bbox_preds]
        matching_targets = [[] for _ in bbox_preds]
        matching_anchs = [[] for _ in bbox_preds]

        preds = []
        for bbox, obj, cls in zip(bbox_preds, objectnesses, cls_scores):
            b, c, h, w = bbox.shape
            bbox = bbox.reshape(b, 3, -1, h, w)
            obj = obj.reshape(b, 3, -1, h, w)
            cls = cls.reshape(b, 3, -1, h, w)
            pred = torch.cat([bbox, obj, cls], dim=2).permute(0, 1, 3, 4,
                                                              2).contiguous()
            preds.append(pred)

        # Each batch is calculated separately
        for batch_idx in range(preds[0].shape[0]):

            b_idx = batch_targets_normed[:, 0] == batch_idx
            this_target = batch_targets_normed[b_idx]
            if this_target.shape[0] == 0:
                continue

            txywh = this_target[:, 2:6] * 640
            txyxy = xywh2xyxy(txywh)

            pxyxys = []
            p_cls = []
            p_obj = []
            from_which_layer = []
            all_b = []
            all_a = []
            all_gj = []
            all_gi = []
            all_anch = []

            for i, pi in enumerate(preds):
                b, a, gj, gi = indices[i]
                idx = (b == batch_idx)
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append(torch.ones(size=(len(b),)) * i)
                # (n,85)
                fg_pred = pi[b, a, gj, gi]
                p_obj.append(fg_pred[:, 4:5])
                p_cls.append(fg_pred[:, 5:])

                grid = torch.stack([gi, gj], dim=1)
                pxy = (fg_pred[:, :2].sigmoid() * 2. - 0.5 +
                       grid) * self.featmap_strides[i]  # / 8.  + grid ? yolov5 don't
                # pxy = (fg_pred[:, :2].sigmoid() * 3. - 1. + grid) * self.stride[i]
                pwh = (fg_pred[:, 2:4].sigmoid() *
                       2) ** 2 * anch[i][idx] * self.featmap_strides[i]  # / 8.
                pxywh = torch.cat([pxy, pwh], dim=-1)
                pxyxy = xywh2xyxy(pxywh)
                pxyxys.append(pxyxy)

            pxyxys = torch.cat(pxyxys, dim=0)
            if pxyxys.shape[0] == 0:
                continue
            p_obj = torch.cat(p_obj, dim=0)
            p_cls = torch.cat(p_cls, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)
            all_b = torch.cat(all_b, dim=0)
            all_a = torch.cat(all_a, dim=0)
            all_gj = torch.cat(all_gj, dim=0)
            all_gi = torch.cat(all_gi, dim=0)
            all_anch = torch.cat(all_anch, dim=0)

            pair_wise_iou = box_iou(txyxy, pxyxys)

            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

            top_k, _ = torch.topk(
                pair_wise_iou, min(10, pair_wise_iou.shape[1]), dim=1)
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)

            gt_cls_per_image = (
                F.one_hot(this_target[:, 1].to(torch.int64),
                          80).float().unsqueeze(1).repeat(
                    1, pxyxys.shape[0], 1))

            num_gt = this_target.shape[0]
            cls_preds_ = (
                    p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_() *
                    p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_())

            y = cls_preds_.sqrt_()
            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
                torch.log(y / (1 - y)), gt_cls_per_image,
                reduction='none').sum(-1)
            del cls_preds_

            cost = (pair_wise_cls_loss + 3.0 * pair_wise_iou_loss)
            # num_gt,num_match_pred
            matching_matrix = torch.zeros_like(cost)

            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(
                    cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
                matching_matrix[gt_idx][pos_idx] = 1.0

            del top_k, dynamic_ks
            anchor_matching_gt = matching_matrix.sum(0) # num_match_pred, 每个预测框匹配的 gt 数目，如果有多个，则只能取代码最小的一个
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(
                    cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = matching_matrix.sum(0) > 0.0
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
            #经过 动态k 后选择的最终的正样本
            from_which_layer = from_which_layer[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anch = all_anch[fg_mask_inboxes]

            this_target = this_target[matched_gt_inds]

            for i in range(self.num_levels):
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(this_target[layer_idx])
                matching_anchs[i].append(all_anch[layer_idx])

        # 转换为预测层的格式输出，方便后续算loss
        for i in range(self.num_levels):
            if matching_targets[i] != []:
                matching_bs[i] = torch.cat(matching_bs[i], dim=0)
                matching_as[i] = torch.cat(matching_as[i], dim=0)
                matching_gjs[i] = torch.cat(matching_gjs[i], dim=0)
                matching_gis[i] = torch.cat(matching_gis[i], dim=0)
                matching_targets[i] = torch.cat(matching_targets[i], dim=0)
                matching_anchs[i] = torch.cat(matching_anchs[i], dim=0)
            else:
                matching_bs[i] = torch.tensor([],
                                              device='cuda:0',
                                              dtype=torch.int64)
                matching_as[i] = torch.tensor([],
                                              device='cuda:0',
                                              dtype=torch.int64)
                matching_gjs[i] = torch.tensor([],
                                               device='cuda:0',
                                               dtype=torch.int64)
                matching_gis[i] = torch.tensor([],
                                               device='cuda:0',
                                               dtype=torch.int64)
                matching_targets[i] = torch.tensor([],
                                                   device='cuda:0',
                                                   dtype=torch.int64)
                matching_anchs[i] = torch.tensor([],
                                                 device='cuda:0',
                                                 dtype=torch.int64)

        pre_gen_gains = [
            torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in preds
        ]

        # Losses
        for i, pi in enumerate(preds):  # layer index, layer predictions
            b, a, gj, gi = matching_bs[i], matching_as[i], matching_gjs[i], matching_gis[i]  # image, anchor, gridy, gridx
            anchors = matching_anchs[i]
            targets = matching_targets[i]
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                grid = torch.stack([gi, gj], dim=1)
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                # pxy = ps[:, :2].sigmoid() * 3. - 1.
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                selected_tbox = targets[:, 2:6] * pre_gen_gains[i]
                selected_tbox[:, :2] -= grid

                loss_box_i, iou = self.loss_bbox(pbox, selected_tbox)
                loss_box += loss_box_i

                # Objectness
                tobj[b, a, gj, gi] = iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                selected_tcls = targets[:, 1].long()
                if self.num_classes > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(
                        ps[:, 5:], 0., device=device)  # targets
                    t[range(n), selected_tcls] = 1.
                    loss_cls += self.loss_cls(ps[:, 5:], t)  # BCE

            obji = self.loss_obj(pi[..., 4], tobj)
            loss_obj += obji * self.obj_level_weights[i]  # obj loss

        bs = preds[0].shape[0]  # batch size
        _, world_size = get_dist_info()
        return dict(
            loss_cls=loss_cls * bs * world_size,
            loss_conf=loss_obj * bs * world_size,
            loss_bbox=loss_box * bs * world_size)

    def _convert_gt_to_norm_format(self,
                                   batch_gt_instances: Sequence[InstanceData],
                                   batch_img_metas: Sequence[dict]) -> Tensor:
        if isinstance(batch_gt_instances, torch.Tensor):
            # fast version
            img_shape = batch_img_metas[0]['batch_input_shape']
            gt_bboxes_xyxy = batch_gt_instances[:, 2:]
            xy1, xy2 = gt_bboxes_xyxy.split((2, 2), dim=-1)
            gt_bboxes_xywh = torch.cat([(xy2 + xy1) / 2, (xy2 - xy1)], dim=-1)
            gt_bboxes_xywh[:, 1::2] /= img_shape[0]
            gt_bboxes_xywh[:, 0::2] /= img_shape[1]
            batch_gt_instances[:, 2:] = gt_bboxes_xywh  # (num_bboxes, 6)
        else:
            # TODO
            batch_target_list = []
            # Convert xyxy bbox to yolo format.
            for i, gt_instances in enumerate(batch_gt_instances):
                img_shape = batch_img_metas[i]['batch_input_shape']
                bboxes = gt_instances.bboxes
                labels = gt_instances.labels

                xy1, xy2 = bboxes.split((2, 2), dim=-1)
                bboxes = torch.cat([(xy2 + xy1) / 2, (xy2 - xy1)], dim=-1)
                # normalized to 0-1
                bboxes[:, 1::2] /= img_shape[0]
                bboxes[:, 0::2] /= img_shape[1]

                index = bboxes.new_full((len(bboxes), 1), i)
                # (batch_idx, label, normed_bbox)
                target = torch.cat((index, labels[:, None].float(), bboxes),
                                   dim=1)
                batch_target_list.append(target)

            # (num_bboxes, 6)
            batch_gt_instances = torch.cat(
                batch_target_list, dim=0)

        return batch_gt_instances


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def bbox_iou(box1,
             box2,
             x1y1x2y2=True,
             GIoU=False,
             DIoU=False,
             CIoU=False,
             eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(
            b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) **
                    2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(
                    torch.atan(w2 / (h2 + eps)) - torch.atan(w1 /
                                                             (h1 + eps)), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
             torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter
                    )  # iou = inter / (area1 + area2 - inter)