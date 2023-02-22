import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import numpy as np
from mmpose.evaluation.functional.nms import oks_iou
from mmpose.datasets.datasets.utils import parse_pose_metainfo
from torch import Tensor
from mmyolo.datasets.utils import Keypoints
from mmyolo.registry import MODELS
from einops import rearrange, reduce, repeat


@MODELS.register_module()
class OksLoss(nn.Module):

    def __init__(self, dataset_info, oks_loss='oks', loss_weight=1.0) -> None:
        super().__init__()
        self.l2_loss = nn.MSELoss(reduction="none")
        if isinstance(dataset_info, dict):
            self.dataset_info = dataset_info
        if isinstance(dataset_info, str):
            _metainfo = dict(from_file=dataset_info)
            self.dataset_info = parse_pose_metainfo(_metainfo)
        else:
            raise TypeError('dataset_info must be a dict or a str')
        self.oks_type = oks_loss
        self.loss_weight = loss_weight

    def forward(self, kpt_preds: Tensor, kpt_targets: Tensor, kpt_mask: Tensor, bbox_targets: Tensor) -> Tensor:
        """
        forward preds and targets to calculate OKS loss.
        Implementation of OKS loss in https://arxiv.org/abs/2204.06806

        Args:
            kpt_preds(Tensor): predicted keypoints in shape N x k x 2, N: batch, K: num of keypoints, 2: xy coordinates.
            kpt_targets(Tensor): same shape as predicted keypoints.
            kpt_mask(Tensor): mask of valid keypoints in shape N x k, 1 for valid, 0 for invalid.
            bbox_targets(Tensor): bounding boxes in shape N x 4, 4: xyxy coordinates.
        """
        ns = kpt_targets.size(0)
        gt_bboxes_area = (bbox_targets[..., 2] - bbox_targets[..., 0]) * (bbox_targets[..., 3]-bbox_targets[..., 1]) * 0.53
        gt_bboxes_area = repeat(gt_bboxes_area, 'n -> n k', k=kpt_preds.size(1))
        kpt_sigmas = torch.tensor(self.dataset_info['sigmas'], device=kpt_targets.device, dtype=kpt_targets.dtype)
        kpt_sigmas = repeat(kpt_sigmas, 'k -> n k', n=ns)
        valid_kpt_preds = kpt_preds[kpt_mask]
        valid_kpt_targets = kpt_targets[kpt_mask]
        e = self.l2_loss(valid_kpt_preds, valid_kpt_targets)
        kn = kpt_sigmas[kpt_mask]
        s = (gt_bboxes_area[kpt_mask] + np.spacing(1)) / 2
        s = s.unsqueeze(1).repeat(1, 2)
        kn = kn.unsqueeze(1).repeat(1, 2)
        loss_kpt = (e/s) * kn
        loss_kpt = 1 - torch.exp(-1 * e/kn/s).mean()
        return loss_kpt * self.loss_weight

    def _oks(self,
             g: Tensor,
             d: Tensor,
             a_g: Tensor,
             a_d: Tensor,
             sigmas: list = None,
             vis_thr: float = 0) -> Tensor:
        """
        _oks calculate oks between ground truth and prediction.

        this oks followed by the original implementation in cocoAPI.

        Args:
            g (Tensor): ground truth keypoints with shape N x K x 3.
            d (Tensor): prediction keypoints with shape N x K x 3.
            a_g (Tensor): area of ground truth keypoints.
            a_d (Tensor): area of prediction keypoints.
            sigmas (list, optional): coco keypoints sigmas. Defaults to None.
            vis_thr (float, optional): visibility threshold. Defaults to 0.

        Returns:
            Tensor: oks value.
        """
        # corner case check
        assert g.shape == d.shape, f'g.shape: {g.shape}, d.shape: {d.shape}'
        assert g.dim() == 3, f'g.shape: {g.shape}'
        assert g.shape[1] == self.dataset_info[
            'num_keypoints'], f'g.shape: {g.shape}'
        assert a_g.shape == a_d.shape, f'a_g.shape: {a_g.shape}, a_d.shape: {a_d.shape}'
        assert a_g.dim() == 1, f'a_g.shape: {a_g.shape}'
        assert a_g.shape[0] == g.shape[
            0], f'a_g.shape: {a_g.shape}, g.shape: {g.shape}'
        assert sigmas is None or len(
            sigmas) == g.shape[1], f'sigmas: {sigmas}, g.shape: {g.shape}'
        assert vis_thr is None or (vis_thr >= 0
                                   and vis_thr <= 1), f'vis_thr: {vis_thr}'

        # get sigmas
        if sigmas is None:
            sigmas = self.dataset_info['sigmas']
        sigmas = torch.tensor(sigmas, device=g.device, dtype=g.dtype)

        kn_square = sigmas**2
        xg = g[..., 0]
        yg = g[..., 1]
        # vg = g[..., 2]
        xd = d[..., 0]
        yd = d[..., 1]
        # vd = d[..., 2]

        dx = xg - xd
        dy = yg - yd
        eps = 1e-6
        d_square = dx**2 + dy**2
        s_square = repeat(a_g, 'n -> n h', h=g.shape[1])**2
        e = d_square / (2 * s_square * kn_square + eps)
        return e
    
    def _oks2(self, g, d, a_g, a_d, sigmas, vis_thr=0):
        kpt_mask = (g[..., 2] > 0).float()
        kpt_sigmas = torch.tensor(sigmas, device=g.device, dtype=g.dtype)
        kpt_sigmas = kpt_sigmas.repeat(g.shape[0], 1)[kpt_mask == 1]
        s = a_g.repeat(g.shape[1], 1).transpose(0, 1)[kpt_mask == 1]
        valid_kpt_preds = d[kpt_mask == 1]
        valid_kpt_gts = g[kpt_mask == 1]

        num_valid_kpts = valid_kpt_preds.shape[0]
        if num_valid_kpts == 0:
            return 0
        # l2 distance
        e = (valid_kpt_preds - valid_kpt_gts)**2
        kn = 2 * kpt_sigmas ** 2
        s = s / 2
        oks = (1  - torch.exp(-1 * e / kn[..., None] / s[..., None])).mean()
        return oks