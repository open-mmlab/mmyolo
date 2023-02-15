import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from mmpose.evaluation.functional.nms import oks_iou
from mmpose.datasets.datasets.utils import parse_pose_metainfo
from torch import Tensor
from mmyolo.datasets.utils import Keypoints
from mmyolo.registry import MODELS
from einops import rearrange, reduce, repeat


@MODELS.register_module()
class OksLoss(nn.Module):
    def __init__(self, dataset_info) -> None:
        super().__init__()
        if isinstance(dataset_info, dict):
            self.dataset_info = dataset_info
        if isinstance(dataset_info, str):
            _metainfo = dict(from_file=dataset_info)
            self.dataset_info = parse_pose_metainfo(_metainfo)
        else:
            raise TypeError('dataset_info must be a dict or a str')

    def forward(self, preds:Tensor, targets:Tensor) -> Tensor:
        """
        forward preds and targets to calculate OKS loss.

        Implementation of OKS loss in https://arxiv.org/abs/2204.06806

        Args:
            preds (Tensor): keypoints tensor with shape N x K x 2, where N is
                the batch size, K is the number of keypoints, 2 is the keypoint
                coordinates.
            targets (Tensor): ground truth keypoints tensor with shape N x K x 2.

        Returns:
            Tensor: loss value.
        """
        assert preds.shape == targets.shape
        assert preds.dim() == 3 and preds.shape[-1] == 2
        assert preds.shape[1] == self.dataset_info['num_keypoints']

        # area
        a_g = Keypoints._kpt_area(targets)
        a_p = Keypoints._kpt_area(preds)

        # calculate oks
        oks_loss = self._oks(targets, preds, a_g, a_p, self.dataset_info['sigmas'])
        return oks_loss

    def _oks(self, g:Tensor, d:Tensor, a_g:Tensor, a_d:Tensor, sigmas:list=None, vis_thr:float=None)->Tensor:
        """
        _oks calculate oks between ground truth and prediction.

        this oks followed by the original implementation in cocoAPI.

        Args:
            g (Tensor): ground truth keypoints with shape N x K x 3.
            d (Tensor): prediction keypoints with shape N x K x 3.
            a_g (Tensor): area of ground truth keypoints.
            a_d (Tensor): area of prediction keypoints.
            sigmas (list, optional): coco keypoints sigmas. Defaults to None.
            vis_thr (float, optional): visibility threshold. Defaults to None.

        Returns:
            Tensor: oks value.
        """
        # corner case check
        assert g.shape == d.shape, f'g.shape: {g.shape}, d.shape: {d.shape}'
        assert g.dim() == 3 and g.shape[-1] == 2, f'g.shape: {g.shape}'
        assert g.shape[1] == self.dataset_info['num_keypoints'], f'g.shape: {g.shape}'
        assert a_g.shape == a_d.shape, f'a_g.shape: {a_g.shape}, a_d.shape: {a_d.shape}'
        assert a_g.dim() == 1, f'a_g.shape: {a_g.shape}'
        assert a_g.shape[0] == g.shape[0], f'a_g.shape: {a_g.shape}, g.shape: {g.shape}'
        assert sigmas is None or len(sigmas) == g.shape[1], f'sigmas: {sigmas}, g.shape: {g.shape}'
        assert vis_thr is None or (vis_thr >= 0 and vis_thr <= 1), f'vis_thr: {vis_thr}'

        # get sigmas
        if sigmas is None:
            sigmas = self.dataset_info['sigmas']
        sigmas = torch.tensor(sigmas, device=g.device, dtype=g.dtype)

        vars = (sigmas * 2)**2
        xg = g[..., 0]
        yg = g[..., 1]
        # vg = g[..., 2]
        xd = d[..., 0]
        yd = d[..., 1]
        # vd = d[..., 2]

        dx = xg - xd
        dy = yg - yd
        eps = 1e-6
        d = dx ** 2 + dy ** 2
        s = repeat(a_g, 'n -> n h', h=g.shape[1])
        e = (-d / (s * vars + eps))
        loss = 1 - torch.exp(e)
        return loss
