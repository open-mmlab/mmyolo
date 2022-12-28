# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence, Union

import torch
from mmdet.models.utils import unpack_gt_instances
from mmengine.structures import InstanceData
from torch import Tensor

from mmyolo.models import YOLOv5Head
from mmyolo.registry import MODELS


@MODELS.register_module()
class YOLOv5HeadAssigner(YOLOv5Head):

    def assign_by_gt_and_feat(
        self,
        batch_gt_instances: Sequence[InstanceData],
        batch_img_metas: Sequence[dict],
        inputs_hw: Union[Tensor, tuple] = (640, 640)
    ) -> dict:
        """Calculate the assigning results based on the gt and features
        extracted by the detection head.

        Args:
            batch_gt_instances (Sequence[InstanceData]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (Sequence[dict]): Meta information of each image,
                e.g., image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            inputs_hw (Union[Tensor, tuple]): Height and width of inputs size.
        Returns:
            dict[str, Tensor]: A dictionary of assigning results.
        """
        # 1. Convert gt to norm format
        batch_targets_normed = self._convert_gt_to_norm_format(
            batch_gt_instances, batch_img_metas)

        device = batch_targets_normed.device
        scaled_factor = torch.ones(7, device=device)
        gt_inds = torch.arange(
            batch_targets_normed.shape[1],
            dtype=torch.long,
            device=device,
            requires_grad=False).unsqueeze(0).repeat((self.num_base_priors, 1))

        assign_results = []
        for i in range(self.num_levels):
            assign_results_feat = []
            h = inputs_hw[0] // self.featmap_strides[i]
            w = inputs_hw[1] // self.featmap_strides[i]

            # empty gt bboxes
            if batch_targets_normed.shape[1] == 0:
                for k in range(self.num_base_priors):
                    assign_results_feat.append({
                        'stride':
                        self.featmap_strides[i],
                        'grid_x_inds':
                        torch.zeros([0], dtype=torch.int64).to(device),
                        'grid_y_inds':
                        torch.zeros([0], dtype=torch.int64).to(device),
                        'img_inds':
                        torch.zeros([0], dtype=torch.int64).to(device),
                        'class_inds':
                        torch.zeros([0], dtype=torch.int64).to(device),
                        'retained_gt_inds':
                        torch.zeros([0], dtype=torch.int64).to(device),
                        'prior_ind':
                        k
                    })
                assign_results.append(assign_results_feat)
                continue

            priors_base_sizes_i = self.priors_base_sizes[i]
            # feature map scale whwh
            scaled_factor[2:6] = torch.tensor([w, h, w, h])
            # Scale batch_targets from range 0-1 to range 0-features_maps size.
            # (num_base_priors, num_bboxes, 7)
            batch_targets_scaled = batch_targets_normed * scaled_factor

            # 2. Shape match
            wh_ratio = batch_targets_scaled[...,
                                            4:6] / priors_base_sizes_i[:, None]
            match_inds = torch.max(
                wh_ratio, 1 / wh_ratio).max(2)[0] < self.prior_match_thr
            batch_targets_scaled = batch_targets_scaled[match_inds]
            match_gt_inds = gt_inds[match_inds]

            # no gt bbox matches anchor
            if batch_targets_scaled.shape[0] == 0:
                for k in range(self.num_base_priors):
                    assign_results_feat.append({
                        'stride':
                        self.featmap_strides[i],
                        'grid_x_inds':
                        torch.zeros([0], dtype=torch.int64).to(device),
                        'grid_y_inds':
                        torch.zeros([0], dtype=torch.int64).to(device),
                        'img_inds':
                        torch.zeros([0], dtype=torch.int64).to(device),
                        'class_inds':
                        torch.zeros([0], dtype=torch.int64).to(device),
                        'retained_gt_inds':
                        torch.zeros([0], dtype=torch.int64).to(device),
                        'prior_ind':
                        k
                    })
                assign_results.append(assign_results_feat)
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
            retained_gt_inds = match_gt_inds.repeat((5, 1))[offset_inds]
            retained_offsets = self.grid_offset.repeat(1, offset_inds.shape[1],
                                                       1)[offset_inds]

            # prepare pred results and positive sample indexes to
            # calculate class loss and bbox lo
            _chunk_targets = batch_targets_scaled.chunk(4, 1)
            img_class_inds, grid_xy, grid_wh, priors_inds = _chunk_targets
            priors_inds, (img_inds, class_inds) = priors_inds.long().view(
                -1), img_class_inds.long().T

            grid_xy_long = (grid_xy -
                            retained_offsets * self.near_neighbor_thr).long()
            grid_x_inds, grid_y_inds = grid_xy_long.T
            for k in range(self.num_base_priors):
                retained_inds = priors_inds == k
                assign_results_prior = {
                    'stride': self.featmap_strides[i],
                    'grid_x_inds': grid_x_inds[retained_inds],
                    'grid_y_inds': grid_y_inds[retained_inds],
                    'img_inds': img_inds[retained_inds],
                    'class_inds': class_inds[retained_inds],
                    'retained_gt_inds': retained_gt_inds[retained_inds],
                    'prior_ind': k
                }
                assign_results_feat.append(assign_results_prior)
            assign_results.append(assign_results_feat)
        return assign_results

    def assign(self, batch_data_samples: Union[list, dict],
               inputs_hw: Union[tuple, torch.Size]) -> dict:
        """Calculate assigning results. This function is provided to the
        `assigner_visualization.py` script.

        Args:
            batch_data_samples (List[:obj:`DetDataSample`], dict): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            inputs_hw: Height and width of inputs size

        Returns:
            dict: A dictionary of assigning components.
        """
        if isinstance(batch_data_samples, list):
            outputs = unpack_gt_instances(batch_data_samples)
            (batch_gt_instances, batch_gt_instances_ignore,
             batch_img_metas) = outputs

            assign_inputs = (batch_gt_instances, batch_img_metas,
                             batch_gt_instances_ignore, inputs_hw)
        else:
            # Fast version
            assign_inputs = (batch_data_samples['bboxes_labels'],
                             batch_data_samples['img_metas'], inputs_hw)
        assign_results = self.assign_by_gt_and_feat(*assign_inputs)

        return assign_results
