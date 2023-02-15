# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

import torch
from mmdet.utils import InstanceList
from torch import Tensor

from mmyolo.models import YOLOv7Head
from mmyolo.registry import MODELS


@MODELS.register_module()
class YOLOv7HeadAssigner(YOLOv7Head):

    def assign_by_gt_and_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        objectnesses: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        inputs_hw: Union[Tensor, tuple],
    ) -> dict:
        """Calculate the assigning results based on the gt and features
        extracted by the detection head.
        Args:
            cls_scores (Sequence[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (Sequence[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            objectnesses (Sequence[Tensor]): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W)
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            inputs_hw (Union[Tensor, tuple]): Height and width of inputs size.
        Returns:
            dict[str, Tensor]: A dictionary of assigning results.
        """
        device = cls_scores[0][0].device

        head_preds = self._merge_predict_results(bbox_preds, objectnesses,
                                                 cls_scores)

        batch_targets_normed = self._convert_gt_to_norm_format(
            batch_gt_instances, batch_img_metas)

        # yolov5_assign and simota_assign
        assigner_results = self.assigner(
            head_preds,
            batch_targets_normed,
            batch_img_metas[0]['batch_input_shape'],
            self.priors_base_sizes,
            self.grid_offset,
            near_neighbor_thr=self.near_neighbor_thr)

        # multi-level positive sample position.
        mlvl_positive_infos = assigner_results['mlvl_positive_infos']
        # assigned results with label and bboxes information.
        mlvl_targets_normed = assigner_results['mlvl_targets_normed']

        assign_results = []
        for i in range(self.num_levels):
            assign_results_feat = []
            # no gt bbox matches anchor
            if mlvl_positive_infos[i].shape[0] == 0:
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

            # (batch_idx, prior_idx, x_scaled, y_scaled)
            positive_info = mlvl_positive_infos[i]
            targets_normed = mlvl_targets_normed[i]
            priors_inds = positive_info[:, 1]
            grid_x_inds = positive_info[:, 2]
            grid_y_inds = positive_info[:, 3]
            img_inds = targets_normed[:, 0]
            class_inds = targets_normed[:, 1].long()
            retained_gt_inds = self.get_gt_inds(
                targets_normed, batch_targets_normed[0]).long()
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

    def get_gt_inds(self, assigned_target, gt_instance):
        """Judging which one gt_ind is assigned by comparing assign_target and
        origin target.

        Args:
           assigned_target (Tensor(assign_nums,7)): YOLOv7 assigning results.
           gt_instance (Tensor(gt_nums,7)):  Normalized gt_instance, It
                usually includes ``bboxes`` and ``labels`` attributes.
        Returns:
           gt_inds (Tensor): the index which one gt is assigned.
        """
        gt_inds = torch.zeros(assigned_target.shape[0])
        for i in range(assigned_target.shape[0]):
            gt_inds[i] = ((assigned_target[i] == gt_instance).sum(
                dim=1) == 7).nonzero().squeeze()
        return gt_inds

    def assign(self, batch_data_samples: Union[list, dict],
               inputs_hw: Union[tuple, torch.Size]) -> dict:
        """Calculate assigning results.

        This function is provided to the
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
            raise NotImplementedError(
                'assigning results_list is not implemented')
        else:
            # Fast version
            cls_scores, bbox_preds, objectnesses = self(
                batch_data_samples['feats'])
            assign_inputs = (cls_scores, bbox_preds, objectnesses,
                             batch_data_samples['bboxes_labels'],
                             batch_data_samples['img_metas'], inputs_hw)
        assign_results = self.assign_by_gt_and_feat(*assign_inputs)
        return assign_results
