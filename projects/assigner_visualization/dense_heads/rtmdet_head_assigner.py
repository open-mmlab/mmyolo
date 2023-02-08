# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
from mmdet.structures.bbox import distance2bbox
from mmdet.utils import InstanceList
from torch import Tensor

from mmyolo.models import RTMDetHead
from mmyolo.registry import MODELS


@MODELS.register_module()
class RTMHeadAssigner(RTMDetHead):

    def assign_by_gt_and_feat(self, cls_scores: List[Tensor],
                              bbox_preds: List[Tensor],
                              batch_gt_instances: InstanceList,
                              batch_img_metas: List[dict]) -> dict:
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
        num_imgs = len(batch_img_metas)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        gt_info = self.gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels = gt_info[:, :, :1]
        gt_bboxes = gt_info[:, :, 1:]  # xyxy
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        device = cls_scores[0].device

        # If the shape does not equal, generate new one
        if featmap_sizes != self.featmap_sizes_train:
            self.featmap_sizes_train = featmap_sizes
            mlvl_priors_with_stride = self.prior_generator.grid_priors(
                featmap_sizes, device=device, with_stride=True)
            self.flatten_priors_train = torch.cat(
                mlvl_priors_with_stride, dim=0)

        flatten_cls_scores = torch.cat([
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.cls_out_channels)
            for cls_score in cls_scores
        ], 1).contiguous()

        flatten_bboxes = torch.cat([
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ], 1)
        flatten_bboxes = flatten_bboxes * self.flatten_priors_train[..., -1,
                                                                    None]
        flatten_bboxes = distance2bbox(self.flatten_priors_train[..., :2],
                                       flatten_bboxes)

        assigned_result = self.assigner(flatten_bboxes.detach(),
                                        flatten_cls_scores.detach(),
                                        self.flatten_priors_train, gt_labels,
                                        gt_bboxes, pad_bbox_flag)

        labels = assigned_result['assigned_labels'].reshape(-1)
        matched_gt_inds = assigned_result['matched_gt_inds']
        bbox_preds = flatten_bboxes.reshape(-1, 4)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)

        level_inds = torch.zeros_like(labels)
        img_inds = torch.zeros_like(labels)
        level_nums = [0] + [f[0] * f[1] for f in featmap_sizes]
        for i in range(len(level_nums) - 1):
            level_nums[i + 1] = level_nums[i] + level_nums[i + 1]
            level_inds[level_nums[i]:level_nums[i + 1]] = i
        level_inds_pos = level_inds[pos_inds]

        img_inds = img_inds[pos_inds]
        labels = labels[pos_inds]

        inputs_hw = batch_img_metas[0]['batch_input_shape']
        assign_results = []
        for i in range(self.num_levels):
            retained_inds = level_inds_pos == i
            if not retained_inds.any():
                assign_results_prior = {
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
                    1
                }
            else:
                w = inputs_hw[1] // self.featmap_strides[i]

                retained_pos_inds = pos_inds[retained_inds] - level_nums[i]
                grid_y_inds = retained_pos_inds // w
                grid_x_inds = retained_pos_inds - retained_pos_inds // w * w
                assign_results_prior = {
                    'stride': self.featmap_strides[i],
                    'grid_x_inds': grid_x_inds,
                    'grid_y_inds': grid_y_inds,
                    'img_inds': img_inds[retained_inds],
                    'class_inds': labels[retained_inds],
                    'retained_gt_inds': matched_gt_inds[retained_inds],
                    'prior_ind': 1
                }
            assign_results.append([assign_results_prior])
        return assign_results

    def assign(self, x: List[Tensor], batch_gt_instances: InstanceList,
               batch_img_metas: List[dict]) -> dict:
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
        if isinstance(x, list):
            # TODO: fix here
            raise NotImplementedError
        else:
            # Fast version
            cls_scores, bbox_preds = self(x)
            assign_inputs = (cls_scores, bbox_preds, batch_gt_instances,
                             batch_img_metas)
        assign_results = self.assign_by_gt_and_feat(*assign_inputs)
        return assign_results
