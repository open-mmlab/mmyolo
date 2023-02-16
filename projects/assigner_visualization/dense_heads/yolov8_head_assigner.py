# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

import torch
from mmdet.utils import InstanceList
from torch import Tensor

from mmyolo.models import YOLOv8Head
from mmyolo.models.utils import gt_instances_preprocess
from mmyolo.registry import MODELS


@MODELS.register_module()
class YOLOv8HeadAssigner(YOLOv8Head):

    def assign_by_gt_and_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        inputs_hw: Union[Tensor, tuple] = (640, 640)
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
            bbox_dist_preds (Sequence[Tensor]): Box distribution logits for
                each scale level with shape (bs, reg_max + 1, H*W, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            inputs_hw (Union[Tensor, tuple]): Height and width of inputs size.
        Returns:
            dict[str, Tensor]: A dictionary of assigning results.
        """
        num_imgs = len(batch_img_metas)
        device = cls_scores[0].device

        current_featmap_sizes = [
            cls_score.shape[2:] for cls_score in cls_scores
        ]
        # If the shape does not equal, generate new one
        if current_featmap_sizes != self.featmap_sizes_train:
            self.featmap_sizes_train = current_featmap_sizes

            mlvl_priors_with_stride = self.prior_generator.grid_priors(
                self.featmap_sizes_train,
                dtype=cls_scores[0].dtype,
                device=device,
                with_stride=True)

            self.num_level_priors = [len(n) for n in mlvl_priors_with_stride]
            self.flatten_priors_train = torch.cat(
                mlvl_priors_with_stride, dim=0)
            self.stride_tensor = self.flatten_priors_train[..., [2]]

        # gt info
        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
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
        # (bs, n, 4 * reg_max)

        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_pred_bboxes = torch.cat(flatten_pred_bboxes, dim=1)
        flatten_pred_bboxes = self.bbox_coder.decode(
            self.flatten_priors_train[..., :2], flatten_pred_bboxes,
            self.stride_tensor[..., 0])

        assigned_result = self.assigner(
            (flatten_pred_bboxes.detach()).type(gt_bboxes.dtype),
            flatten_cls_preds.detach().sigmoid(), self.flatten_priors_train,
            gt_labels, gt_bboxes, pad_bbox_flag)

        labels = assigned_result['assigned_labels'].reshape(-1)
        bbox_targets = assigned_result['assigned_bboxes'].reshape(-1, 4)
        fg_mask_pre_prior = assigned_result['fg_mask_pre_prior'].squeeze(0)

        pos_inds = fg_mask_pre_prior.nonzero().squeeze(1)

        targets = bbox_targets[pos_inds]
        gt_bboxes = gt_bboxes.squeeze(0)
        matched_gt_inds = torch.tensor(
            [((t == gt_bboxes).sum(dim=1) == t.shape[0]).nonzero()[0]
             for t in targets],
            device=device)

        level_inds = torch.zeros_like(labels)
        img_inds = torch.zeros_like(labels)
        level_nums = [0] + self.num_level_priors
        for i in range(len(level_nums) - 1):
            level_nums[i + 1] = level_nums[i] + level_nums[i + 1]
            level_inds[level_nums[i]:level_nums[i + 1]] = i
        level_inds_pos = level_inds[pos_inds]

        img_inds = img_inds[pos_inds]
        labels = labels[pos_inds]

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
                    0
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
                    'prior_ind': 0
                }
            assign_results.append([assign_results_prior])
        return assign_results

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
            cls_scores, bbox_preds = self(batch_data_samples['feats'])
            assign_inputs = (cls_scores, bbox_preds,
                             batch_data_samples['bboxes_labels'],
                             batch_data_samples['img_metas'], inputs_hw)
        assign_results = self.assign_by_gt_and_feat(*assign_inputs)
        return assign_results
