# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence, Union

import torch
from torch import Tensor


def gt_instances_preprocess(batch_gt_instances: Union[Tensor, Sequence],
                            batch_size: int) -> Tensor:
    """Split batch_gt_instances with batch size, from [all_gt_bboxes, 6] to.

    [batch_size, number_gt, 5]. If some shape of single batch smaller than
    gt bbox len, then using [-1., 0., 0., 0., 0.] to fill.

    Args:
        batch_gt_instances (Sequence[Tensor]): Ground truth
            instances for whole batch, shape [all_gt_bboxes, 6]
        batch_size (int): Batch size.

    Returns:
        Tensor: batch gt instances data, shape [batch_size, number_gt, 5]
    """
    if isinstance(batch_gt_instances, Sequence):
        max_gt_bbox_len = max(
            [len(gt_instances) for gt_instances in batch_gt_instances])
        # fill [-1., 0., 0., 0., 0.] if some shape of
        # single batch not equal max_gt_bbox_len
        batch_instance_list = []
        for index, gt_instance in enumerate(batch_gt_instances):
            bboxes = gt_instance.bboxes
            labels = gt_instance.labels
            batch_instance_list.append(
                torch.cat((labels[:, None], bboxes), dim=-1))

            if bboxes.shape[0] >= max_gt_bbox_len:
                continue

            fill_tensor = bboxes.new_full(
                [max_gt_bbox_len - bboxes.shape[0], 5], 0)
            fill_tensor[:, 0] = -1.
            batch_instance_list[index] = torch.cat(
                (batch_instance_list[-1], fill_tensor), dim=0)

        return torch.stack(batch_instance_list)
    else:
        # faster version
        # format of batch_gt_instances:
        # [img_ind, cls_ind, x1, y1, x2, y2]

        # sqlit batch gt instance [all_gt_bboxes, 6] ->
        # [batch_size, max_gt_bbox_len, 5]
        assert isinstance(batch_gt_instances, Tensor)
        if len(batch_gt_instances) > 0:
            gt_images_indexes = batch_gt_instances[:, 0]
            max_gt_bbox_len = gt_images_indexes.unique(
                return_counts=True)[1].max()
            batch_instance = torch.zeros((batch_size, max_gt_bbox_len, 5),
                                         dtype=batch_gt_instances.dtype,
                                         device=batch_gt_instances.device)
            # fill [-1., 0., 0., 0., 0.] if some shape of
            # single batch not equal max_gt_bbox_len
            batch_instance[:, :, 0] = -1.

            for i in range(batch_size):
                match_indexes = gt_images_indexes == i
                gt_num = match_indexes.sum()
                if gt_num:
                    batch_instance[i, :gt_num] = batch_gt_instances[
                        match_indexes, 1:]
        else:
            batch_instance = torch.zeros((batch_size, 0, 5),
                                         dtype=batch_gt_instances.dtype,
                                         device=batch_gt_instances.device)

        return batch_instance
