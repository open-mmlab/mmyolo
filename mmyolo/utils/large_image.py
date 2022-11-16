# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence, Tuple

from mmcv.ops import batched_nms
from mmdet.structures import DetDataSample, SampleList
from mmengine.structures import InstanceData


def merge_results_by_nms(results: SampleList, offsets: Sequence[Tuple[int,
                                                                      int]],
                         full_shape: Tuple[int, int],
                         nms_cfg: dict) -> DetDataSample:
    """Merge patch results by nms.

    Args:
        results (List[:obj:`DetDataSample`]): A list of patches results.
        offsets (Sequence[Tuple[int, int]]): Positions of the left top points
            of patches.
        full_shape (Tuple[int, int]): A (height, width) tuple of the large
            image's width and height.
        nms_cfg (dict): it should specify nms type and other parameters
            like `iou_threshold`.
    Retunrns:
        :obj:`DetDataSample`: merged results.
    """
    from sahi.slicing import shift_bboxes, shift_masks

    assert len(results) == len(
        offsets), 'The `results` should has the ' 'same length with `offsets`.'
    pred_instances = []
    for result, offset in zip(results, offsets):
        pred_inst = result.pred_instances
        pred_inst.bboxes = shift_bboxes(pred_inst.bboxes, offset)
        if 'masks' in result:
            pred_inst.masks = shift_masks(pred_inst.masks, offset, full_shape)
        pred_instances.append(pred_inst)

    instances = InstanceData.cat(pred_instances)
    _, keeps = batched_nms(
        boxes=instances.bboxes,
        scores=instances.scores,
        idxs=instances.labels,
        nms_cfg=nms_cfg)
    merged_instances = instances[keeps]

    merged_result = DetDataSample()
    # update items like gt_instances, ignore_instances
    merged_result.update(results[0])
    merged_result.pred_instances = merged_instances
    return merged_result
