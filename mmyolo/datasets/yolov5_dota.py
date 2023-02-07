# Copyright (c) OpenMMLab. All rights reserved.
from mmrotate.datasets import DOTADataset

from mmyolo.datasets.yolov5_coco import BatchShapePolicyDataset
from ..registry import DATASETS


@DATASETS.register_module()
class YOLOv5DOTADataset(BatchShapePolicyDataset, DOTADataset):
    """Dataset for YOLOv5 DOTA Dataset.

    We only add `BatchShapePolicy` function compared with DOTADataset. See
    `mmyolo/datasets/utils.py#BatchShapePolicy` for details
    """
    pass
