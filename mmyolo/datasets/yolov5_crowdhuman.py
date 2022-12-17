# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.datasets import CrowdHumanDataset

from ..registry import DATASETS
from .yolov5_coco import BatchShapePolicyDataset


@DATASETS.register_module()
class YOLOv5CrowdHumanDataset(BatchShapePolicyDataset, CrowdHumanDataset):
    """Dataset for YOLOv5 CrowdHuman Dataset.

    We only add `BatchShapePolicy` function compared with CrowdHumanDataset.
    See `mmyolo/datasets/utils.py#BatchShapePolicy` for details
    """
    pass
