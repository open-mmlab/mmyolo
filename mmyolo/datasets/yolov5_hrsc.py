# Copyright (c) OpenMMLab. All rights reserved.
from mmrotate.datasets import HRSCDataset

from mmyolo.datasets.yolov5_coco import BatchShapePolicyDataset
from ..registry import DATASETS


@DATASETS.register_module()
class YOLOv5HRSCDataset(BatchShapePolicyDataset, HRSCDataset):
    """Dataset for YOLOv5 HRSC Dataset.

    We only add `BatchShapePolicy` function compared with HRSCDataset. See
    `mmyolo/datasets/utils.py#BatchShapePolicy` for details
    """
    pass
