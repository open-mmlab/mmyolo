# Copyright (c) OpenMMLab. All rights reserved.

from mmyolo.datasets.yolov5_coco import BatchShapePolicyDataset
from ..registry import DATASETS

try:
    from mmrotate.datasets import DOTADataset
    MMROTATE_AVAILABLE = True
except ImportError:
    from mmengine.dataset import BaseDataset
    DOTADataset = BaseDataset
    MMROTATE_AVAILABLE = False


@DATASETS.register_module()
class YOLOv5DOTADataset(BatchShapePolicyDataset, DOTADataset):
    """Dataset for YOLOv5 DOTA Dataset.

    We only add `BatchShapePolicy` function compared with DOTADataset. See
    `mmyolo/datasets/utils.py#BatchShapePolicy` for details
    """

    def __init__(self, *args, **kwargs):
        if not MMROTATE_AVAILABLE:
            raise ImportError(
                'Please run "mim install -r requirements/mmrotate.txt" '
                'to install mmrotate first for rotated detection.')

        super().__init__(*args, **kwargs)
