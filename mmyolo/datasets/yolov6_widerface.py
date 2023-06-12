# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.datasets import WIDERFaceDataset

from mmyolo.datasets.yolov5_coco import BatchShapePolicyDataset
from ..registry import DATASETS


@DATASETS.register_module()
class YOLOv6WIDERFaceDataset(BatchShapePolicyDataset, WIDERFaceDataset):
    """Dataset for YOLOv6 WIDERFace Dataset."""
    pass
