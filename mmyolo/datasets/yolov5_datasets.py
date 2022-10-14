# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.datasets import VOCDataset

from ..registry import DATASETS
from .yolov5_coco import BatchShapePolicyDataset


@DATASETS.register_module()
class YOLOv5VOCDataset(BatchShapePolicyDataset, VOCDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: Because of the bug in mmengine `ConcatDataset`
        # add this to avoid ValueError
        if 'VOC2007' in self.sub_data_root:
            self._metainfo['DATASET_TYPE'] = 'VOC'
        elif 'VOC2012' in self.sub_data_root:
            self._metainfo['DATASET_TYPE'] = 'VOC'
        else:
            self._metainfo['DATASET_TYPE'] = None
