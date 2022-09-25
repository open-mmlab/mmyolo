# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Any, Optional

from mmdet.datasets import VOCDataset
from ..registry import DATASETS, TASK_UTILS


@DATASETS.register_module()
class YOLOv5VOCDataset(VOCDataset):
    """Dataset for YOLOv5 VOC Dataset.

    We only add `BatchShapePolicy` function compared with VOCDataset. See
    `mmyolo/datasets/utils.py#BatchShapePolicy` for details
    """

    def __init__(self,
                 *args,
                 batch_shapes_cfg: Optional[dict] = None,
                 **kwargs):

        self.batch_shapes_cfg = batch_shapes_cfg
        super().__init__(*args, **kwargs)
        if 'VOC2007' in self.sub_data_root:
            self._metainfo['DATASET_TYPE'] = 'VOC'
        elif 'VOC2012' in self.sub_data_root:
            self._metainfo['DATASET_TYPE'] = 'VOC'
        else:
            self._metainfo['DATASET_TYPE'] = None

    def full_init(self):
        """rewrite full_init() to be compatible with serialize_data in
        BatchShapesPolicy."""
        if self._fully_initialized:
            return
        # load data information
        self.data_list = self.load_data_list()

        # batch_shapes_cfg
        if self.batch_shapes_cfg:
            batch_shapes_policy = TASK_UTILS.build(self.batch_shapes_cfg)
            self.data_list = batch_shapes_policy(self.data_list)
            del batch_shapes_policy

        # filter illegal data, such as data that has no annotations.
        self.data_list = self.filter_data()
        # Get subset data according to indices.
        if self._indices is not None:
            self.data_list = self._get_unserialized_subset(self._indices)

        # serialize data_list
        if self.serialize_data:
            self.data_bytes, self.data_address = self._serialize_data()

        self._fully_initialized = True

    def prepare_data(self, idx: int) -> Any:
        """Pass the dataset to the pipeline during training to support mixed
        data augmentation, such as Mosaic and MixUp."""
        if self.test_mode is False:
            data_info = self.get_data_info(idx)
            data_info['dataset'] = self
            return self.pipeline(data_info)
        else:
            return super().prepare_data(idx)
