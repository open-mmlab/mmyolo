# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, List, Optional

from mmdet.datasets import CocoDataset

from ..registry import DATASETS, TASK_UTILS


@DATASETS.register_module()
class YOLOv5CocoDataset(CocoDataset):
    """Dataset for YOLOv5 COCO Dataset.

    We only add `BatchShapePolicy` function compared with CocoDataset. See
    `mmyolo/datasets/utils.py#BatchShapePolicy` for details
    """

    def __init__(self,
                 *args,
                 batch_shapes_cfg: Optional[dict] = None,
                 **kwargs):

        self.batch_shapes_cfg = batch_shapes_cfg
        super().__init__(*args, **kwargs)

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


@DATASETS.register_module()
class PPYOLOECocoDataset(CocoDataset):

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
            if filter_empty_gt:
                min_size = self.filter_cfg.get('min_size', 0)

                # obtain images that contain annotation
                ids_with_ann = {
                    data_info['img_id']
                    for data_info in self.data_list
                }
                # obtain images that contain annotations of the
                # required categories
                ids_in_cat = set()
                for i, class_id in enumerate(self.cat_ids):
                    ids_in_cat |= set(self.cat_img_map[class_id])
                # merge the image id sets of the two conditions and
                # use the merged set to filter out images
                # if self.filter_empty_gt=True
                ids_in_cat &= ids_with_ann

                valid_data_infos = []
                for i, data_info in enumerate(self.data_list):
                    img_id = data_info['img_id']
                    width = data_info['width']
                    height = data_info['height']
                    if filter_empty_gt and img_id not in ids_in_cat:
                        continue
                    if min(width, height) >= min_size:
                        valid_data_infos.append(data_info)
            else:
                return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        # obtain images that contain annotation
        ids_with_ann = {data_info['img_id'] for data_info in self.data_list}
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            img_id = data_info['img_id']
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos
