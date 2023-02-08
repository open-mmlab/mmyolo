# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, List, Optional

from mmdet.datasets import BaseDetDataset, CocoDataset

from ..registry import DATASETS, TASK_UTILS


class BatchShapePolicyDataset(BaseDetDataset):
    """Dataset with the batch shape policy that makes paddings with least
    pixels during batch inference process, which does not require the image
    scales of all batches to be the same throughout validation."""

    def __init__(self,
                 *args,
                 batch_shapes_cfg: Optional[dict] = None,
                 **kwargs):
        self.batch_shapes_cfg = batch_shapes_cfg
        super().__init__(*args, **kwargs)

    def full_init(self):
        """rewrite full_init() to be compatible with serialize_data in
        BatchShapePolicy."""
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
class YOLOv5CocoDataset(BatchShapePolicyDataset, CocoDataset):
    """Dataset for YOLOv5 COCO Dataset.

    We only add `BatchShapePolicy` function compared with CocoDataset. See
    `mmyolo/datasets/utils.py#BatchShapePolicy` for details
    """
    pass


@DATASETS.register_module()
class PPYOLOECocoDataset(YOLOv5CocoDataset):

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """

        pass
        # 过滤宽或高小于1e-5的
        eps = 1e-5
        for i in self.data_list:
            instances = i['instances']
            filter_instances = []
            for instance in instances:
                bbox = instance['bbox']
                x1, y1, x2, y2 = bbox
                if (x2 - x1 > eps) and (y2 - y1 > eps):
                    filter_instances.append(instance)
                else:
                    print('filter', x1, y1, x2, y2)
            i['instances'] = filter_instances

        filter_data_list = []
        # 过滤没有gt的图
        for i in self.data_list:
            instances = i['instances']
            ignore_flag_list = [k['ignore_flag'] for k in instances]
            # 没有gtbbox的图过滤掉
            if len(instances) == 0:
                print('filter no gt img', i['img_id'], self.test_mode)
                continue
            # 如果一个图里的gt_bbox都ignore，也过滤
            if sum(ignore_flag_list) == len(instances):
                print('filter all bboxes are ignore img', i['img_id'])
                continue
            filter_data_list.append(i)

        return filter_data_list
