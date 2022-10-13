# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from mmdet.datasets import CocoDataset

from ..registry import DATASETS


@DATASETS.register_module()
class PPYOLOECocoDataset(CocoDataset):
    """Dataset for PPYOLOE COCO Dataset.

    While test in ppyoloe, it do not use img with no gt.
    """

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
