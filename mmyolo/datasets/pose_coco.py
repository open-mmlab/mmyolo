# Copyright (c) OpenMMLab. All rights reserved.
from itertools import filterfalse, groupby
from typing import Any, Dict, List

import numpy as np
from mmengine.dataset import force_full_init
from mmpose.datasets import CocoDataset as MMPoseCocoDataset

from ..registry import DATASETS


@DATASETS.register_module()
class CocoPoseDataset(MMPoseCocoDataset):

    METAINFO: dict = dict(from_file='configs/_base_/pose/coco.py')

    @force_full_init
    def prepare_data(self, idx) -> Any:
        data_info = self.get_data_info(idx)
        data_info['dataset'] = self
        return self.pipeline(data_info)

    def _get_bottomup_data_infos(self, instance_list: List[Dict],
                                 image_list: List[Dict]) -> List[Dict]:
        """Organize the data list in bottom-up mode."""
        # bottom-up data list
        data_list_bu = []

        used_img_ids = set()

        # group instances by img_id
        for img_id, data_infos in groupby(instance_list,
                                          lambda x: x['img_id']):
            used_img_ids.add(img_id)
            data_infos = list(data_infos)

            # image data
            img_path = data_infos[0]['img_path']
            data_info_bu = {
                'img_id': img_id,
                'img_path': img_path,
                'seg_map_path': None,
            }

            for key in data_infos[0].keys():
                if key not in data_info_bu:
                    seq = [d[key] for d in data_infos]
                    if isinstance(seq[0], np.ndarray):
                        seq = np.concatenate(seq, axis=0)
                    data_info_bu[key] = seq
                    # print(img_id,':',key)
                    # print(img_id,':',data_info_bu[key])

            instances = []
            for data_info in data_infos:
                instance = {}
                instance['ignore_flag'] = 0
                instance['bbox'] = data_info['bbox'][0]
                instance['bbox_label'] = data_info['category_id'] - 1
                instances.append(instance)

            data_info_bu['instances'] = instances

            # The segmentation annotation of invalid objects will be used
            # to generate valid region mask in the pipeline.
            invalid_segs = []
            for data_info_invalid in filterfalse(self._is_valid_instance,
                                                 data_infos):
                if 'segmentation' in data_info_invalid:
                    invalid_segs.append(data_info_invalid['segmentation'])
            data_info_bu['invalid_segs'] = invalid_segs

            data_list_bu.append(data_info_bu)
        # add images without instance for evaluation
        if self.test_mode:
            for img_info in image_list:
                if img_info['img_id'] not in used_img_ids:
                    data_info_bu = {
                        'img_id': img_info['img_id'],
                        'img_path': img_info['img_path'],
                        'id': list(),
                        'raw_ann_info': None,
                    }
                    data_list_bu.append(data_info_bu)

        return data_list_bu
