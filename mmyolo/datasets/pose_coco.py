# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any

from mmengine.dataset import force_full_init

try:
    from mmpose.datasets import CocoDataset as MMPoseCocoDataset
except ImportError:
    MMPoseCocoDataset = object

from ..registry import DATASETS


@DATASETS.register_module()
class PoseCocoDataset(MMPoseCocoDataset):

    METAINFO: dict = dict(from_file='configs/_base_/pose/coco.py')

    def __init__(self, *args, **kwargs):
        if MMPoseCocoDataset is object:
            raise ImportError(
                'Please run "mim install -r requirements/mmpose.txt" '
                'to install mmpose first for PoseCocoDataset.')
        super().__init__(*args, **kwargs)

    @force_full_init
    def prepare_data(self, idx) -> Any:
        data_info = self.get_data_info(idx)
        data_info['dataset'] = self
        return self.pipeline(data_info)
