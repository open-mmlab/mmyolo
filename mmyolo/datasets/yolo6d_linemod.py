# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.datasets import CocoDataset

from mmyolo.registry import DATASETS


@DATASETS.register_module()
class YOLO6dDataset(CocoDataset):
    """Dataset for YOLO6d."""

    METAINFO = {
        'dataset_type':
        'linemod_preprocessed',
        'task_name':
        'PoseEstimation',
        'CLASSES':
        ('ape', 'benchvise', 'bowl', 'cam', 'can', 'cat', 'cup', 'driller',
         'duck', 'eggbox', 'glue', 'holepuncher', 'iron', 'lamp', 'phone'),
        'PALETTE': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
                    (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
                    (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
                    (175, 116, 175), (200, 123, 23), (80, 160, 200)]
    }

    # def load_data_list(self) -> List[dict]:
    #     """Load annotations from an annotation
    #        file named as ``self.ann_file``
    #     Returns:
    #         List[dict]: A list of annotation
    #     """
    #     with self.file_client.get_local_path(self.ann_file)
    #     as local_path: self.data_infos = self.load_annotations(local_path)

    #     return data_list
