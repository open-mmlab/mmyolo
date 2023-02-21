# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dataset import BaseDataset

from mmyolo.registry import DATASETS


@DATASETS.register_module()
class YOLO6DDataset(BaseDataset):
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
    #     """Load annotations from an annotation file named as "self.annfile

    #     Returns:
    #         List[dict]: A list of annotation
    #     """

    #     # ids '01'
    #     # cat 'ape'
    #     self.cat2id = {'ape': '01',
    #                    'benchvise': '02',
    #                    'bowl': '03',
    #                    'cam': '04',
    #                    'can': '05',
    #                    'cat': '06',
    #                    'cup': '07',
    #                    'driller': '08',
    #                    'duck': '09',
    #                    'eggbox': '10',
    #                    'glue': '11',
    #                    'holepuncher': '12',
    #                    'iron': '13',
    #                    'lamp': '14',
    #                    'phone': '15',
    #                     }
    #     self.id2cat = {id:cat for cat, id in self.cat2id}
    #     self.id2model_point = {}

    #     annotations = load(self.ann_file)
    #     if not isinstance(annotations, dict):
    #         raise TypeError(f'The annotations loaded from annotation file '
    #                         f'should be a dict, but got {type(annotations)}!
    #                           ')
    #     if 'data_list' not in annotations or 'metainfo' not in annotations:
    #         raise ValueError('Annotation must have data_list and metainfo '
    #                          'keys')

    #     data_list = []
    #     return data_list
