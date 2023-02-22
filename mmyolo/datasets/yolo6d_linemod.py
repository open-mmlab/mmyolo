# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import numpy as np
from mmengine import load
from mmengine.dataset import BaseDataset
from plyfile import PlyData

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

    def __init__(
        self,
        with_ply: bool = True,
        with_ply_loss: bool = True,
        *args,
        **kwargs,
    ):
        self.with_ply = with_ply
        self.with_ply_loss = with_ply_loss
        super().__init__(*args, **kwargs)

    def load_data_list(self) -> List[dict]:

        annotations = load(self.ann_file)
        if not isinstance(annotations, dict):
            raise TypeError(f'The annotations loaded from annotation file '
                            f'should be a dict, but got {type(annotations)}!')
        if 'data_list' not in annotations or 'metainfo' not in annotations:
            raise ValueError('Annotation must have data_list and metainfo '
                             'keys')
        if 'model_list' not in annotations or 'metainfo' not in annotations:
            raise ValueError('Annotation must have data_list and metainfo '
                             'keys')

        model_list = annotations['model_list']
        self.model_list = self.load_model_list(model_list)

        metainfo = annotations['metainfo']
        raw_data_list = annotations['data_list']

        for k, v in metainfo.items():
            self._metainfo.setdefault(k, v)

        data_list = []
        for raw_data_info in raw_data_list:
            # parse raw data information to target format
            data_info = self.parse_data_info(raw_data_info)
            if isinstance(data_info, dict):
                # For image tasks, `data_info` should information if single
                # image, such as dict(img_path='xxx', width=360, ...)
                data_list.append(data_info)
            elif isinstance(data_info, list):
                # For video tasks, `data_info` could contain image
                # information of multiple frames, such as
                # [dict(video_path='xxx', timestamps=...),
                #  dict(video_path='xxx', timestamps=...)]
                for item in data_info:
                    if not isinstance(item, dict):
                        raise TypeError('data_info must be list of dict, but '
                                        f'got {type(item)}')
                data_list.extend(data_info)
            else:
                raise TypeError('data_info should be a dict or list of dict, '
                                f'but got {type(data_info)}')

        return data_list

    def _load_model_ply(self, name):
        ply_path = name
        model_data = PlyData.read(ply_path)
        vertex = model_data['vertex']
        ply = np.stack([vertex[:]['x'], vertex[:]['y'], vertex[:]['z']],
                       axis=-1)

        return ply

    def _load_model_point_for_loss(self, ply):
        """Loads 500 points from model for loss."""
        num_points = ply.shape[0]
        num_points_for_shape_match_loss = 500

        if num_points == num_points_for_shape_match_loss:
            return ply
        elif num_points < num_points_for_shape_match_loss:
            points = np.zeros((num_points_for_shape_match_loss, 3))
            points[:num_points, :] = ply
            return points
        else:
            step_size = (num_points // num_points_for_shape_match_loss) - 1
            if step_size < 1:
                step_size = 1
            points = ply[::step_size, :]
            return points[:num_points_for_shape_match_loss, :]

    def load_model_list(self, model_list):
        if self.with_ply:
            models_point = {}
            for key, value in model_list['model_path_list'].items():
                ply = self._load_model_ply(value)
                models_point[key] = ply
            model_list['models'] = models_point

        if self.with_ply_loss:
            models_point_loss = {}
            for key, value in model_list['models'].items():
                points_for_loss = self._load_model_point_for_loss(value)
                models_point_loss[key] = points_for_loss
            model_list['models_point_loss'] = models_point_loss

        model_list['model_path_list'] = None

        return model_list
