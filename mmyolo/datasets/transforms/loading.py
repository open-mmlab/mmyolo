from typing import Optional, Tuple, Union, Dict

import numpy as np
from plyfile import PlyData

from mmyolo.registry import TRANSFORMS
from mmcv.transforms import BaseTransform
from mmcv.transforms import LoadAnnotations as MMCV_LoadAnnotations

@TRANSFORMS.register_module()
class Load6DAnnotations(MMCV_LoadAnnotations):
    def __init__(self,
                 with_label: bool = True,
                 with_2d_bbox: bool = True,
                 with_corners: bool = True,
                 with_center: bool = True,
                 with_translation: bool = True,
                 with_rotation: bool = True,
                 file_client_args: dict = dict(backend='disk')
                 ) -> None:
        super(Load6DAnnotations, self).__init__(
            with_bbox=with_2d_bbox,
            with_label=with_label,
            file_client_args=file_client_args
        )
        self.with_corners = with_corners
        self.with_center = with_center
        self.with_translation = with_translation
        self.with_rotation = with_rotation

    
    def _load_rotation(self, results:dict) -> None:
        gt_rotations = []
        for instance in results.get('instances', []):
            gt_rotations.append(instance['rotation'])
        results['gt_rotations'] = np.array(gt_rotations, dtype=np.float32).reshape(-1, 10)

    def _load_translation(self, results:dict) -> None:
        gt_translations = []
        for instance in results.get('instances', []):
            gt_translations.append(instance['translation'])
        results['gt_translations'] = np.array(gt_translations, dtype=np.float32).reshape(-1,3)

    def _load_center(self, results:dict) -> None:
        gt_center = []
        for instance in results['instances']:
            gt_center.append(instance['center'])
        results['gt_center'] = np.array(gt_center, dtype=np.float32).reshape(-1,2)

    def _load_cornors(self, results: dict) -> None:
        gt_cornors = []
        for instance in results['instances']:
            gt_cornors.append(instance['corners'])
        results['gt_corners'] = np.array(gt_cornors, dtype=np.int64).reshape(1,-1)

    def transform(self, results: dict) -> dict:
        if self.with_label:
            self._load_labels(results)
        if self.with_bbox:
            self._load_bboxes(results)
        if self.with_corners:
            self._load_cornors(results)
        if self.with_center:
            self._load_center(results)
        if self.with_translation:
            self._load_translation(results)
        if self.with_rotation:
            self._load_rotation(results)
        
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_corners={self.with_corners}, '
        repr_str += f'with_center={self.with_center}, '
        repr_str += f'with_translation={self.with_translation}'
        repr_str += f'file_client_args={self.file_client})'
        return repr_str
