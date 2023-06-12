# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from mmdet.visualization import DetLocalVisualizer
from mmengine.structures import InstanceData

from mmyolo.registry import VISUALIZERS


@VISUALIZERS.register_module()
class FaceVisualizer(DetLocalVisualizer):

    def __init__(self,
                 name: str = 'visualizer',
                 image: Optional[np.ndarray] = None,
                 vis_backends: Optional[Dict] = None,
                 save_dir: Optional[str] = None,
                 bbox_color: Optional[Union[str, Tuple[int]]] = None,
                 text_color: Optional[Union[str,
                                            Tuple[int]]] = (200, 200, 200),
                 mask_color: Optional[Union[str, Tuple[int]]] = None,
                 keypoint_color: Optional[Union[str,
                                                Tuple[int]]] = ('blue',
                                                                'green', 'red',
                                                                'cyan',
                                                                'yellow'),
                 line_width: Union[int, float] = 3,
                 alpha: float = 0.8) -> None:
        super().__init__(name, image, vis_backends, save_dir, bbox_color,
                         text_color, mask_color, line_width, alpha)
        self.keypoint_color = keypoint_color

    def _draw_instances(self, image: np.ndarray, instances: List[InstanceData],
                        classes: Optional[List[str]],
                        palette: Optional[List[tuple]]) -> np.ndarray:
        super()._draw_instances(image, instances, classes, palette)
        if 'keypoints' in instances:
            keypoints = instances.keypoints
            for i in range(5):
                self.draw_points(
                    positions=keypoints[:, i * 2:(i + 1) * 2],
                    colors=self.keypoint_color[i],
                    sizes=5)
        return self.get_image()
