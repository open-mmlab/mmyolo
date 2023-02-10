# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

from mmyolo.models import YOLODetector
from mmyolo.registry import MODELS


@MODELS.register_module()
class YOLOv7DetectorAssigner(YOLODetector):

    def assign(self, data: dict) -> Union[dict, list]:
        """Calculate assigning results from a batch of inputs and data
        samples.This function is provided to the `assigner_visualization.py`
        script.

        Args:
            data (dict or tuple or list): Data sampled from dataset.
        Returns:
            dict: A dictionary of assigning components.
        """
        assert isinstance(data, dict)
        assert len(data['inputs']) == 1, 'Only support batchsize == 1'
        data = self.data_preprocessor(data, True)
        x = self.extract_feat(data['inputs'])
        assign_results = self.bbox_head.assign(
            x, data['data_samples']['bboxes_labels'],
            data['data_samples']['img_metas'])
        return assign_results
