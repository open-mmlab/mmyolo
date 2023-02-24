# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

from mmyolo.models import YOLODetector
from mmyolo.registry import MODELS
from projects.assigner_visualization.dense_heads import (RTMHeadAssigner,
                                                         YOLOv7HeadAssigner,
                                                         YOLOv8HeadAssigner)


@MODELS.register_module()
class YOLODetectorAssigner(YOLODetector):

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
        available_assigners = (YOLOv7HeadAssigner, YOLOv8HeadAssigner,
                               RTMHeadAssigner)
        if isinstance(self.bbox_head, available_assigners):
            data['data_samples']['feats'] = self.extract_feat(data['inputs'])
        inputs_hw = data['inputs'].shape[-2:]
        assign_results = self.bbox_head.assign(data['data_samples'], inputs_hw)
        return assign_results
