# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.models.data_preprocessors import DetDataPreprocessor

from mmyolo.registry import MODELS


@MODELS.register_module()
class YOLOv5DetDataPreprocessor(DetDataPreprocessor):
    """Rewrite collate_fn to get faster training speed.

    Note: It must be used together with `mmyolo.datasets.utils.yolov5_collate`
    """

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization, padding and bgr2rgb conversion based on
        ``DetDataPreprocessorr``.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        if not training:
            return super().forward(data, training)

        inputs = data['inputs'].to(self.device, non_blocking=True)

        if self._channel_conversion and inputs.shape[1] == 3:
            inputs = inputs[:, [2, 1, 0], ...]

        if self._enable_normalize:
            inputs = (inputs - self.mean) / self.std

        masks = None
        if isinstance(data['data_samples'], dict):
            data_samples = data['data_samples']
            bboxes_labels = data_samples['bboxes_labels'].to(
                self.device, non_blocking=True)
            masks = data_samples['masks'].to(self.device, non_blocking=True)
        else:
            assert isinstance(data['data_samples'], torch.Tensor)
            bboxes_labels = data['data_samples'].to(
                self.device, non_blocking=True)

        # TODO: Support batch aug

        img_metas = [{'batch_input_shape': inputs.shape[2:]}] * len(inputs)
        data_samples = {'bboxes_labels': bboxes_labels, 'img_metas': img_metas}
        if masks is not None:
            data_samples['masks'] = masks
        return {'inputs': inputs, 'data_samples': data_samples}
