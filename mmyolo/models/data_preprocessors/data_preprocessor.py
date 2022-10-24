# Copyright (c) OpenMMLab. All rights reserved.
import random
from typing import List, Tuple

import torch
import torch.nn.functional as F
from mmdet.models import BatchSyncRandomResize
from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmengine import MessageHub, is_list_of
from mmengine.dist import barrier, broadcast

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
        assert isinstance(data['data_samples'], torch.Tensor), \
            '"data_samples" should be a tensor, but got ' \
            f'{type(data["data_samples"])}. The possible reason for this ' \
            'is that you are not using it with ' \
            '"mmyolo.datasets.utils.yolov5_collate". Please refer to ' \
            '"configs/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py".'

        inputs = data['inputs'].to(self.device, non_blocking=True)

        if self._channel_conversion and inputs.shape[1] == 3:
            inputs = inputs[:, [2, 1, 0], ...]

        if self._enable_normalize:
            inputs = (inputs - self.mean) / self.std

        data_samples = data['data_samples'].to(self.device, non_blocking=True)

        if self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                inputs, data_samples = batch_aug(inputs, data_samples)

        img_metas = [{'batch_input_shape': inputs.shape[2:]}] * len(inputs)
        data_samples = {'bboxes_labels': data_samples, 'img_metas': img_metas}

        return {'inputs': inputs, 'data_samples': data_samples}


@MODELS.register_module()
class PPYOLOEDetDataPreprocessor(DetDataPreprocessor):

    def forward(self, data: dict, training: bool = False) -> dict:
        # 图片shape在这里之前就已经是统一的了
        if not training:
            return super().forward(data, training)

        # 这里图片尺度是不一样的

        data = self.cast_data(data)
        _batch_inputs = data['inputs']
        # 进行图像预处理
        # Process data with `pseudo_collate`.
        if is_list_of(_batch_inputs, torch.Tensor):
            batch_inputs = []
            for _batch_input in _batch_inputs:
                # channel transform
                if self._channel_conversion:
                    _batch_input = _batch_input[[2, 1, 0], ...]
                # Convert to float after channel conversion to ensure
                # efficiency
                _batch_input = _batch_input.float()
                # Normalization.
                if self._enable_normalize:
                    if self.mean.shape[0] == 3:
                        assert _batch_input.dim(
                        ) == 3 and _batch_input.shape[0] == 3, (
                            'If the mean has 3 values, the input tensor '
                            'should in shape of (3, H, W), but got the tensor '
                            f'with shape {_batch_input.shape}')
                    _batch_input = (_batch_input - self.mean) / self.std
                batch_inputs.append(_batch_input)
        elif isinstance(_batch_inputs, torch.Tensor):
            raise NotImplementedError
        else:
            raise NotImplementedError
        data['inputs'] = batch_inputs
        data.setdefault('data_samples', None)

        inputs, data_samples = data['inputs'], data['data_samples']

        if self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                inputs, data_samples = batch_aug(inputs, data_samples)

        img_metas = [{'batch_input_shape': inputs.shape[2:]}] * len(inputs)
        data_samples = {'bboxes_labels': data_samples, 'img_metas': img_metas}

        return {'inputs': inputs, 'data_samples': data_samples}


@MODELS.register_module()
class PPYOLOEBatchSyncRandomResize(BatchSyncRandomResize):
    """
    TODO: doc
    """

    def __init__(self,
                 random_size_range: Tuple[int, int],
                 interval: int = 0,
                 size_divisor: int = 32,
                 interp_list: List[str] = [
                     'nearest', 'linear', 'area', 'bicubic'
                 ],
                 keep_ratio: bool = False) -> None:
        super().__init__(random_size_range, interval, size_divisor)
        self.interp_list = interp_list
        self.keep_ratio = keep_ratio

    def forward(self, inputs, data_samples):
        # TODO: 区分tensor和list情况
        assert isinstance(inputs, list)
        for i in range(len(inputs)):
            _batch_input = inputs[i]
            data_sample = data_samples[i]
            # 确认下这里那个是wh
            h, w = _batch_input.shape[-2:]
            if self._input_size is None:
                self._input_size = (h, w)
            scale_y = self._input_size[0] / h
            scale_x = self._input_size[1] / w
            if scale_x != 1 or scale_y != 1:
                inputs = F.interpolate(
                    inputs,
                    size=self._input_size,
                    mode='bilinear',
                    align_corners=False)
                img_shape = (int(data_sample.img_shape[0] * scale_y),
                             int(data_sample.img_shape[1] * scale_x))
                pad_shape = (int(data_sample.pad_shape[0] * scale_y),
                             int(data_sample.pad_shape[1] * scale_x))
                data_sample.set_metainfo({
                    'img_shape': img_shape,
                    'pad_shape': pad_shape,
                    'batch_input_shape': self._input_size
                })
                data_sample.gt_instances.bboxes[
                    ...,
                    0::2] = data_sample.gt_instances.bboxes[...,
                                                            0::2] * scale_x
                data_sample.gt_instances.bboxes[
                    ...,
                    1::2] = data_sample.gt_instances.bboxes[...,
                                                            1::2] * scale_y
                if 'ignored_instances' in data_sample:
                    data_sample.ignored_instances.bboxes[
                        ..., 0::2] = data_sample.ignored_instances.bboxes[
                            ..., 0::2] * scale_x
                    data_sample.ignored_instances.bboxes[
                        ..., 1::2] = data_sample.ignored_instances.bboxes[
                            ..., 1::2] * scale_y

        message_hub = MessageHub.get_current_instance()
        if (message_hub.get_info('iter') + 1) % self._interval == 0:
            self._input_size = self._get_random_size(device=inputs.device)
        return inputs, data_samples

    def _get_random_size(self, device: torch.device) -> Tuple[int, int]:
        """Randomly generate a shape in ``_random_size_range`` and broadcast to
        all ranks."""
        tensor = torch.LongTensor(2).to(device)
        if self.rank == 0:
            size = random.randint(*self._random_size_range)
            size = self._size_divisor * size
            tensor[0] = size
            tensor[1] = size
        barrier()
        broadcast(tensor, 0)
        input_size = (tensor[0].item(), tensor[1].item())
        return input_size
