# Copyright (c) OpenMMLab. All rights reserved.
import random
from typing import List, Tuple, Union

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
    """# 这边还要完善处理机制类型，各种异常情况的判断 有些参数用不到的要在init做判断."""

    def forward(self, data: dict, training: bool = False) -> dict:
        # 图片shape在这里之前就已经是统一的了
        if not training:
            return super().forward(data, training)

        # 这里图片尺度是不一样的
        data = self.cast_data(data)
        inputs, data_samples = data['inputs'], data['data_samples']
        # 进行图像预处理
        # Process data with `pseudo_collate`.
        if is_list_of(inputs, torch.Tensor):
            batch_inputs = []
            for _batch_input, data_sample in zip(inputs, data_samples):
                # channel transform
                if self._channel_conversion:
                    _batch_input = _batch_input[[2, 1, 0], ...]
                # Convert to float after channel conversion to ensure
                # efficiency
                _batch_input = _batch_input.float()

                batch_inputs.append(_batch_input)
        elif isinstance(inputs, torch.Tensor):
            raise NotImplementedError
        else:
            raise NotImplementedError

        if self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                inputs, data_samples = batch_aug(batch_inputs, data_samples)

        # 这里是先resize，在normalize
        if self._enable_normalize:
            inputs = (inputs - self.mean) / self.std

        img_metas = [{'batch_input_shape': inputs.shape[2:]}] * len(inputs)
        data_samples = {'bboxes_labels': data_samples, 'img_metas': img_metas}

        return {'inputs': inputs, 'data_samples': data_samples}


@MODELS.register_module()
class PPYOLOEBatchSyncRandomResize(BatchSyncRandomResize):
    """# TODO: doc."""

    def __init__(self,
                 random_size_range: Tuple[int, int],
                 interval: int = 1,
                 size_divisor: int = 32,
                 random_interp=True,
                 interp_mode: Union[List[str], str] = [
                     'nearest', 'bilinear', 'bicubic', 'area'
                 ],
                 keep_ratio: bool = False) -> None:
        super().__init__(random_size_range, interval, size_divisor)
        self.random_interp = random_interp
        self.keep_ratio = keep_ratio

        if self.random_interp:
            assert isinstance(interp_mode, list) and len(interp_mode) > 1
            self.interp_mode_list = interp_mode
            self.interp_mode = None
        elif isinstance(interp_mode, str):
            self.interp_mode_list = None
            self.interp_mode = interp_mode
        else:
            # TODO:
            raise RuntimeError('xxx')

    def forward(self, inputs, data_samples):
        assert isinstance(inputs, list)
        # TODO: random_interp为True的时候
        message_hub = MessageHub.get_current_instance()
        if (message_hub.get_info('iter') + 1) % self._interval == 0:
            self._input_size, interp_mode = self._get_random_size_and_interp(
                device=inputs[0].device)
            if self.random_interp:
                self.interp_mode = interp_mode

        # TODO: 区分tensor和list情况
        # TODO: keep ratio的情况
        if isinstance(inputs, list):
            outputs = []
            for i in range(len(inputs)):
                _batch_input = inputs[i]
                data_sample = data_samples[i]
                h, w = _batch_input.shape[-2:]
                scale_y = self._input_size[0] / h
                scale_x = self._input_size[1] / w
                if scale_x != 1. or scale_y != 1.:
                    if interp_mode in ('nearest', 'area'):
                        align_corners = None
                    else:
                        align_corners = False
                    _batch_input = F.interpolate(
                        _batch_input.unsqueeze(0),
                        size=self._input_size,
                        mode=self.interp_mode,
                        align_corners=align_corners)

                    # rescale bbox
                    data_sample[:, 2] *= scale_x
                    data_sample[:, 3] *= scale_y
                    data_sample[:, 4] *= scale_x
                    data_sample[:, 5] *= scale_y
                else:
                    _batch_input = _batch_input.unsqueeze(0)

                outputs.append(_batch_input)
            return torch.cat(outputs, dim=0), torch.cat(data_samples, dim=0)
        else:
            raise NotImplementedError

    def _get_random_size_and_interp(self,
                                    device: torch.device) -> Tuple[int, int]:

        tensor = torch.LongTensor(3).to(device)
        if self.rank == 0:
            size = random.randint(*self._random_size_range)
            size = (self._size_divisor * size, self._size_divisor * size)
            tensor[0] = size[0]
            tensor[1] = size[1]

            if self.random_interp:
                interp_ind = random.randint(0, len(self.interp_mode_list) - 1)
                tensor[2] = interp_ind
        barrier()
        broadcast(tensor, 0)
        input_size = (tensor[0].item(), tensor[1].item())
        if self.random_interp:
            interp_mode = self.interp_mode_list[tensor[2].item()]
        else:
            interp_mode = None
        return input_size, interp_mode
