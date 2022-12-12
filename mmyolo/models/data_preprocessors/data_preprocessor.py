# Copyright (c) OpenMMLab. All rights reserved.
import random
from typing import List, Tuple, Union

import cv2
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
    """Rewrite collate_fn to get faster training speed.

    Note: It must be used together with `mmyolo.datasets.utils.ppyoloe_collate`
    """

    def forward(self, data: dict, training: bool = False) -> dict:
        if not training:
            return super().forward(data, training)

        assert isinstance(data['inputs'], list) and is_list_of(
            data['inputs'], torch.Tensor), \
            '"inputs" should be a list of Tensor, but got ' \
            f'{type(data["inputs"])}. The possible reason for this ' \
            'is that you are not using it with ' \
            '"mmyolo.datasets.utils.ppyoloe_collate". Please refer to ' \
            '"cconfigs/ppyoloe/ppyoloe_plus_s_fast_8xb8-80e_coco.py".'

        data = self.cast_data(data)
        inputs, data_samples = data['inputs'], data['data_samples']

        # Process data.
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

        # Batch random resize image.
        if self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                inputs, data_samples = batch_aug(batch_inputs, data_samples)

        if self._enable_normalize:
            inputs = (inputs - self.mean) / self.std

        img_metas = [{'batch_input_shape': inputs.shape[2:]}] * len(inputs)
        data_samples = {'bboxes_labels': data_samples, 'img_metas': img_metas}

        return {'inputs': inputs, 'data_samples': data_samples}


@MODELS.register_module()
class PPYOLOEBatchSyncRandomResize(BatchSyncRandomResize):
    """PPYOLOE batch random resize which synchronizes the random size across
    ranks.

    Args:
        random_size_range (tuple): The multi-scale random range during
            multi-scale training.
        interval (int): The iter interval of change
            image size. Defaults to 10.
        size_divisor (int): Image size divisible factor.
            Defaults to 32.
        random_interp (bool): Whether to choose interp_mode randomly.
            If set to True, the type of `interp_mode` must be list.
            If set to False, the type of `interp_mode` must be str.
            Defaults to True.
        interp_mode (Union[List, str]): The modes available for resizing
            are ('nearest', 'bilinear', 'bicubic', 'area').
        keep_ratio (bool): Whether to keep the aspect ratio when resizing
            the image. Now we only support keep_ratio=False.
            Defaults to False.
        broadcast_flag (bool): Whether to use same image size between
            gpus while resize image.
            Defaults to True.
    """

    def __init__(self,
                 random_size_range: Tuple[int, int],
                 interval: int = 1,
                 size_divisor: int = 32,
                 random_interp=True,
                 interp_mode: Union[List[str], str] = [
                     'nearest', 'bilinear', 'bicubic', 'area', 'lanczos4'
                 ],
                 keep_ratio: bool = False,
                 broadcast_flag: bool = False) -> None:
        super().__init__(random_size_range, interval, size_divisor)
        self.random_interp = random_interp
        self.keep_ratio = keep_ratio
        assert not self.keep_ratio, 'We do not yet support keep_ratio=True'
        # TODO: 删除这个
        self.broadcast_flag = broadcast_flag

        if self.random_interp:
            assert isinstance(interp_mode, list) and len(interp_mode) > 1,\
                'While random_interp==True, the type of `interp_mode`' \
                ' must be list and len(interp_mode) must large than 1'
            self.interp_mode_list = interp_mode
            self.interp_mode = None
        else:
            assert isinstance(interp_mode, str),\
                'While random_interp==False, the type of ' \
                '`interp_mode` must be str'
            assert interp_mode in ['nearest', 'bilinear', 'bicubic', 'area']
            self.interp_mode_list = None
            self.interp_mode = interp_mode

        self.interp_dict = {
            'area': cv2.INTER_AREA,
            'lanczos4': cv2.INTER_LANCZOS4
        }

    def forward(self, inputs, data_samples):
        assert isinstance(inputs, list)
        message_hub = MessageHub.get_current_instance()
        if (message_hub.get_info('iter') + 1) % self._interval == 0:
            # get current input size
            self._input_size, interp_mode = self._get_random_size_and_interp(
                device=inputs[0].device)
            if self.random_interp:
                self.interp_mode = interp_mode

        # TODO: need to support type(inputs)==Tensor
        # TODO: need to support keep_ratio==True
        if isinstance(inputs, list):
            outputs = []
            for i in range(len(inputs)):
                _batch_input = inputs[i]
                data_sample = data_samples[i]
                h, w = _batch_input.shape[-2:]
                scale_y = self._input_size[0] / h
                scale_x = self._input_size[1] / w
                if scale_x != 1. or scale_y != 1.:
                    if self.interp_mode in ('area', 'lanczos4'):
                        # print('interp', self.interp_mode)
                        device = _batch_input.device
                        input_numpy = _batch_input.cpu().numpy().transpose(
                            (1, 2, 0))
                        input_numpy = cv2.resize(
                            input_numpy,
                            None,
                            None,
                            fx=scale_x,
                            fy=scale_y,
                            interpolation=self.interp_dict[self.interp_mode])
                        _batch_input = input_numpy.transpose((2, 0, 1))
                        _batch_input = torch.from_numpy(_batch_input).to(
                            device).unsqueeze(0)

                    else:
                        if self.interp_mode in ('nearest', 'area'):
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

            # convert to Tensor
            return torch.cat(outputs, dim=0), torch.cat(data_samples, dim=0)
        else:
            raise NotImplementedError('Not implemented yet!')

    def _get_random_size_and_interp(self,
                                    device: torch.device) -> Tuple[int, int]:
        """Randomly generate a shape in ``_random_size_range`` and a
        interp_mode in interp_mode_list, and broadcast to all ranks."""
        tensor = torch.LongTensor(3).to(device)
        if (self.broadcast_flag
                and self.rank == 0) or (not self.broadcast_flag):
            size = random.randint(*self._random_size_range)
            size = (self._size_divisor * size, self._size_divisor * size)
            tensor[0] = size[0]
            tensor[1] = size[1]

            if self.random_interp:
                interp_ind = random.randint(0, len(self.interp_mode_list) - 1)
                tensor[2] = interp_ind

        # broadcast to all gpu
        if self.broadcast_flag:
            barrier()
            broadcast(tensor, 0)
        input_size = (tensor[0].item(), tensor[1].item())
        if self.random_interp and self.broadcast_flag:
            interp_mode = self.interp_mode_list[tensor[2].item()]
        elif self.random_interp:
            interp_ind = random.randint(0, len(self.interp_mode_list) - 1)
            interp_mode = self.interp_mode_list[interp_ind]
        else:
            interp_mode = None
        return input_size, interp_mode
