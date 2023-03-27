from torchvision.transforms import functional as F
from mmcv.transforms import to_tensor
from mmdet.structures import DetDataSample
from typing import Tuple, Union
import numpy as np
from mmyolo.registry import TRANSFORMS
from mmcv.transforms import BaseTransform


@TRANSFORMS.register_module()
class TVLetterResize(BaseTransform):
    def __init__(self,
                 scale: Union[int, Tuple[int, int]],
                 pad_val: dict = dict(img=0, mask=0, seg=255),
                 use_mini_pad: bool = False,
                 stretch_only: bool = False,
                 allow_scale_up: bool = True,
                 interpolation='bilinear'):
        if isinstance(scale, int):
            self.scale = (scale, scale)
        else:
            self.scale = scale
        self.pad_val = pad_val
        self.interpolation = interpolation
        if isinstance(pad_val, (int, float)):
            pad_val = dict(img=pad_val, seg=255)
        assert isinstance(
            pad_val, dict), f'pad_val must be dict, but got {type(pad_val)}'

        self.use_mini_pad = use_mini_pad
        self.stretch_only = stretch_only
        self.allow_scale_up = allow_scale_up

    def transform(self, results: dict) -> dict:
        image = results.get('img', None)
        if image is None:
            return results

        if not image.flags.c_contiguous:
            image = np.ascontiguousarray(image.transpose(2, 0, 1))
            image = to_tensor(image)
        else:
            image = to_tensor(image).permute(2, 0, 1).contiguous()

        image = image[None]  # batch

        scale = self.scale[::-1]  # wh -> hw

        image_shape = image.shape[2:]  # height, width

        # Scale ratio (new / old)
        ratio = min(scale[0] / image_shape[0], scale[1] / image_shape[1])

        # only scale down, do not scale up (for better test mAP)
        if not self.allow_scale_up:
            ratio = min(ratio, 1.0)

        ratio = [ratio, ratio]  # float -> (float, float) for (height, width)

        # compute the best size of the image
        no_pad_shape = (int(round(image_shape[0] * ratio[0])),
                        int(round(image_shape[1] * ratio[1])))

        # padding height & width
        padding_h, padding_w = [
            scale[0] - no_pad_shape[0], scale[1] - no_pad_shape[1]
        ]
        if self.use_mini_pad:
            # minimum rectangle padding
            padding_w, padding_h = np.mod(padding_w, 32), np.mod(padding_h, 32)

        elif self.stretch_only:
            # stretch to the specified size directly
            padding_h, padding_w = 0.0, 0.0
            no_pad_shape = (scale[0], scale[1])
            ratio = [scale[0] / image_shape[0],
                     scale[1] / image_shape[1]]  # height, width ratios

        if image_shape != no_pad_shape:
            image = F.resize(image, [no_pad_shape[1], no_pad_shape[0]])

        scale_factor = (ratio[1], ratio[0])  # mmcv scale factor is (w, h)

        if 'scale_factor' in results:
            results['scale_factor_origin'] = results['scale_factor']
        results['scale_factor'] = scale_factor

        # padding
        top_padding, left_padding = int(round(padding_h // 2 - 0.1)), int(
            round(padding_w // 2 - 0.1))
        bottom_padding = padding_h - top_padding
        right_padding = padding_w - left_padding

        padding_list = [
            top_padding, bottom_padding, left_padding, right_padding
        ]
        if top_padding != 0 or bottom_padding != 0 or \
                left_padding != 0 or right_padding != 0:

            pad_val = self.pad_val.get('img', 0)
            if isinstance(pad_val, int) and image.ndim == 3:
                pad_val = tuple(pad_val for _ in range(image.shape[2]))

            image = F.pad(image, [padding_list[2], padding_list[0], padding_list[3], padding_list[1]], pad_val,
                          'constant')

        results['img'] = image
        results['img_shape'] = image.shape
        results['pad_param'] = np.array(padding_list, dtype=np.float32)
        return results


@TRANSFORMS.register_module()
class InferencePackDetInputs(BaseTransform):
    def __init__(self,
                 meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                            'scale_factor', 'flip', 'flip_direction')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        packed_results = dict()
        packed_results['inputs'] = results['img']
        data_sample = DetDataSample()

        img_meta = {}
        for key in self.meta_keys:
            assert key in results, f'`{key}` is not found in `results`, ' \
                                   f'the valid keys are {list(results)}.'
            img_meta[key] = results[key]

        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results
