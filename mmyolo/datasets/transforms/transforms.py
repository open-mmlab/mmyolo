# Copyright (c) OpenMMLab. All rights reserved.
import math
from numbers import Number
from typing import Sequence, Tuple, Union

import cv2
import mmcv
import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness
from mmdet.datasets.transforms import LoadAnnotations as MMDET_LoadAnnotations
from mmdet.datasets.transforms import RandomCrop
from mmdet.datasets.transforms import Resize as MMDET_Resize
from mmdet.structures.bbox import (HorizontalBoxes, autocast_box_type,
                                   get_box_type)
from numpy import random

from mmyolo.registry import TRANSFORMS


@TRANSFORMS.register_module()
class YOLOv5KeepRatioResize(MMDET_Resize):
    """Resize images & bbox(if existed).

    This transform resizes the input image according to ``scale``.
    Bboxes (if existed) are then resized with the same scale factor.

    Required Keys:

    - img (np.uint8)
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)

    Modified Keys:

    - img (np.uint8)
    - img_shape (tuple)
    - gt_bboxes (optional)
    - scale (float)

    Added Keys:

    - scale_factor (np.float32)

    Args:
        scale (Union[int, Tuple[int, int]]): Images scales for resizing.
    """

    def __init__(self,
                 scale: Union[int, Tuple[int, int]],
                 keep_ratio: bool = True,
                 **kwargs):
        assert keep_ratio is True
        super().__init__(scale=scale, keep_ratio=True, **kwargs)

    @staticmethod
    def _get_rescale_ratio(old_size: Tuple[int, int],
                           scale: Union[float, Tuple[int]]) -> float:
        """Calculate the ratio for rescaling.

        Args:
            old_size (tuple[int]): The old size (w, h) of image.
            scale (float | tuple[int]): The scaling factor or maximum size.
                If it is a float number, then the image will be rescaled by
                this factor, else if it is a tuple of 2 integers, then
                the image will be rescaled as large as possible within
                the scale.

        Returns:
            float: The resize ratio.
        """
        w, h = old_size
        if isinstance(scale, (float, int)):
            if scale <= 0:
                raise ValueError(f'Invalid scale {scale}, must be positive.')
            scale_factor = scale
        elif isinstance(scale, tuple):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            scale_factor = min(max_long_edge / max(h, w),
                               max_short_edge / min(h, w))
        else:
            raise TypeError('Scale must be a number or tuple of int, '
                            f'but got {type(scale)}')

        return scale_factor

    def _resize_img(self, results: dict):
        """Resize images with ``results['scale']``."""
        assert self.keep_ratio is True

        if results.get('img', None) is not None:
            image = results['img']
            original_h, original_w = image.shape[:2]
            ratio = self._get_rescale_ratio((original_h, original_w),
                                            self.scale)

            if ratio != 1:
                # resize image according to the ratio
                image = mmcv.imrescale(
                    img=image,
                    scale=ratio,
                    interpolation='area' if ratio < 1 else 'bilinear',
                    backend=self.backend)

            resized_h, resized_w = image.shape[:2]
            scale_ratio = resized_h / original_h

            scale_factor = np.array([scale_ratio, scale_ratio],
                                    dtype=np.float32)

            results['img'] = image
            results['img_shape'] = image.shape[:2]
            results['scale_factor'] = scale_factor


@TRANSFORMS.register_module()
class LetterResize(MMDET_Resize):
    """Resize and pad image while meeting stride-multiple constraints.

    Required Keys:

    - img (np.uint8)
    - batch_shape (np.int64) (optional)

    Modified Keys:

    - img (np.uint8)
    - img_shape (tuple)
    - gt_bboxes (optional)

    Added Keys:
    - pad_param (np.float32)

    Args:
        scale (Union[int, Tuple[int, int]]): Images scales for resizing.
        pad_val (dict): Padding value. Defaults to dict(img=0, seg=255).
        use_mini_pad (bool): Whether using minimum rectangle padding.
            Defaults to True
        stretch_only (bool): Whether stretch to the specified size directly.
            Defaults to False
        allow_scale_up (bool): Allow scale up when ratio > 1. Defaults to True
    """

    def __init__(self,
                 scale: Union[int, Tuple[int, int]],
                 pad_val: dict = dict(img=0, mask=0, seg=255),
                 use_mini_pad: bool = False,
                 stretch_only: bool = False,
                 allow_scale_up: bool = True,
                 **kwargs):
        super().__init__(scale=scale, keep_ratio=True, **kwargs)

        self.pad_val = pad_val
        if isinstance(pad_val, (int, float)):
            pad_val = dict(img=pad_val, seg=255)
        assert isinstance(
            pad_val, dict), f'pad_val must be dict, but got {type(pad_val)}'

        self.use_mini_pad = use_mini_pad
        self.stretch_only = stretch_only
        self.allow_scale_up = allow_scale_up

    def _resize_img(self, results: dict):
        """Resize images with ``results['scale']``."""
        image = results.get('img', None)
        if image is None:
            return

        # Use batch_shape if a batch_shape policy is configured
        if 'batch_shape' in results:
            scale = tuple(results['batch_shape'])
        else:
            scale = self.scale

        image_shape = image.shape[:2]  # height, width

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
            # compare with no resize and padding size
            image = mmcv.imresize(
                image, (no_pad_shape[1], no_pad_shape[0]),
                interpolation=self.interpolation,
                backend=self.backend)

        scale_factor = np.array([ratio[0], ratio[1]], dtype=np.float32)

        if 'scale_factor' in results:
            results['scale_factor'] = results['scale_factor'] * scale_factor
        else:
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

            image = mmcv.impad(
                img=image,
                padding=(padding_list[2], padding_list[0], padding_list[3],
                         padding_list[1]),
                pad_val=pad_val,
                padding_mode='constant')

        results['img'] = image
        results['img_shape'] = image.shape
        results['pad_param'] = np.array(padding_list, dtype=np.float32)

    def _resize_masks(self, results: dict):
        """Resize masks with ``results['scale']``"""
        if results.get('gt_masks', None) is None:
            return

        # resize the gt_masks
        gt_mask_height = results['gt_masks'].height * \
            results['scale_factor'][0]
        gt_mask_width = results['gt_masks'].width * \
            results['scale_factor'][1]
        gt_masks = results['gt_masks'].resize(
            (int(round(gt_mask_height)), int(round(gt_mask_width))))

        # padding the gt_masks
        if len(gt_masks) == 0:
            padded_masks = np.empty((0, *results['img_shape'][:2]),
                                    dtype=np.uint8)
        else:
            # TODO: The function is incorrect. Because the mask may not
            #  be able to pad.
            padded_masks = np.stack([
                mmcv.impad(
                    mask,
                    padding=(int(results['pad_param'][2]),
                             int(results['pad_param'][0]),
                             int(results['pad_param'][3]),
                             int(results['pad_param'][1])),
                    pad_val=self.pad_val.get('masks', 0)) for mask in gt_masks
            ])
        results['gt_masks'] = type(results['gt_masks'])(
            padded_masks, *results['img_shape'][:2])

    def _resize_bboxes(self, results: dict):
        """Resize bounding boxes with ``results['scale_factor']``."""
        if results.get('gt_bboxes', None) is None:
            return
        results['gt_bboxes'].rescale_(results['scale_factor'])

        if len(results['pad_param']) != 4:
            return
        results['gt_bboxes'].translate_(
            (results['pad_param'][2], results['pad_param'][1]))

        if self.clip_object_border:
            results['gt_bboxes'].clip_(results['img_shape'])


# TODO: Check if it can be merged with mmdet.YOLOXHSVRandomAug
@TRANSFORMS.register_module()
class YOLOv5HSVRandomAug(BaseTransform):
    """Apply HSV augmentation to image sequentially.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        hue_delta ([int, float]): delta of hue. Defaults to 0.015.
        saturation_delta ([int, float]): delta of saturation. Defaults to 0.7.
        value_delta ([int, float]): delta of value. Defaults to 0.4.
    """

    def __init__(self,
                 hue_delta: Union[int, float] = 0.015,
                 saturation_delta: Union[int, float] = 0.7,
                 value_delta: Union[int, float] = 0.4):
        self.hue_delta = hue_delta
        self.saturation_delta = saturation_delta
        self.value_delta = value_delta

    def transform(self, results: dict) -> dict:
        """The HSV augmentation transform function.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        hsv_gains = \
            random.uniform(-1, 1, 3) * \
            [self.hue_delta, self.saturation_delta, self.value_delta] + 1
        hue, sat, val = cv2.split(
            cv2.cvtColor(results['img'], cv2.COLOR_BGR2HSV))

        table_list = np.arange(0, 256, dtype=hsv_gains.dtype)
        lut_hue = ((table_list * hsv_gains[0]) % 180).astype(np.uint8)
        lut_sat = np.clip(table_list * hsv_gains[1], 0, 255).astype(np.uint8)
        lut_val = np.clip(table_list * hsv_gains[2], 0, 255).astype(np.uint8)

        im_hsv = cv2.merge(
            (cv2.LUT(hue, lut_hue), cv2.LUT(sat,
                                            lut_sat), cv2.LUT(val, lut_val)))
        results['img'] = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
        return results


# TODO: can be accelerated
@TRANSFORMS.register_module()
class LoadAnnotations(MMDET_LoadAnnotations):
    """Because the yolo series does not need to consider ignore bboxes for the
    time being, in order to speed up the pipeline, it can be excluded in
    advance."""

    def _load_bboxes(self, results: dict):
        """Private function to load bounding box annotations.

        Note: BBoxes with ignore_flag of 1 is not considered.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """
        gt_bboxes = []
        gt_ignore_flags = []
        for instance in results.get('instances', []):
            if instance['ignore_flag'] == 0:
                gt_bboxes.append(instance['bbox'])
                gt_ignore_flags.append(instance['ignore_flag'])
        results['gt_ignore_flags'] = np.array(gt_ignore_flags, dtype=bool)

        if self.box_type is None:
            results['gt_bboxes'] = np.array(
                gt_bboxes, dtype=np.float32).reshape((-1, 4))
        else:
            _, box_type_cls = get_box_type(self.box_type)
            results['gt_bboxes'] = box_type_cls(gt_bboxes, dtype=torch.float32)

    def _load_labels(self, results: dict):
        """Private function to load label annotations.

        Note: BBoxes with ignore_flag of 1 is not considered.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded label annotations.
        """
        gt_bboxes_labels = []
        for instance in results.get('instances', []):
            if instance['ignore_flag'] == 0:
                gt_bboxes_labels.append(instance['bbox_label'])
        results['gt_bboxes_labels'] = np.array(
            gt_bboxes_labels, dtype=np.int64)


@TRANSFORMS.register_module()
class YOLOv5RandomAffine(BaseTransform):
    """Random affine transform data augmentation in YOLOv5. It is different
    from the implementation in YOLOX.

    This operation randomly generates affine transform matrix which including
    rotation, translation, shear and scaling transforms.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (np.bool) (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)

    Args:
        max_rotate_degree (float): Maximum degrees of rotation transform.
            Defaults to 10.
        max_translate_ratio (float): Maximum ratio of translation.
            Defaults to 0.1.
        scaling_ratio_range (tuple[float]): Min and max ratio of
            scaling transform. Defaults to (0.5, 1.5).
        max_shear_degree (float): Maximum degrees of shear
            transform. Defaults to 2.
        border (tuple[int]): Distance from height and width sides of input
            image to adjust output shape. Only used in mosaic dataset.
            Defaults to (0, 0).
        border_val (tuple[int]): Border padding values of 3 channels.
            Defaults to (114, 114, 114).
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
    """

    def __init__(self,
                 max_rotate_degree: float = 10.0,
                 max_translate_ratio: float = 0.1,
                 scaling_ratio_range: Tuple[float, float] = (0.5, 1.5),
                 max_shear_degree: float = 2.0,
                 border: Tuple[int, int] = (0, 0),
                 border_val: Tuple[int, int, int] = (114, 114, 114),
                 bbox_clip_border: bool = True,
                 min_bbox_size: int = 2,
                 min_area_ratio: float = 0.1,
                 max_aspect_ratio: int = 20):
        assert 0 <= max_translate_ratio <= 1
        assert scaling_ratio_range[0] <= scaling_ratio_range[1]
        assert scaling_ratio_range[0] > 0
        self.max_rotate_degree = max_rotate_degree
        self.max_translate_ratio = max_translate_ratio
        self.scaling_ratio_range = scaling_ratio_range
        self.max_shear_degree = max_shear_degree
        self.border = border
        self.border_val = border_val
        self.bbox_clip_border = bbox_clip_border
        self.min_bbox_size = min_bbox_size
        self.min_bbox_size = min_bbox_size
        self.min_area_ratio = min_area_ratio
        self.max_aspect_ratio = max_aspect_ratio

    @cache_randomness
    def _get_random_homography_matrix(self, height: int,
                                      width: int) -> Tuple[np.ndarray, float]:
        """Get random homography matrix.

        Args:
            height (int): Image height.
            width (int): Image width.

        Returns:
            Tuple[np.ndarray, float]: The result of warp_matrix and
            scaling_ratio.
        """
        # Rotation
        rotation_degree = random.uniform(-self.max_rotate_degree,
                                         self.max_rotate_degree)
        rotation_matrix = self._get_rotation_matrix(rotation_degree)

        # Scaling
        scaling_ratio = random.uniform(self.scaling_ratio_range[0],
                                       self.scaling_ratio_range[1])
        scaling_matrix = self._get_scaling_matrix(scaling_ratio)

        # Shear
        x_degree = random.uniform(-self.max_shear_degree,
                                  self.max_shear_degree)
        y_degree = random.uniform(-self.max_shear_degree,
                                  self.max_shear_degree)
        shear_matrix = self._get_shear_matrix(x_degree, y_degree)

        # Translation
        trans_x = random.uniform(0.5 - self.max_translate_ratio,
                                 0.5 + self.max_translate_ratio) * width
        trans_y = random.uniform(0.5 - self.max_translate_ratio,
                                 0.5 + self.max_translate_ratio) * height
        translate_matrix = self._get_translation_matrix(trans_x, trans_y)
        warp_matrix = (
            translate_matrix @ shear_matrix @ rotation_matrix @ scaling_matrix)
        return warp_matrix, scaling_ratio

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """The YOLOv5 random affine transform function.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        img = results['img']
        height = img.shape[0] + self.border[0] * 2
        width = img.shape[1] + self.border[1] * 2

        # Note: Different from YOLOX
        center_matrix = np.eye(3, dtype=np.float32)
        center_matrix[0, 2] = -img.shape[1] / 2
        center_matrix[1, 2] = -img.shape[0] / 2

        warp_matrix, scaling_ratio = self._get_random_homography_matrix(
            height, width)
        warp_matrix = warp_matrix @ center_matrix

        img = cv2.warpPerspective(
            img,
            warp_matrix,
            dsize=(width, height),
            borderValue=self.border_val)
        results['img'] = img
        results['img_shape'] = img.shape

        bboxes = results['gt_bboxes']
        num_bboxes = len(bboxes)
        if num_bboxes:
            orig_bboxes = bboxes.clone()

            bboxes.project_(warp_matrix)
            if self.bbox_clip_border:
                bboxes.clip_([height, width])

            # filter bboxes
            orig_bboxes.rescale_([scaling_ratio, scaling_ratio])

            # Be careful: valid_index must convert to numpy,
            # otherwise it will raise out of bounds when len(valid_index)=1
            valid_index = self.filter_gt_bboxes(orig_bboxes, bboxes).numpy()
            results['gt_bboxes'] = bboxes[valid_index]
            results['gt_bboxes_labels'] = results['gt_bboxes_labels'][
                valid_index]
            results['gt_ignore_flags'] = results['gt_ignore_flags'][
                valid_index]

            if 'gt_masks' in results:
                raise NotImplementedError('RandomAffine only supports bbox.')
        return results

    def filter_gt_bboxes(self, origin_bboxes: HorizontalBoxes,
                         wrapped_bboxes: HorizontalBoxes) -> torch.Tensor:
        """Filter gt bboxes.

        Args:
            origin_bboxes (HorizontalBoxes): Origin bboxes.
            wrapped_bboxes (HorizontalBoxes): Wrapped bboxes

        Returns:
            dict: The result dict.
        """
        origin_w = origin_bboxes.widths
        origin_h = origin_bboxes.heights
        wrapped_w = wrapped_bboxes.widths
        wrapped_h = wrapped_bboxes.heights
        aspect_ratio = np.maximum(wrapped_w / (wrapped_h + 1e-16),
                                  wrapped_h / (wrapped_w + 1e-16))

        wh_valid_idx = (wrapped_w > self.min_bbox_size) & \
                       (wrapped_h > self.min_bbox_size)
        area_valid_idx = wrapped_w * wrapped_h / (origin_w * origin_h +
                                                  1e-16) > self.min_area_ratio
        aspect_ratio_valid_idx = aspect_ratio < self.max_aspect_ratio
        return wh_valid_idx & area_valid_idx & aspect_ratio_valid_idx

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(max_rotate_degree={self.max_rotate_degree}, '
        repr_str += f'max_translate_ratio={self.max_translate_ratio}, '
        repr_str += f'scaling_ratio_range={self.scaling_ratio_range}, '
        repr_str += f'max_shear_degree={self.max_shear_degree}, '
        repr_str += f'border={self.border}, '
        repr_str += f'border_val={self.border_val}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str

    @staticmethod
    def _get_rotation_matrix(rotate_degrees: float) -> np.ndarray:
        """Get rotation matrix.

        Args:
            rotate_degrees (float): Rotate degrees.

        Returns:
            np.ndarray: The rotation matrix.
        """
        radian = math.radians(rotate_degrees)
        rotation_matrix = np.array(
            [[np.cos(radian), -np.sin(radian), 0.],
             [np.sin(radian), np.cos(radian), 0.], [0., 0., 1.]],
            dtype=np.float32)
        return rotation_matrix

    @staticmethod
    def _get_scaling_matrix(scale_ratio: float) -> np.ndarray:
        """Get scaling matrix.

        Args:
            scale_ratio (float): Scale ratio.

        Returns:
            np.ndarray: The scaling matrix.
        """
        scaling_matrix = np.array(
            [[scale_ratio, 0., 0.], [0., scale_ratio, 0.], [0., 0., 1.]],
            dtype=np.float32)
        return scaling_matrix

    @staticmethod
    def _get_shear_matrix(x_shear_degrees: float,
                          y_shear_degrees: float) -> np.ndarray:
        """Get shear matrix.

        Args:
            x_shear_degrees (float): X shear degrees.
            y_shear_degrees (float): Y shear degrees.

        Returns:
            np.ndarray: The shear matrix.
        """
        x_radian = math.radians(x_shear_degrees)
        y_radian = math.radians(y_shear_degrees)
        shear_matrix = np.array([[1, np.tan(x_radian), 0.],
                                 [np.tan(y_radian), 1, 0.], [0., 0., 1.]],
                                dtype=np.float32)
        return shear_matrix

    @staticmethod
    def _get_translation_matrix(x: float, y: float) -> np.ndarray:
        """Get translation matrix.

        Args:
            x (float): X translation.
            y (float): Y translation.

        Returns:
            np.ndarray: The translation matrix.
        """
        translation_matrix = np.array([[1, 0., x], [0., 1, y], [0., 0., 1.]],
                                      dtype=np.float32)
        return translation_matrix


@TRANSFORMS.register_module()
class PPYOLOERandomDistort(BaseTransform):

    def __init__(self,
                 hue=[-18, 18, 0.5],
                 saturation=[0.5, 1.5, 0.5],
                 contrast=[0.5, 1.5, 0.5],
                 brightness=[0.5, 1.5, 0.5],
                 random_apply=True,
                 count=4,
                 random_channel=False):
        self.hue = hue
        self.saturation = saturation
        self.contrast = contrast
        self.brightness = brightness
        self.random_apply = random_apply
        self.count = count
        self.random_channel = random_channel

    def transform_hue(self, results):
        low, high, prob = self.hue
        if np.random.uniform(0., 1.) < prob:
            return results

        img = results['img']
        img = img.astype(np.float32)
        # it works, but result differ from HSV version
        delta = np.random.uniform(low, high)
        u = np.cos(delta * np.pi)
        w = np.sin(delta * np.pi)
        bt = np.array([[1.0, 0.0, 0.0], [0.0, u, -w], [0.0, w, u]])
        tyiq = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.321],
                         [0.211, -0.523, 0.311]])
        ityiq = np.array([[1.0, 0.956, 0.621], [1.0, -0.272, -0.647],
                          [1.0, -1.107, 1.705]])
        t = np.dot(np.dot(ityiq, bt), tyiq).T
        img = np.dot(img, t)
        results['img'] = img
        return results

    def transform_saturation(self, results):
        low, high, prob = self.saturation

        if np.random.uniform(0., 1.) < prob:
            return results
        img = results['img']
        delta = np.random.uniform(low, high)
        img = img.astype(np.float32)
        # it works, but result differ from HSV version
        gray = img * np.array([[[0.299, 0.587, 0.114]]], dtype=np.float32)
        gray = gray.sum(axis=2, keepdims=True)
        gray *= (1.0 - delta)
        img *= delta
        img += gray
        results['img'] = img
        return results

    def transform_contrast(self, results):
        low, high, prob = self.contrast

        if np.random.uniform(0., 1.) < prob:
            return results
        img = results['img']
        delta = np.random.uniform(low, high)
        img = img.astype(np.float32)
        img *= delta
        results['img'] = img
        return results

    def transform_brightness(self, results):
        low, high, prob = self.brightness
        if np.random.uniform(0., 1.) < prob:
            return results
        img = results['img']
        delta = np.random.uniform(low, high)
        img = img.astype(np.float32)
        img += delta
        results['img'] = img
        return results

    def transform(self, results: dict) -> dict:
        # img = results['img']
        if self.random_apply:
            functions = [
                self.transform_brightness, self.transform_contrast,
                self.transform_saturation, self.transform_hue
            ]
            distortions = np.random.permutation(functions)[:self.count]
            for func in distortions:
                results = func(results)
            return results
        raise NotImplementedError


@TRANSFORMS.register_module()
class PPYOLOERandomExpand(LetterResize):

    def __init__(self, ratio=4., prob=0.5, fill_value=(127.5, 127.5, 127.5)):
        assert ratio > 1.01, 'expand ratio must be larger than 1.01'
        self.ratio = ratio
        self.prob = prob
        assert isinstance(fill_value, (Number, Sequence)), \
            'fill value must be either float or sequence'
        if isinstance(fill_value, Number):
            fill_value = (fill_value, ) * 3
        if not isinstance(fill_value, tuple):
            fill_value = tuple(fill_value)
        self.fill_value = fill_value

    def _resize_img(self, results: dict):
        if np.random.uniform(0., 1.) < self.prob:
            return results

        img = results['img']
        height, width = img.shape[:2]
        ratio = np.random.uniform(1., self.ratio)
        h = int(height * ratio)
        w = int(width * ratio)
        if not h > height or not w > width:
            return results
        y = np.random.randint(0, h - height)
        x = np.random.randint(0, w - width)
        # offsets, size = [x, y], [h, w]
        left_padding = x
        top_padding = y
        right_padding = h - height - left_padding
        bottom_padding = w - width - top_padding

        img = mmcv.impad(
            img=img,
            padding=(left_padding, top_padding, right_padding, bottom_padding),
            pad_val=self.fill_value,
            padding_mode='constant')
        if 'scale_factor' not in results:
            results['scale_factor'] = np.array([1, 1], dtype=np.float32)
        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_param'] = np.array(
            [top_padding, bottom_padding, left_padding, right_padding],
            dtype=np.float32)


class PPYOLOERandomCrop(RandomCrop):
    """
    TODO: 没搞完
    """

    def __init__(self,
                 aspect_ratio=[.5, 2.],
                 thresholds=[.0, .1, .3, .5, .7, .9],
                 scaling=[.3, 1.],
                 num_attempts=50,
                 allow_no_crop=True,
                 cover_all_box=False,
                 is_mask_crop=False):
        self.aspect_ratio = aspect_ratio
        self.thresholds = thresholds
        self.scaling = scaling
        self.num_attempts = num_attempts
        self.allow_no_crop = allow_no_crop
        self.cover_all_box = cover_all_box
        self.is_mask_crop = is_mask_crop

    def transform(self, results: dict) -> Union[dict, None]:
        if 'gt_bboxes' in results and len(results['gt_bboxes']) == 0:
            return results

        h, w = results['img'].shape[:2]
        gt_bbox = results['gt_bboxes']

        # NOTE Original method attempts to generate one candidate for each
        # threshold then randomly sample one from the resulting list.
        # Here a short circuit approach is taken, i.e., randomly choose a
        # threshold and attempt to find a valid crop, and simply return the
        # first one found.
        # The probability is not exactly the same, kinda resembling the
        # "Monty Hall" problem. Actually carrying out the attempts will affect
        # observability (just like opening doors in the "Monty Hall" game).
        thresholds = list(self.thresholds)
        if self.allow_no_crop:
            thresholds.append('no_crop')
        np.random.shuffle(thresholds)

        for thresh in thresholds:
            if thresh == 'no_crop':
                return results

            found = False
            for i in range(self.num_attempts):
                scale = np.random.uniform(*self.scaling)
                if self.aspect_ratio is not None:
                    min_ar, max_ar = self.aspect_ratio
                    aspect_ratio = np.random.uniform(
                        max(min_ar, scale**2), min(max_ar, scale**-2))
                    h_scale = scale / np.sqrt(aspect_ratio)
                    w_scale = scale * np.sqrt(aspect_ratio)
                else:
                    h_scale = np.random.uniform(*self.scaling)
                    w_scale = np.random.uniform(*self.scaling)
                crop_h = h * h_scale
                crop_w = w * w_scale
                if self.aspect_ratio is None:
                    if crop_h / crop_w < 0.5 or crop_h / crop_w > 2.0:
                        continue

                crop_h = int(crop_h)
                crop_w = int(crop_w)
                crop_y = np.random.randint(0, h - crop_h)
                crop_x = np.random.randint(0, w - crop_w)
                crop_box = [crop_x, crop_y, crop_x + crop_w, crop_y + crop_h]
                # TODO: 这里能不能求？
                iou = self._iou_matrix(
                    gt_bbox, torch.Tensor([crop_box], dtype=np.float32))
                if iou.max() < thresh:
                    continue

                if self.cover_all_box and iou.min() < thresh:
                    continue

                cropped_box, valid_ids = self._crop_box_with_center_constraint(
                    gt_bbox, np.array(crop_box, dtype=np.float32))
                if valid_ids.size > 0:
                    found = True
                    break

            if found:
                if self.is_mask_crop and 'gt_masks' in results and len(
                        results['gt_masks']) > 0:
                    raise NotImplementedError
                    # crop_polys = self.crop_segms(
                    #     sample['gt_poly'],
                    #     valid_ids,
                    #     np.array(
                    #         crop_box, dtype=np.int64),
                    #     h,
                    #     w)
                    # if [] in crop_polys:
                    #     delete_id = list()
                    #     valid_polys = list()
                    #     for id, crop_poly in enumerate(crop_polys):
                    #         if crop_poly == []:
                    #             delete_id.append(id)
                    #         else:
                    #             valid_polys.append(crop_poly)
                    #     valid_ids = np.delete(valid_ids, delete_id)
                    #     if len(valid_polys) == 0:
                    #         return sample
                    #     sample['gt_poly'] = valid_polys
                    # else:
                    #     sample['gt_poly'] = crop_polys

                # if 'gt_segm' in sample:
                #     sample['gt_segm'] = self._crop_segm(sample['gt_segm'],
                #                                         crop_box)
                #     sample['gt_segm'] = np.take(
                #         sample['gt_segm'], valid_ids, axis=0)

        #         results['image'] = self._crop_image(results['img'], crop_box)
        #         results['gt_bbox'] = np.take(cropped_box, valid_ids, axis=0)
        #         results['gt_class'] = np.take(
        #             results['gt_class'], valid_ids, axis=0)
        #         if 'gt_score' in sample:
        #             sample['gt_score'] = np.take(
        #                 sample['gt_score'], valid_ids, axis=0)
        #
        #         if 'is_crowd' in sample:
        #             sample['is_crowd'] = np.take(
        #                 sample['is_crowd'], valid_ids, axis=0)
        #
        #         if 'difficult' in sample:
        #             sample['difficult'] = np.take(
        #                 sample['difficult'], valid_ids, axis=0)
        #
        #         return sample
        #
        # return sample

    def _iou_matrix(self, a, b):
        tl_i = torch.maximum(a[:, np.newaxis, :2], b[:, :2])
        br_i = torch.minimum(a[:, np.newaxis, 2:], b[:, 2:])

        area_i = torch.prod(br_i - tl_i, dim=2) * (tl_i < br_i).all(dim=2)
        area_a = torch.prod(a[:, 2:] - a[:, :2], dim=1)
        area_b = torch.prod(b[:, 2:] - b[:, :2], dim=1)
        area_o = (area_a[:, np.newaxis] + area_b - area_i)
        return area_i / (area_o + 1e-10)

    def _crop_box_with_center_constraint(self, box, crop):
        cropped_box = box.copy()

        cropped_box[:, :2] = torch.maximum(box[:, :2], crop[:2])
        cropped_box[:, 2:] = torch.minimum(box[:, 2:], crop[2:])
        cropped_box[:, :2] -= crop[:2]
        cropped_box[:, 2:] -= crop[:2]

        centers = (box[:, :2] + box[:, 2:]) / 2
        valid = np.logical_and(crop[:2] <= centers,
                               centers < crop[2:]).all(axis=1)
        valid = np.logical_and(
            valid, (cropped_box[:, :2] < cropped_box[:, 2:]).all(axis=1))

        return cropped_box, np.where(valid)[0]
