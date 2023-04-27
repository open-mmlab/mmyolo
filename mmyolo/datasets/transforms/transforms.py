# Copyright (c) OpenMMLab. All rights reserved.
import math
from copy import deepcopy
from typing import List, Sequence, Tuple, Union

import cv2
import mmcv
import numpy as np
import torch
from mmcv.image.geometric import _scale_size
from mmcv.transforms import BaseTransform, Compose
from mmcv.transforms.utils import cache_randomness
from mmdet.datasets.transforms import FilterAnnotations as FilterDetAnnotations
from mmdet.datasets.transforms import LoadAnnotations as MMDET_LoadAnnotations
from mmdet.datasets.transforms import RandomAffine as MMDET_RandomAffine
from mmdet.datasets.transforms import RandomFlip as MMDET_RandomFlip
from mmdet.datasets.transforms import Resize as MMDET_Resize
from mmdet.structures.bbox import (HorizontalBoxes, autocast_box_type,
                                   get_box_type)
from mmdet.structures.mask import PolygonMasks, polygon_to_bitmap
from numpy import random

from mmyolo.registry import TRANSFORMS
from .keypoint_structure import Keypoints

# TODO: Waiting for MMCV support
TRANSFORMS.register_module(module=Compose, force=True)


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
                # resize image according to the shape
                # NOTE: We are currently testing on COCO that modifying
                # this code will not affect the results.
                # If you find that it has an effect on your results,
                # please feel free to contact us.
                image = mmcv.imresize(
                    img=image,
                    size=(int(original_w * ratio), int(original_h * ratio)),
                    interpolation='area' if ratio < 1 else 'bilinear',
                    backend=self.backend)

            resized_h, resized_w = image.shape[:2]
            scale_ratio_h = resized_h / original_h
            scale_ratio_w = resized_w / original_w
            scale_factor = (scale_ratio_w, scale_ratio_h)

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
        half_pad_param (bool): If set to True, left and right pad_param will
            be given by dividing padding_h by 2. If set to False, pad_param is
            in int format. We recommend setting this to False for object
            detection tasks, and True for instance segmentation tasks.
            Default to False.
    """

    def __init__(self,
                 scale: Union[int, Tuple[int, int]],
                 pad_val: dict = dict(img=0, mask=0, seg=255),
                 use_mini_pad: bool = False,
                 stretch_only: bool = False,
                 allow_scale_up: bool = True,
                 half_pad_param: bool = False,
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
        self.half_pad_param = half_pad_param

    def _resize_img(self, results: dict):
        """Resize images with ``results['scale']``."""
        image = results.get('img', None)
        if image is None:
            return

        # Use batch_shape if a batch_shape policy is configured
        if 'batch_shape' in results:
            scale = tuple(results['batch_shape'])  # hw
        else:
            scale = self.scale[::-1]  # wh -> hw

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

        scale_factor = (no_pad_shape[1] / image_shape[1],
                        no_pad_shape[0] / image_shape[0])

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

            image = mmcv.impad(
                img=image,
                padding=(padding_list[2], padding_list[0], padding_list[3],
                         padding_list[1]),
                pad_val=pad_val,
                padding_mode='constant')

        results['img'] = image
        results['img_shape'] = image.shape
        if 'pad_param' in results:
            results['pad_param_origin'] = results['pad_param'] * \
                                          np.repeat(ratio, 2)

        if self.half_pad_param:
            results['pad_param'] = np.array(
                [padding_h / 2, padding_h / 2, padding_w / 2, padding_w / 2],
                dtype=np.float32)
        else:
            # We found in object detection, using padding list with
            # int type can get higher mAP.
            results['pad_param'] = np.array(padding_list, dtype=np.float32)

    def _resize_masks(self, results: dict):
        """Resize masks with ``results['scale']``"""
        if results.get('gt_masks', None) is None:
            return

        gt_masks = results['gt_masks']
        assert isinstance(
            gt_masks, PolygonMasks
        ), f'Only supports PolygonMasks, but got {type(gt_masks)}'

        # resize the gt_masks
        gt_mask_h = results['gt_masks'].height * results['scale_factor'][1]
        gt_mask_w = results['gt_masks'].width * results['scale_factor'][0]
        gt_masks = results['gt_masks'].resize(
            (int(round(gt_mask_h)), int(round(gt_mask_w))))

        top_padding, _, left_padding, _ = results['pad_param']
        if int(left_padding) != 0:
            gt_masks = gt_masks.translate(
                out_shape=results['img_shape'][:2],
                offset=int(left_padding),
                direction='horizontal')
        if int(top_padding) != 0:
            gt_masks = gt_masks.translate(
                out_shape=results['img_shape'][:2],
                offset=int(top_padding),
                direction='vertical')
        results['gt_masks'] = gt_masks

    def _resize_bboxes(self, results: dict):
        """Resize bounding boxes with ``results['scale_factor']``."""
        if results.get('gt_bboxes', None) is None:
            return
        results['gt_bboxes'].rescale_(results['scale_factor'])

        if len(results['pad_param']) != 4:
            return
        results['gt_bboxes'].translate_(
            (results['pad_param'][2], results['pad_param'][0]))

        if self.clip_object_border:
            results['gt_bboxes'].clip_(results['img_shape'])

    def transform(self, results: dict) -> dict:
        results = super().transform(results)
        if 'scale_factor_origin' in results:
            scale_factor_origin = results.pop('scale_factor_origin')
            results['scale_factor'] = (results['scale_factor'][0] *
                                       scale_factor_origin[0],
                                       results['scale_factor'][1] *
                                       scale_factor_origin[1])
        if 'pad_param_origin' in results:
            pad_param_origin = results.pop('pad_param_origin')
            results['pad_param'] += pad_param_origin
        return results


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

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(hue_delta={self.hue_delta}, '
        repr_str += f'saturation_delta={self.saturation_delta}, '
        repr_str += f'value_delta={self.value_delta})'
        return repr_str


@TRANSFORMS.register_module()
class LoadAnnotations(MMDET_LoadAnnotations):
    """Because the yolo series does not need to consider ignore bboxes for the
    time being, in order to speed up the pipeline, it can be excluded in
    advance.

    Args:
        mask2bbox (bool): Whether to use mask annotation to get bbox.
            Defaults to False.
        poly2mask (bool): Whether to transform the polygons to bitmaps.
            Defaults to False.
        merge_polygons (bool): Whether to merge polygons into one polygon.
            If merged, the storage structure is simpler and training is more
            effcient, especially if the mask inside a bbox is divided into
            multiple polygons. Defaults to True.
    """

    def __init__(self,
                 mask2bbox: bool = False,
                 poly2mask: bool = False,
                 merge_polygons: bool = True,
                 **kwargs):
        self.mask2bbox = mask2bbox
        self.merge_polygons = merge_polygons
        assert not poly2mask, 'Does not support BitmapMasks considering ' \
                              'that bitmap consumes more memory.'
        super().__init__(poly2mask=poly2mask, **kwargs)
        if self.mask2bbox:
            assert self.with_mask, 'Using mask2bbox requires ' \
                                   'with_mask is True.'
        self._mask_ignore_flag = None

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded bounding box, label and
            semantic segmentation.
        """
        if self.mask2bbox:
            self._load_masks(results)
            if self.with_label:
                self._load_labels(results)
                self._update_mask_ignore_data(results)
            gt_bboxes = results['gt_masks'].get_bboxes(dst_type='hbox')
            results['gt_bboxes'] = gt_bboxes
        elif self.with_keypoints:
            self._load_kps(results)
            _, box_type_cls = get_box_type(self.box_type)
            results['gt_bboxes'] = box_type_cls(
                results.get('bbox', []), dtype=torch.float32)
        else:
            results = super().transform(results)
            self._update_mask_ignore_data(results)
        return results

    def _update_mask_ignore_data(self, results: dict) -> None:
        if 'gt_masks' not in results:
            return

        if 'gt_bboxes_labels' in results and len(
                results['gt_bboxes_labels']) != len(results['gt_masks']):
            assert len(results['gt_bboxes_labels']) == len(
                self._mask_ignore_flag)
            results['gt_bboxes_labels'] = results['gt_bboxes_labels'][
                self._mask_ignore_flag]

        if 'gt_bboxes' in results and len(results['gt_bboxes']) != len(
                results['gt_masks']):
            assert len(results['gt_bboxes']) == len(self._mask_ignore_flag)
            results['gt_bboxes'] = results['gt_bboxes'][self._mask_ignore_flag]

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

    def _load_masks(self, results: dict) -> None:
        """Private function to load mask annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.
        """
        gt_masks = []
        gt_ignore_flags = []
        self._mask_ignore_flag = []
        for instance in results.get('instances', []):
            if instance['ignore_flag'] == 0:
                if 'mask' in instance:
                    gt_mask = instance['mask']
                    if isinstance(gt_mask, list):
                        gt_mask = [
                            np.array(polygon) for polygon in gt_mask
                            if len(polygon) % 2 == 0 and len(polygon) >= 6
                        ]
                        if len(gt_mask) == 0:
                            # ignore
                            self._mask_ignore_flag.append(0)
                        else:
                            if len(gt_mask) > 1 and self.merge_polygons:
                                gt_mask = self.merge_multi_segment(gt_mask)
                            gt_masks.append(gt_mask)
                            gt_ignore_flags.append(instance['ignore_flag'])
                            self._mask_ignore_flag.append(1)
                    else:
                        raise NotImplementedError(
                            'Only supports mask annotations in polygon '
                            'format currently')
                else:
                    # TODO: Actually, gt with bbox and without mask needs
                    #  to be retained
                    self._mask_ignore_flag.append(0)
        self._mask_ignore_flag = np.array(self._mask_ignore_flag, dtype=bool)
        results['gt_ignore_flags'] = np.array(gt_ignore_flags, dtype=bool)

        h, w = results['ori_shape']
        gt_masks = PolygonMasks([mask for mask in gt_masks], h, w)
        results['gt_masks'] = gt_masks

    def merge_multi_segment(self,
                            gt_masks: List[np.ndarray]) -> List[np.ndarray]:
        """Merge multi segments to one list.

        Find the coordinates with min distance between each segment,
        then connect these coordinates with one thin line to merge all
        segments into one.
        Args:
            gt_masks(List(np.array)):
                original segmentations in coco's json file.
                like [segmentation1, segmentation2,...],
                each segmentation is a list of coordinates.
        Return:
            gt_masks(List(np.array)): merged gt_masks
        """
        s = []
        segments = [np.array(i).reshape(-1, 2) for i in gt_masks]
        idx_list = [[] for _ in range(len(gt_masks))]

        # record the indexes with min distance between each segment
        for i in range(1, len(segments)):
            idx1, idx2 = self.min_index(segments[i - 1], segments[i])
            idx_list[i - 1].append(idx1)
            idx_list[i].append(idx2)

        # use two round to connect all the segments
        # first round: first to end, i.e. A->B(partial)->C
        # second round: end to first, i.e. C->B(remaining)-A
        for k in range(2):
            # forward first round
            if k == 0:
                for i, idx in enumerate(idx_list):
                    # middle segments have two indexes
                    # reverse the index of middle segments
                    if len(idx) == 2 and idx[0] > idx[1]:
                        idx = idx[::-1]
                        segments[i] = segments[i][::-1, :]
                    # add the idx[0] point for connect next segment
                    segments[i] = np.roll(segments[i], -idx[0], axis=0)
                    segments[i] = np.concatenate(
                        [segments[i], segments[i][:1]])
                    # deal with the first segment and the last one
                    if i in [0, len(idx_list) - 1]:
                        s.append(segments[i])
                    # deal with the middle segment
                    # Note that in the first round, only partial segment
                    # are appended.
                    else:
                        idx = [0, idx[1] - idx[0]]
                        s.append(segments[i][idx[0]:idx[1] + 1])
            # forward second round
            else:
                for i in range(len(idx_list) - 1, -1, -1):
                    # deal with the middle segment
                    # append the remaining points
                    if i not in [0, len(idx_list) - 1]:
                        idx = idx_list[i]
                        nidx = abs(idx[1] - idx[0])
                        s.append(segments[i][nidx:])
        return [np.concatenate(s).reshape(-1, )]

    def min_index(self, arr1: np.ndarray, arr2: np.ndarray) -> Tuple[int, int]:
        """Find a pair of indexes with the shortest distance.

        Args:
            arr1: (N, 2).
            arr2: (M, 2).
        Return:
            tuple: a pair of indexes.
        """
        dis = ((arr1[:, None, :] - arr2[None, :, :])**2).sum(-1)
        return np.unravel_index(np.argmin(dis, axis=None), dis.shape)

    def _load_kps(self, results: dict) -> None:
        """Private function to load keypoints annotations.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded keypoints annotations.
        """
        results['height'] = results['img_shape'][0]
        results['width'] = results['img_shape'][1]
        num_instances = len(results.get('bbox', []))

        if num_instances == 0:
            results['keypoints'] = np.empty(
                (0, len(results['flip_indices']), 2), dtype=np.float32)
            results['keypoints_visible'] = np.empty(
                (0, len(results['flip_indices'])), dtype=np.int32)
            results['category_id'] = []

        results['gt_keypoints'] = Keypoints(
            keypoints=results['keypoints'],
            keypoints_visible=results['keypoints_visible'],
            flip_indices=results['flip_indices'],
        )

        results['gt_ignore_flags'] = np.array([False] * num_instances)
        results['gt_bboxes_labels'] = np.array(results['category_id']) - 1

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'mask2bbox={self.mask2bbox}, '
        repr_str += f'poly2mask={self.poly2mask}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str


@TRANSFORMS.register_module()
class YOLOv5RandomAffine(BaseTransform):
    """Random affine transform data augmentation in YOLOv5 and YOLOv8. It is
    different from the implementation in YOLOX.

    This operation randomly generates affine transform matrix which including
    rotation, translation, shear and scaling transforms.
    If you set use_mask_refine == True, the code will use the masks
    annotation to refine the bbox.
    Our implementation is slightly different from the official. In COCO
    dataset, a gt may have multiple mask tags.  The official YOLOv5
    annotation file already combines the masks that an object has,
    but our code takes into account the fact that an object has multiple masks.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)
    - gt_masks (PolygonMasks) (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)
    - gt_masks (PolygonMasks) (optional)

    Args:
        max_rotate_degree (float): Maximum degrees of rotation transform.
            Defaults to 10.
        max_translate_ratio (float): Maximum ratio of translation.
            Defaults to 0.1.
        scaling_ratio_range (tuple[float]): Min and max ratio of
            scaling transform. Defaults to (0.5, 1.5).
        max_shear_degree (float): Maximum degrees of shear
            transform. Defaults to 2.
        border (tuple[int]): Distance from width and height sides of input
            image to adjust output shape. Only used in mosaic dataset.
            Defaults to (0, 0).
        border_val (tuple[int]): Border padding values of 3 channels.
            Defaults to (114, 114, 114).
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        min_bbox_size (float): Width and height threshold to filter bboxes.
            If the height or width of a box is smaller than this value, it
            will be removed. Defaults to 2.
        min_area_ratio (float): Threshold of area ratio between
            original bboxes and wrapped bboxes. If smaller than this value,
            the box will be removed. Defaults to 0.1.
        use_mask_refine (bool): Whether to refine bbox by mask. Deprecated.
        max_aspect_ratio (float): Aspect ratio of width and height
            threshold to filter bboxes. If max(h/w, w/h) larger than this
            value, the box will be removed. Defaults to 20.
        resample_num (int): Number of poly to resample to.
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
                 use_mask_refine: bool = False,
                 max_aspect_ratio: float = 20.,
                 resample_num: int = 1000):
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
        self.min_area_ratio = min_area_ratio
        # The use_mask_refine parameter has been deprecated.
        self.use_mask_refine = use_mask_refine
        self.max_aspect_ratio = max_aspect_ratio
        self.resample_num = resample_num

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """The YOLOv5 random affine transform function.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        img = results['img']
        # self.border is wh format
        height = img.shape[0] + self.border[1] * 2
        width = img.shape[1] + self.border[0] * 2

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
        img_h, img_w = img.shape[:2]

        bboxes = results['gt_bboxes']
        num_bboxes = len(bboxes)
        if num_bboxes:
            orig_bboxes = bboxes.clone()
            if 'gt_masks' in results:
                # If the dataset has annotations of mask,
                # the mask will be used to refine bbox.
                gt_masks = results['gt_masks']

                gt_masks_resample = self.resample_masks(gt_masks)
                gt_masks = self.warp_mask(gt_masks_resample, warp_matrix,
                                          img_h, img_w)

                # refine bboxes by masks
                bboxes = self.segment2box(gt_masks, height, width)
                # filter bboxes outside image
                valid_index = self.filter_gt_bboxes(orig_bboxes,
                                                    bboxes).numpy()
                if self.bbox_clip_border:
                    bboxes.clip_([height - 1e-3, width - 1e-3])
                    gt_masks = self.clip_polygons(gt_masks, height, width)
                results['gt_masks'] = gt_masks[valid_index]
            else:
                bboxes.project_(warp_matrix)
                if self.bbox_clip_border:
                    bboxes.clip_([height, width])

                # filter bboxes
                orig_bboxes.rescale_([scaling_ratio, scaling_ratio])

                # Be careful: valid_index must convert to numpy,
                # otherwise it will raise out of bounds when len(valid_index)=1
                valid_index = self.filter_gt_bboxes(orig_bboxes,
                                                    bboxes).numpy()

            results['gt_bboxes'] = bboxes[valid_index]
            results['gt_bboxes_labels'] = results['gt_bboxes_labels'][
                valid_index]
            results['gt_ignore_flags'] = results['gt_ignore_flags'][
                valid_index]
        else:
            if 'gt_masks' in results:
                results['gt_masks'] = PolygonMasks([], img_h, img_w)

        return results

    def segment2box(self, gt_masks: PolygonMasks, height: int,
                    width: int) -> HorizontalBoxes:
        """
        Convert 1 segment label to 1 box label, applying inside-image
        constraint i.e. (xy1, xy2, ...) to (xyxy)
        Args:
            gt_masks (torch.Tensor): the segment label
            width (int): the width of the image. Defaults to 640
            height (int): The height of the image. Defaults to 640
        Returns:
            HorizontalBoxes: the clip bboxes from gt_masks.
        """
        bboxes = []
        for _, poly_per_obj in enumerate(gt_masks):
            # simply use a number that is big enough for comparison with
            # coordinates
            xy_min = np.array([width * 2, height * 2], dtype=np.float32)
            xy_max = np.zeros(2, dtype=np.float32) - 1

            for p in poly_per_obj:
                xy = np.array(p).reshape(-1, 2).astype(np.float32)
                x, y = xy.T
                inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
                x, y = x[inside], y[inside]
                if not any(x):
                    continue
                xy = np.stack([x, y], axis=0).T

                xy_min = np.minimum(xy_min, np.min(xy, axis=0))
                xy_max = np.maximum(xy_max, np.max(xy, axis=0))
            if xy_max[0] == -1:
                bbox = np.zeros(4, dtype=np.float32)
            else:
                bbox = np.concatenate([xy_min, xy_max], axis=0)
            bboxes.append(bbox)

        return HorizontalBoxes(np.stack(bboxes, axis=0))

    # TODO: Move to mmdet
    def clip_polygons(self, gt_masks: PolygonMasks, height: int,
                      width: int) -> PolygonMasks:
        """Function to clip points of polygons with height and width.

        Args:
            gt_masks (PolygonMasks): Annotations of instance segmentation.
            height (int): height of clip border.
            width (int): width of clip border.
        Return:
            clipped_masks (PolygonMasks):
                Clip annotations of instance segmentation.
        """
        if len(gt_masks) == 0:
            clipped_masks = PolygonMasks([], height, width)
        else:
            clipped_masks = []
            for poly_per_obj in gt_masks:
                clipped_poly_per_obj = []
                for p in poly_per_obj:
                    p = p.copy()
                    p[0::2] = p[0::2].clip(0, width)
                    p[1::2] = p[1::2].clip(0, height)
                    clipped_poly_per_obj.append(p)
                clipped_masks.append(clipped_poly_per_obj)
            clipped_masks = PolygonMasks(clipped_masks, height, width)
        return clipped_masks

    @staticmethod
    def warp_poly(poly: np.ndarray, warp_matrix: np.ndarray, img_w: int,
                  img_h: int) -> np.ndarray:
        """Function to warp one mask and filter points outside image.

        Args:
            poly (np.ndarray): Segmentation annotation with shape (n, ) and
                with format (x1, y1, x2, y2, ...).
            warp_matrix (np.ndarray): Affine transformation matrix.
                Shape: (3, 3).
            img_w (int): Width of output image.
            img_h (int): Height of output image.
        """
        # TODO: Current logic may cause retained masks unusable for
        #  semantic segmentation training, which is same as official
        #  implementation.
        poly = poly.reshape((-1, 2))
        poly = np.concatenate((poly, np.ones(
            (len(poly), 1), dtype=poly.dtype)),
                              axis=-1)
        # transform poly
        poly = poly @ warp_matrix.T
        poly = poly[:, :2] / poly[:, 2:3]

        return poly.reshape(-1)

    def warp_mask(self, gt_masks: PolygonMasks, warp_matrix: np.ndarray,
                  img_w: int, img_h: int) -> PolygonMasks:
        """Warp masks by warp_matrix and retain masks inside image after
        warping.

        Args:
            gt_masks (PolygonMasks): Annotations of semantic segmentation.
            warp_matrix (np.ndarray): Affine transformation matrix.
                Shape: (3, 3).
            img_w (int): Width of output image.
            img_h (int): Height of output image.

        Returns:
            PolygonMasks: Masks after warping.
        """
        masks = gt_masks.masks

        new_masks = []
        for poly_per_obj in masks:
            warpped_poly_per_obj = []
            # One gt may have multiple masks.
            for poly in poly_per_obj:
                valid_poly = self.warp_poly(poly, warp_matrix, img_w, img_h)
                if len(valid_poly):
                    warpped_poly_per_obj.append(valid_poly.reshape(-1))
            # If all the masks are invalid,
            # add [0, 0, 0, 0, 0, 0,] here.
            if not warpped_poly_per_obj:
                # This will be filtered in function `filter_gt_bboxes`.
                warpped_poly_per_obj = [
                    np.zeros(6, dtype=poly_per_obj[0].dtype)
                ]
            new_masks.append(warpped_poly_per_obj)

        gt_masks = PolygonMasks(new_masks, img_h, img_w)
        return gt_masks

    def resample_masks(self, gt_masks: PolygonMasks) -> PolygonMasks:
        """Function to resample each mask annotation with shape (2 * n, ) to
        shape (resample_num * 2, ).

        Args:
            gt_masks (PolygonMasks): Annotations of semantic segmentation.
        """
        masks = gt_masks.masks
        new_masks = []
        for poly_per_obj in masks:
            resample_poly_per_obj = []
            for poly in poly_per_obj:
                poly = poly.reshape((-1, 2))  # xy
                poly = np.concatenate((poly, poly[0:1, :]), axis=0)
                x = np.linspace(0, len(poly) - 1, self.resample_num)
                xp = np.arange(len(poly))
                poly = np.concatenate([
                    np.interp(x, xp, poly[:, i]) for i in range(2)
                ]).reshape(2, -1).T.reshape(-1)
                resample_poly_per_obj.append(poly)
            new_masks.append(resample_poly_per_obj)
        return PolygonMasks(new_masks, gt_masks.height, gt_masks.width)

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


@TRANSFORMS.register_module()
class PPYOLOERandomDistort(BaseTransform):
    """Random hue, saturation, contrast and brightness distortion.

    Required Keys:

    - img

    Modified Keys:

    - img (np.float32)

    Args:
        hue_cfg (dict): Hue settings. Defaults to dict(min=-18,
            max=18, prob=0.5).
        saturation_cfg (dict): Saturation settings. Defaults to dict(
            min=0.5, max=1.5, prob=0.5).
        contrast_cfg (dict): Contrast settings. Defaults to dict(
            min=0.5, max=1.5, prob=0.5).
        brightness_cfg (dict): Brightness settings. Defaults to dict(
            min=0.5, max=1.5, prob=0.5).
        num_distort_func (int): The number of distort function. Defaults
            to 4.
    """

    def __init__(self,
                 hue_cfg: dict = dict(min=-18, max=18, prob=0.5),
                 saturation_cfg: dict = dict(min=0.5, max=1.5, prob=0.5),
                 contrast_cfg: dict = dict(min=0.5, max=1.5, prob=0.5),
                 brightness_cfg: dict = dict(min=0.5, max=1.5, prob=0.5),
                 num_distort_func: int = 4):
        self.hue_cfg = hue_cfg
        self.saturation_cfg = saturation_cfg
        self.contrast_cfg = contrast_cfg
        self.brightness_cfg = brightness_cfg
        self.num_distort_func = num_distort_func
        assert 0 < self.num_distort_func <= 4, \
            'num_distort_func must > 0 and <= 4'
        for cfg in [
                self.hue_cfg, self.saturation_cfg, self.contrast_cfg,
                self.brightness_cfg
        ]:
            assert 0. <= cfg['prob'] <= 1., 'prob must >=0 and <=1'

    def transform_hue(self, results):
        """Transform hue randomly."""
        if random.uniform(0., 1.) >= self.hue_cfg['prob']:
            return results
        img = results['img']
        delta = random.uniform(self.hue_cfg['min'], self.hue_cfg['max'])
        u = np.cos(delta * np.pi)
        w = np.sin(delta * np.pi)
        delta_iq = np.array([[1.0, 0.0, 0.0], [0.0, u, -w], [0.0, w, u]])
        rgb2yiq_matrix = np.array([[0.114, 0.587, 0.299],
                                   [-0.321, -0.274, 0.596],
                                   [0.311, -0.523, 0.211]])
        yiq2rgb_matric = np.array([[1.0, -1.107, 1.705], [1.0, -0.272, -0.647],
                                   [1.0, 0.956, 0.621]])
        t = np.dot(np.dot(yiq2rgb_matric, delta_iq), rgb2yiq_matrix).T
        img = np.dot(img, t)
        results['img'] = img
        return results

    def transform_saturation(self, results):
        """Transform saturation randomly."""
        if random.uniform(0., 1.) >= self.saturation_cfg['prob']:
            return results
        img = results['img']
        delta = random.uniform(self.saturation_cfg['min'],
                               self.saturation_cfg['max'])

        # convert bgr img to gray img
        gray = img * np.array([[[0.114, 0.587, 0.299]]], dtype=np.float32)
        gray = gray.sum(axis=2, keepdims=True)
        gray *= (1.0 - delta)
        img *= delta
        img += gray
        results['img'] = img
        return results

    def transform_contrast(self, results):
        """Transform contrast randomly."""
        if random.uniform(0., 1.) >= self.contrast_cfg['prob']:
            return results
        img = results['img']
        delta = random.uniform(self.contrast_cfg['min'],
                               self.contrast_cfg['max'])
        img *= delta
        results['img'] = img
        return results

    def transform_brightness(self, results):
        """Transform brightness randomly."""
        if random.uniform(0., 1.) >= self.brightness_cfg['prob']:
            return results
        img = results['img']
        delta = random.uniform(self.brightness_cfg['min'],
                               self.brightness_cfg['max'])
        img += delta
        results['img'] = img
        return results

    def transform(self, results: dict) -> dict:
        """The hue, saturation, contrast and brightness distortion function.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        results['img'] = results['img'].astype(np.float32)

        functions = [
            self.transform_brightness, self.transform_contrast,
            self.transform_saturation, self.transform_hue
        ]
        distortions = random.permutation(functions)[:self.num_distort_func]
        for func in distortions:
            results = func(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(hue_cfg={self.hue_cfg}, '
        repr_str += f'saturation_cfg={self.saturation_cfg}, '
        repr_str += f'contrast_cfg={self.contrast_cfg}, '
        repr_str += f'brightness_cfg={self.brightness_cfg}, '
        repr_str += f'num_distort_func={self.num_distort_func})'
        return repr_str


@TRANSFORMS.register_module()
class PPYOLOERandomCrop(BaseTransform):
    """Random crop the img and bboxes. Different thresholds are used in PPYOLOE
    to judge whether the clipped image meets the requirements. This
    implementation is different from the implementation of RandomCrop in mmdet.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)

    Added Keys:
    - pad_param (np.float32)

    Args:
        aspect_ratio (List[float]): Aspect ratio of cropped region. Default to
             [.5, 2].
        thresholds (List[float]): Iou thresholds for deciding a valid bbox crop
            in [min, max] format. Defaults to [.0, .1, .3, .5, .7, .9].
        scaling (List[float]): Ratio between a cropped region and the original
            image in [min, max] format. Default to [.3, 1.].
        num_attempts (int): Number of tries for each threshold before
            giving up. Default to 50.
        allow_no_crop (bool): Allow return without actually cropping them.
            Default to True.
        cover_all_box (bool): Ensure all bboxes are covered in the final crop.
            Default to False.
    """

    def __init__(self,
                 aspect_ratio: List[float] = [.5, 2.],
                 thresholds: List[float] = [.0, .1, .3, .5, .7, .9],
                 scaling: List[float] = [.3, 1.],
                 num_attempts: int = 50,
                 allow_no_crop: bool = True,
                 cover_all_box: bool = False):
        self.aspect_ratio = aspect_ratio
        self.thresholds = thresholds
        self.scaling = scaling
        self.num_attempts = num_attempts
        self.allow_no_crop = allow_no_crop
        self.cover_all_box = cover_all_box

    def _crop_data(self, results: dict, crop_box: Tuple[int, int, int, int],
                   valid_inds: np.ndarray) -> Union[dict, None]:
        """Function to randomly crop images, bounding boxes, masks, semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
            crop_box (Tuple[int, int, int, int]): Expected absolute coordinates
                for cropping, (x1, y1, x2, y2).
            valid_inds (np.ndarray): The indexes of gt that needs to be
                retained.

        Returns:
            results (Union[dict, None]): Randomly cropped results, 'img_shape'
                key in result dict is updated according to crop size. None will
                be returned when there is no valid bbox after cropping.
        """
        # crop the image
        img = results['img']
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        results['img'] = img
        img_shape = img.shape
        results['img_shape'] = img.shape

        # crop bboxes accordingly and clip to the image boundary
        if results.get('gt_bboxes', None) is not None:
            bboxes = results['gt_bboxes']
            bboxes.translate_([-crop_x1, -crop_y1])
            bboxes.clip_(img_shape[:2])

            results['gt_bboxes'] = bboxes[valid_inds]

            if results.get('gt_ignore_flags', None) is not None:
                results['gt_ignore_flags'] = \
                    results['gt_ignore_flags'][valid_inds]

            if results.get('gt_bboxes_labels', None) is not None:
                results['gt_bboxes_labels'] = \
                    results['gt_bboxes_labels'][valid_inds]

            if results.get('gt_masks', None) is not None:
                results['gt_masks'] = results['gt_masks'][
                    valid_inds.nonzero()[0]].crop(
                        np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))

        # crop semantic seg
        if results.get('gt_seg_map', None) is not None:
            results['gt_seg_map'] = results['gt_seg_map'][crop_y1:crop_y2,
                                                          crop_x1:crop_x2]

        return results

    @autocast_box_type()
    def transform(self, results: dict) -> Union[dict, None]:
        """The random crop transform function.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        if results.get('gt_bboxes', None) is None or len(
                results['gt_bboxes']) == 0:
            return results

        orig_img_h, orig_img_w = results['img'].shape[:2]
        gt_bboxes = results['gt_bboxes']

        thresholds = list(self.thresholds)
        if self.allow_no_crop:
            thresholds.append('no_crop')
        random.shuffle(thresholds)

        for thresh in thresholds:
            # Determine the coordinates for cropping
            if thresh == 'no_crop':
                return results

            found = False
            for i in range(self.num_attempts):
                crop_h, crop_w = self._get_crop_size((orig_img_h, orig_img_w))
                if self.aspect_ratio is None:
                    if crop_h / crop_w < 0.5 or crop_h / crop_w > 2.0:
                        continue

                # get image crop_box
                margin_h = max(orig_img_h - crop_h, 0)
                margin_w = max(orig_img_w - crop_w, 0)
                offset_h, offset_w = self._rand_offset((margin_h, margin_w))
                crop_y1, crop_y2 = offset_h, offset_h + crop_h
                crop_x1, crop_x2 = offset_w, offset_w + crop_w

                crop_box = [crop_x1, crop_y1, crop_x2, crop_y2]
                # Calculate the iou between gt_bboxes and crop_boxes
                iou = self._iou_matrix(gt_bboxes,
                                       np.array([crop_box], dtype=np.float32))
                # If the maximum value of the iou is less than thresh,
                # the current crop_box is considered invalid.
                if iou.max() < thresh:
                    continue

                # If cover_all_box == True and the minimum value of
                # the iou is less than thresh, the current crop_box
                # is considered invalid.
                if self.cover_all_box and iou.min() < thresh:
                    continue

                # Get which gt_bboxes to keep after cropping.
                valid_inds = self._get_valid_inds(
                    gt_bboxes, np.array(crop_box, dtype=np.float32))
                if valid_inds.size > 0:
                    found = True
                    break

            if found:
                results = self._crop_data(results, crop_box, valid_inds)
                return results
        return results

    @cache_randomness
    def _rand_offset(self, margin: Tuple[int, int]) -> Tuple[int, int]:
        """Randomly generate crop offset.

        Args:
            margin (Tuple[int, int]): The upper bound for the offset generated
                randomly.

        Returns:
            Tuple[int, int]: The random offset for the crop.
        """
        margin_h, margin_w = margin
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)

        return (offset_h, offset_w)

    @cache_randomness
    def _get_crop_size(self, image_size: Tuple[int, int]) -> Tuple[int, int]:
        """Randomly generates the crop size based on `image_size`.

        Args:
            image_size (Tuple[int, int]): (h, w).

        Returns:
            crop_size (Tuple[int, int]): (crop_h, crop_w) in absolute pixels.
        """
        h, w = image_size
        scale = random.uniform(*self.scaling)
        if self.aspect_ratio is not None:
            min_ar, max_ar = self.aspect_ratio
            aspect_ratio = random.uniform(
                max(min_ar, scale**2), min(max_ar, scale**-2))
            h_scale = scale / np.sqrt(aspect_ratio)
            w_scale = scale * np.sqrt(aspect_ratio)
        else:
            h_scale = random.uniform(*self.scaling)
            w_scale = random.uniform(*self.scaling)
        crop_h = h * h_scale
        crop_w = w * w_scale
        return int(crop_h), int(crop_w)

    def _iou_matrix(self,
                    gt_bbox: HorizontalBoxes,
                    crop_bbox: np.ndarray,
                    eps: float = 1e-10) -> np.ndarray:
        """Calculate iou between gt and image crop box.

        Args:
            gt_bbox (HorizontalBoxes): Ground truth bounding boxes.
            crop_bbox (np.ndarray): Image crop coordinates in
                [x1, y1, x2, y2] format.
            eps (float): Default to 1e-10.
        Return:
            (np.ndarray): IoU.
        """
        gt_bbox = gt_bbox.tensor.numpy()
        lefttop = np.maximum(gt_bbox[:, np.newaxis, :2], crop_bbox[:, :2])
        rightbottom = np.minimum(gt_bbox[:, np.newaxis, 2:], crop_bbox[:, 2:])

        overlap = np.prod(
            rightbottom - lefttop,
            axis=2) * (lefttop < rightbottom).all(axis=2)
        area_gt_bbox = np.prod(gt_bbox[:, 2:] - crop_bbox[:, :2], axis=1)
        area_crop_bbox = np.prod(gt_bbox[:, 2:] - crop_bbox[:, :2], axis=1)
        area_o = (area_gt_bbox[:, np.newaxis] + area_crop_bbox - overlap)
        return overlap / (area_o + eps)

    def _get_valid_inds(self, gt_bbox: HorizontalBoxes,
                        img_crop_bbox: np.ndarray) -> np.ndarray:
        """Get which Bboxes to keep at the current cropping coordinates.

        Args:
            gt_bbox (HorizontalBoxes): Ground truth bounding boxes.
            img_crop_bbox (np.ndarray): Image crop coordinates in
                [x1, y1, x2, y2] format.

        Returns:
            (np.ndarray): Valid indexes.
        """
        cropped_box = gt_bbox.tensor.numpy().copy()
        gt_bbox = gt_bbox.tensor.numpy().copy()

        cropped_box[:, :2] = np.maximum(gt_bbox[:, :2], img_crop_bbox[:2])
        cropped_box[:, 2:] = np.minimum(gt_bbox[:, 2:], img_crop_bbox[2:])
        cropped_box[:, :2] -= img_crop_bbox[:2]
        cropped_box[:, 2:] -= img_crop_bbox[:2]

        centers = (gt_bbox[:, :2] + gt_bbox[:, 2:]) / 2
        valid = np.logical_and(img_crop_bbox[:2] <= centers,
                               centers < img_crop_bbox[2:]).all(axis=1)
        valid = np.logical_and(
            valid, (cropped_box[:, :2] < cropped_box[:, 2:]).all(axis=1))

        return np.where(valid)[0]

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(aspect_ratio={self.aspect_ratio}, '
        repr_str += f'thresholds={self.thresholds}, '
        repr_str += f'scaling={self.scaling}, '
        repr_str += f'num_attempts={self.num_attempts}, '
        repr_str += f'allow_no_crop={self.allow_no_crop}, '
        repr_str += f'cover_all_box={self.cover_all_box})'
        return repr_str


@TRANSFORMS.register_module()
class YOLOv5CopyPaste(BaseTransform):
    """Copy-Paste used in YOLOv5 and YOLOv8.

    This transform randomly copy some objects in the image to the mirror
    position of the image.It is different from the `CopyPaste` in mmdet.

    Required Keys:

    - img (np.uint8)
    - gt_bboxes (BaseBoxes[torch.float32])
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)
    - gt_masks (PolygonMasks) (optional)

    Modified Keys:

    - img
    - gt_bboxes
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (optional)
    - gt_masks (optional)

    Args:
        ioa_thresh (float): Ioa thresholds for deciding valid bbox.
        prob (float): Probability of choosing objects.
            Defaults to 0.5.
    """

    def __init__(self, ioa_thresh: float = 0.3, prob: float = 0.5):
        self.ioa_thresh = ioa_thresh
        self.prob = prob

    @autocast_box_type()
    def transform(self, results: dict) -> Union[dict, None]:
        """The YOLOv5 and YOLOv8 Copy-Paste transform function.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        if len(results.get('gt_masks', [])) == 0:
            return results
        gt_masks = results['gt_masks']
        assert isinstance(gt_masks, PolygonMasks), \
            'only support type of PolygonMasks,' \
            ' but get type: %s' % type(gt_masks)
        gt_bboxes = results['gt_bboxes']
        gt_bboxes_labels = results.get('gt_bboxes_labels', None)
        img = results['img']
        img_h, img_w = img.shape[:2]

        # calculate ioa
        gt_bboxes_flip = deepcopy(gt_bboxes)
        gt_bboxes_flip.flip_(img.shape)

        ioa = self.bbox_ioa(gt_bboxes_flip, gt_bboxes)
        indexes = torch.nonzero((ioa < self.ioa_thresh).all(1))[:, 0]
        n = len(indexes)
        valid_inds = random.choice(
            indexes, size=round(self.prob * n), replace=False)
        if len(valid_inds) == 0:
            return results

        if gt_bboxes_labels is not None:
            # prepare labels
            gt_bboxes_labels = np.concatenate(
                (gt_bboxes_labels, gt_bboxes_labels[valid_inds]), axis=0)

        # prepare bboxes
        copypaste_bboxes = gt_bboxes_flip[valid_inds]
        gt_bboxes = gt_bboxes.cat([gt_bboxes, copypaste_bboxes])

        # prepare images
        copypaste_gt_masks = gt_masks[valid_inds]
        copypaste_gt_masks_flip = copypaste_gt_masks.flip()
        # convert poly format to bitmap format
        # example: poly: [[array(0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]]
        #  -> bitmap: a mask with shape equal to (1, img_h, img_w)
        # # type1 low speed
        # copypaste_gt_masks_bitmap = copypaste_gt_masks.to_ndarray()
        # copypaste_mask = np.sum(copypaste_gt_masks_bitmap, axis=0) > 0

        # type2
        copypaste_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        for poly in copypaste_gt_masks.masks:
            poly = [i.reshape((-1, 1, 2)).astype(np.int32) for i in poly]
            cv2.drawContours(copypaste_mask, poly, -1, (1, ), cv2.FILLED)

        copypaste_mask = copypaste_mask.astype(bool)

        # copy objects, and paste to the mirror position of the image
        copypaste_mask_flip = mmcv.imflip(
            copypaste_mask, direction='horizontal')
        copypaste_img = mmcv.imflip(img, direction='horizontal')
        img[copypaste_mask_flip] = copypaste_img[copypaste_mask_flip]

        # prepare masks
        gt_masks = copypaste_gt_masks.cat([gt_masks, copypaste_gt_masks_flip])

        if 'gt_ignore_flags' in results:
            # prepare gt_ignore_flags
            gt_ignore_flags = results['gt_ignore_flags']
            gt_ignore_flags = np.concatenate(
                [gt_ignore_flags, gt_ignore_flags[valid_inds]], axis=0)
            results['gt_ignore_flags'] = gt_ignore_flags

        results['img'] = img
        results['gt_bboxes'] = gt_bboxes
        if gt_bboxes_labels is not None:
            results['gt_bboxes_labels'] = gt_bboxes_labels
        results['gt_masks'] = gt_masks

        return results

    @staticmethod
    def bbox_ioa(gt_bboxes_flip: HorizontalBoxes,
                 gt_bboxes: HorizontalBoxes,
                 eps: float = 1e-7) -> np.ndarray:
        """Calculate ioa between gt_bboxes_flip and gt_bboxes.

        Args:
            gt_bboxes_flip (HorizontalBoxes): Flipped ground truth
                bounding boxes.
            gt_bboxes (HorizontalBoxes): Ground truth bounding boxes.
            eps (float): Default to 1e-10.
        Return:
            (Tensor): Ioa.
        """
        gt_bboxes_flip = gt_bboxes_flip.tensor
        gt_bboxes = gt_bboxes.tensor

        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = gt_bboxes_flip.T
        b2_x1, b2_y1, b2_x2, b2_y2 = gt_bboxes.T

        # Intersection area
        inter_area = (torch.minimum(b1_x2[:, None],
                                    b2_x2) - torch.maximum(b1_x1[:, None],
                                                           b2_x1)).clip(0) * \
                     (torch.minimum(b1_y2[:, None],
                                    b2_y2) - torch.maximum(b1_y1[:, None],
                                                           b2_y1)).clip(0)

        # box2 area
        box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

        # Intersection over box2 area
        return inter_area / box2_area

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(ioa_thresh={self.ioa_thresh},'
        repr_str += f'prob={self.prob})'
        return repr_str


@TRANSFORMS.register_module()
class RemoveDataElement(BaseTransform):
    """Remove unnecessary data element in results.

    Args:
        keys (Union[str, Sequence[str]]): Keys need to be removed.
    """

    def __init__(self, keys: Union[str, Sequence[str]]):
        self.keys = [keys] if isinstance(keys, str) else keys

    def transform(self, results: dict) -> dict:
        for key in self.keys:
            results.pop(key, None)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(keys={self.keys})'
        return repr_str


@TRANSFORMS.register_module()
class RegularizeRotatedBox(BaseTransform):
    """Regularize rotated boxes.

    Due to the angle periodicity, one rotated box can be represented in
    many different (x, y, w, h, t). To make each rotated box unique,
    ``regularize_boxes`` will take the remainder of the angle divided by
    180 degrees.

    For convenience, three angle_version can be used here:

    - 'oc': OpenCV Definition. Has the same box representation as
        ``cv2.minAreaRect`` the angle ranges in [-90, 0).
    - 'le90': Long Edge Definition (90). the angle ranges in [-90, 90).
        The width is always longer than the height.
    - 'le135': Long Edge Definition (135). the angle ranges in [-45, 135).
        The width is always longer than the height.

    Required Keys:

    - gt_bboxes (RotatedBoxes[torch.float32])

    Modified Keys:

    - gt_bboxes

    Args:
        angle_version (str): Angle version. Can only be 'oc',
            'le90', or 'le135'. Defaults to 'le90.
    """

    def __init__(self, angle_version='le90') -> None:
        self.angle_version = angle_version
        try:
            from mmrotate.structures.bbox import RotatedBoxes
            self.box_type = RotatedBoxes
        except ImportError:
            raise ImportError(
                'Please run "mim install -r requirements/mmrotate.txt" '
                'to install mmrotate first for rotated detection.')

    def transform(self, results: dict) -> dict:
        assert isinstance(results['gt_bboxes'], self.box_type)
        results['gt_bboxes'] = self.box_type(
            results['gt_bboxes'].regularize_boxes(self.angle_version))
        return results


@TRANSFORMS.register_module()
class Polygon2Mask(BaseTransform):
    """Polygons to bitmaps in YOLOv5.

    Args:
        downsample_ratio (int): Downsample ratio of mask.
        mask_overlap (bool): Whether to use maskoverlap in mask process.
            When set to True, the implementation here is the same as the
            official, with higher training speed. If set to True, all gt masks
            will compress into one overlap mask, the value of mask indicates
            the index of gt masks. If set to False, one mask is a binary mask.
            Default to True.
        coco_style (bool): Whether to use coco_style to convert the polygons to
            bitmaps. Note that this option is only used to test if there is an
            improvement in training speed and we recommend setting it to False.
    """

    def __init__(self,
                 downsample_ratio: int = 4,
                 mask_overlap: bool = True,
                 coco_style: bool = False):
        self.downsample_ratio = downsample_ratio
        self.mask_overlap = mask_overlap
        self.coco_style = coco_style

    def polygon2mask(self,
                     img_shape: Tuple[int, int],
                     polygons: np.ndarray,
                     color: int = 1) -> np.ndarray:
        """
        Args:
            img_shape (tuple): The image size.
            polygons (np.ndarray): [N, M], N is the number of polygons,
                M is the number of points(Be divided by 2).
            color (int): color in fillPoly.
        Return:
            np.ndarray: the overlap mask.
        """
        nh, nw = (img_shape[0] // self.downsample_ratio,
                  img_shape[1] // self.downsample_ratio)
        if self.coco_style:
            # This practice can lead to the loss of small objects
            # polygons = polygons.resize((nh, nw)).masks
            # polygons = np.asarray(polygons).reshape(-1)
            # mask = polygon_to_bitmap([polygons], nh, nw)

            polygons = np.asarray(polygons).reshape(-1)
            mask = polygon_to_bitmap([polygons], img_shape[0],
                                     img_shape[1]).astype(np.uint8)
            mask = mmcv.imresize(mask, (nw, nh))
        else:
            mask = np.zeros(img_shape, dtype=np.uint8)
            polygons = np.asarray(polygons)
            polygons = polygons.astype(np.int32)
            shape = polygons.shape
            polygons = polygons.reshape(shape[0], -1, 2)
            cv2.fillPoly(mask, polygons, color=color)
            # NOTE: fillPoly firstly then resize is trying the keep the same
            #  way of loss calculation when mask-ratio=1.
            mask = mmcv.imresize(mask, (nw, nh))
        return mask

    def polygons2masks(self,
                       img_shape: Tuple[int, int],
                       polygons: PolygonMasks,
                       color: int = 1) -> np.ndarray:
        """Return a list of bitmap masks.

        Args:
            img_shape (tuple): The image size.
            polygons (PolygonMasks): The mask annotations.
            color (int): color in fillPoly.
        Return:
            List[np.ndarray]: the list of masks in bitmaps.
        """
        if self.coco_style:
            nh, nw = (img_shape[0] // self.downsample_ratio,
                      img_shape[1] // self.downsample_ratio)
            masks = polygons.resize((nh, nw)).to_ndarray()
            return masks
        else:
            masks = []
            for si in range(len(polygons)):
                mask = self.polygon2mask(img_shape, polygons[si], color)
                masks.append(mask)
            return np.array(masks)

    def polygons2masks_overlap(
            self, img_shape: Tuple[int, int],
            polygons: PolygonMasks) -> Tuple[np.ndarray, np.ndarray]:
        """Return a overlap mask and the sorted idx of area.

        Args:
            img_shape (tuple): The image size.
            polygons (PolygonMasks): The mask annotations.
            color (int): color in fillPoly.
        Return:
            Tuple[np.ndarray, np.ndarray]:
                the overlap mask and the sorted idx of area.
        """
        masks = np.zeros((img_shape[0] // self.downsample_ratio,
                          img_shape[1] // self.downsample_ratio),
                         dtype=np.int32 if len(polygons) > 255 else np.uint8)
        areas = []
        ms = []
        for si in range(len(polygons)):
            mask = self.polygon2mask(img_shape, polygons[si], color=1)
            ms.append(mask)
            areas.append(mask.sum())
        areas = np.asarray(areas)
        index = np.argsort(-areas)
        ms = np.array(ms)[index]
        for i in range(len(polygons)):
            mask = ms[i] * (i + 1)
            masks = masks + mask
            masks = np.clip(masks, a_min=0, a_max=i + 1)
        return masks, index

    def transform(self, results: dict) -> dict:
        gt_masks = results['gt_masks']
        assert isinstance(gt_masks, PolygonMasks)

        if self.mask_overlap:
            masks, sorted_idx = self.polygons2masks_overlap(
                (gt_masks.height, gt_masks.width), gt_masks)
            results['gt_bboxes'] = results['gt_bboxes'][sorted_idx]
            results['gt_bboxes_labels'] = results['gt_bboxes_labels'][
                sorted_idx]

            # In this case we put gt_masks in gt_panoptic_seg
            results.pop('gt_masks')
            results['gt_panoptic_seg'] = torch.from_numpy(masks[None])
        else:
            masks = self.polygons2masks((gt_masks.height, gt_masks.width),
                                        gt_masks,
                                        color=1)
            masks = torch.from_numpy(masks)
            # Consistent logic with mmdet
            results['gt_masks'] = masks
        return results


@TRANSFORMS.register_module()
class FilterAnnotations(FilterDetAnnotations):
    """Filter invalid annotations.

    In addition to the conditions checked by ``FilterDetAnnotations``, this
    filter adds a new condition requiring instances to have at least one
    visible keypoints.
    """

    def __init__(self, by_keypoints: bool = False, **kwargs) -> None:
        # TODO: add more filter options
        super().__init__(**kwargs)
        self.by_keypoints = by_keypoints

    @autocast_box_type()
    def transform(self, results: dict) -> Union[dict, None]:
        """Transform function to filter annotations.

        Args:
            results (dict): Result dict.
        Returns:
            dict: Updated result dict.
        """
        assert 'gt_bboxes' in results
        gt_bboxes = results['gt_bboxes']
        if gt_bboxes.shape[0] == 0:
            return results

        tests = []
        if self.by_box:
            tests.append(
                ((gt_bboxes.widths > self.min_gt_bbox_wh[0]) &
                 (gt_bboxes.heights > self.min_gt_bbox_wh[1])).numpy())

        if self.by_mask:
            assert 'gt_masks' in results
            gt_masks = results['gt_masks']
            tests.append(gt_masks.areas >= self.min_gt_mask_area)

        if self.by_keypoints:
            assert 'gt_keypoints' in results
            num_keypoints = results['gt_keypoints'].num_keypoints
            tests.append((num_keypoints > 0).numpy())

        keep = tests[0]
        for t in tests[1:]:
            keep = keep & t

        if not keep.any():
            if self.keep_empty:
                return None

        keys = ('gt_bboxes', 'gt_bboxes_labels', 'gt_masks', 'gt_ignore_flags',
                'gt_keypoints')
        for key in keys:
            if key in results:
                results[key] = results[key][keep]

        return results


# TODO: Check if it can be merged with mmdet.YOLOXHSVRandomAug
@TRANSFORMS.register_module()
class RandomAffine(MMDET_RandomAffine):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        img = results['img']
        height = img.shape[0] + self.border[1] * 2
        width = img.shape[1] + self.border[0] * 2

        warp_matrix = self._get_random_homography_matrix(height, width)

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
            bboxes.project_(warp_matrix)
            if self.bbox_clip_border:
                bboxes.clip_([height, width])
            # remove outside bbox
            valid_index = bboxes.is_inside([height, width]).numpy()
            results['gt_bboxes'] = bboxes[valid_index]
            results['gt_bboxes_labels'] = results['gt_bboxes_labels'][
                valid_index]
            results['gt_ignore_flags'] = results['gt_ignore_flags'][
                valid_index]

            if 'gt_masks' in results:
                raise NotImplementedError('RandomAffine only supports bbox.')

            if 'gt_keypoints' in results:
                keypoints = results['gt_keypoints']
                keypoints.project_(warp_matrix)
                if self.bbox_clip_border:
                    keypoints.clip_([height, width])
                results['gt_keypoints'] = keypoints[valid_index]

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(hue_delta={self.hue_delta}, '
        repr_str += f'saturation_delta={self.saturation_delta}, '
        repr_str += f'value_delta={self.value_delta})'
        return repr_str


# TODO: Check if it can be merged with mmdet.YOLOXHSVRandomAug
@TRANSFORMS.register_module()
class RandomFlip(MMDET_RandomFlip):

    @autocast_box_type()
    def _flip(self, results: dict) -> None:
        """Flip images, bounding boxes, and semantic segmentation map."""
        # flip image
        results['img'] = mmcv.imflip(
            results['img'], direction=results['flip_direction'])

        img_shape = results['img'].shape[:2]

        # flip bboxes
        if results.get('gt_bboxes', None) is not None:
            results['gt_bboxes'].flip_(img_shape, results['flip_direction'])

        # flip keypoints
        if results.get('gt_keypoints', None) is not None:
            results['gt_keypoints'].flip_(img_shape, results['flip_direction'])

        # flip masks
        if results.get('gt_masks', None) is not None:
            results['gt_masks'] = results['gt_masks'].flip(
                results['flip_direction'])

        # flip segs
        if results.get('gt_seg_map', None) is not None:
            results['gt_seg_map'] = mmcv.imflip(
                results['gt_seg_map'], direction=results['flip_direction'])

        # record homography matrix for flip
        self._record_homography_matrix(results)


@TRANSFORMS.register_module()
class Resize(MMDET_Resize):

    def _resize_keypoints(self, results: dict) -> None:
        """Resize bounding boxes with ``results['scale_factor']``."""
        if results.get('gt_keypoints', None) is not None:
            results['gt_keypoints'].rescale_(results['scale_factor'])
            if self.clip_object_border:
                results['gt_keypoints'].clip_(results['img_shape'])

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Transform function to resize images, bounding boxes and semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img', 'gt_bboxes', 'gt_seg_map',
            'scale', 'scale_factor', 'height', 'width', and 'keep_ratio' keys
            are updated in result dict.
        """
        if self.scale:
            results['scale'] = self.scale
        else:
            img_shape = results['img'].shape[:2]
            results['scale'] = _scale_size(img_shape[::-1], self.scale_factor)
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_keypoints(results)
        self._resize_masks(results)
        self._resize_seg(results)
        self._record_homography_matrix(results)
        return results
