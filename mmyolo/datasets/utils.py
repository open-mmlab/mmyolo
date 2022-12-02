# Copyright (c) OpenMMLab. All rights reserved.
import uuid
from numbers import Integral
from typing import List, Sequence

import cv2
import numpy as np
import torch
from mmengine.dataset import COLLATE_FUNCTIONS

from ..registry import TASK_UTILS


@COLLATE_FUNCTIONS.register_module()
def yolov5_collate(data_batch: Sequence) -> dict:
    """Rewrite collate_fn to get faster training speed."""
    batch_imgs = []
    batch_bboxes_labels = []
    for i in range(len(data_batch)):
        datasamples = data_batch[i]['data_samples']
        inputs = data_batch[i]['inputs']

        gt_bboxes = datasamples.gt_instances.bboxes.tensor
        gt_labels = datasamples.gt_instances.labels
        batch_idx = gt_labels.new_full((len(gt_labels), 1), i)
        bboxes_labels = torch.cat((batch_idx, gt_labels[:, None], gt_bboxes),
                                  dim=1)
        batch_bboxes_labels.append(bboxes_labels)

        batch_imgs.append(inputs)
    return {
        'inputs': torch.stack(batch_imgs, 0),
        'data_samples': torch.cat(batch_bboxes_labels, 0)
    }


@COLLATE_FUNCTIONS.register_module()
def ppyoloe_collate(data_batch: Sequence) -> dict:
    """Rewrite collate_fn to get faster training speed."""
    batch_imgs = []
    batch_bboxes_labels = []
    for i in range(len(data_batch)):
        datasamples = data_batch[i]['data_samples']
        inputs = data_batch[i]['inputs']

        gt_bboxes = datasamples.gt_instances.bboxes.tensor
        gt_labels = datasamples.gt_instances.labels
        batch_idx = gt_labels.new_full((len(gt_labels), 1), i)
        bboxes_labels = torch.cat((batch_idx, gt_labels[:, None], gt_bboxes),
                                  dim=1)
        batch_bboxes_labels.append(bboxes_labels)

        batch_imgs.append(inputs)

    return {
        # type: list
        'inputs': batch_imgs,
        'data_samples': batch_bboxes_labels
    }


@TASK_UTILS.register_module()
class BatchShapePolicy:
    """BatchShapePolicy is only used in the testing phase, which can reduce the
    number of pad pixels during batch inference.

    Args:
       batch_size (int): Single GPU batch size during batch inference.
           Defaults to 32.
       img_size (int): Expected output image size. Defaults to 640.
       size_divisor (int): The minimum size that is divisible
           by size_divisor. Defaults to 32.
       extra_pad_ratio (float):  Extra pad ratio. Defaults to 0.5.
    """

    def __init__(self,
                 batch_size: int = 32,
                 img_size: int = 640,
                 size_divisor: int = 32,
                 extra_pad_ratio: float = 0.5):
        self.batch_size = batch_size
        self.img_size = img_size
        self.size_divisor = size_divisor
        self.extra_pad_ratio = extra_pad_ratio

    def __call__(self, data_list: List[dict]) -> List[dict]:
        image_shapes = []
        for data_info in data_list:
            image_shapes.append((data_info['width'], data_info['height']))

        image_shapes = np.array(image_shapes, dtype=np.float64)

        n = len(image_shapes)  # number of images
        batch_index = np.floor(np.arange(n) / self.batch_size).astype(
            np.int)  # batch index
        number_of_batches = batch_index[-1] + 1  # number of batches

        aspect_ratio = image_shapes[:, 1] / image_shapes[:, 0]  # aspect ratio
        irect = aspect_ratio.argsort()

        data_list = [data_list[i] for i in irect]

        aspect_ratio = aspect_ratio[irect]
        # Set training image shapes
        shapes = [[1, 1]] * number_of_batches
        for i in range(number_of_batches):
            aspect_ratio_index = aspect_ratio[batch_index == i]
            min_index, max_index = aspect_ratio_index.min(
            ), aspect_ratio_index.max()
            if max_index < 1:
                shapes[i] = [max_index, 1]
            elif min_index > 1:
                shapes[i] = [1, 1 / min_index]

        batch_shapes = np.ceil(
            np.array(shapes) * self.img_size / self.size_divisor +
            self.extra_pad_ratio).astype(np.int) * self.size_divisor

        for i, data_info in enumerate(data_list):
            data_info['batch_shape'] = batch_shapes[batch_index[i]]

        return data_list


class BaseOperator(object):
    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__
        self._id = name + '_' + str(uuid.uuid4())[-6:]

    def apply(self, sample, context=None):
        """ Process a sample.
        Args:
            sample (dict): a dict of sample, eg: {'image':xx, 'label': xxx}
            context (dict): info about this sample processing
        Returns:
            result (dict): a processed sample
        """
        return sample

    def __call__(self, sample, context=None):
        """ Process a sample.
        Args:
            sample (dict): a dict of sample, eg: {'image':xx, 'label': xxx}
            context (dict): info about this sample processing
        Returns:
            result (dict): a processed sample
        """
        if isinstance(sample, Sequence):
            for i in range(len(sample)):
                sample[i] = self.apply(sample[i], context)
        else:
            sample = self.apply(sample, context)
        return sample

    def __str__(self):
        return str(self._id)


class Resize(BaseOperator):

    def __init__(self, target_size, keep_ratio, interp=cv2.INTER_LINEAR):
        """
        Resize image to target size. if keep_ratio is True,
        resize the image's long side to the maximum of target_size
        if keep_ratio is False, resize the image to target size(h, w)
        Args:
            target_size (int|list): image target size
            keep_ratio (bool): whether keep_ratio or not, default true
            interp (int): the interpolation method
        """
        super(Resize, self).__init__()
        self.keep_ratio = keep_ratio
        self.interp = interp
        if isinstance(target_size, Integral):
            target_size = [target_size, target_size]
        self.target_size = target_size

    def apply_image(self, image, scale):
        im_scale_x, im_scale_y = scale

        return cv2.resize(
            image,
            None,
            None,
            fx=im_scale_x,
            fy=im_scale_y,
            interpolation=self.interp)

    def apply_bbox(self, bbox, scale, size):
        im_scale_x, im_scale_y = scale
        resize_w, resize_h = size
        bbox[:, 0::2] *= im_scale_x
        bbox[:, 1::2] *= im_scale_y
        bbox[:, 0::2] = np.clip(bbox[:, 0::2], 0, resize_w)
        bbox[:, 1::2] = np.clip(bbox[:, 1::2], 0, resize_h)
        return bbox

    def apply(self, sample, context=None):
        """ Resize the image numpy.
        """
        im = sample['img']
        if not isinstance(im, np.ndarray):
            raise TypeError("{}: image type is not numpy.".format(self))

        # apply image
        im_shape = im.shape
        if self.keep_ratio:

            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])

            target_size_min = np.min(self.target_size)
            target_size_max = np.max(self.target_size)

            im_scale = min(target_size_min / im_size_min,
                           target_size_max / im_size_max)

            resize_h = im_scale * float(im_shape[0])
            resize_w = im_scale * float(im_shape[1])

            im_scale_x = im_scale
            im_scale_y = im_scale
        else:
            resize_h, resize_w = self.target_size
            im_scale_y = resize_h / im_shape[0]
            im_scale_x = resize_w / im_shape[1]

        im = self.apply_image(sample['img'], [im_scale_x, im_scale_y])
        sample['img'] = im
        sample['im_shape'] = np.asarray([resize_h, resize_w], dtype=np.float32)
        if 'scale_factor' in sample:
            scale_factor = sample['scale_factor']
            sample['scale_factor'] = np.asarray(
                [scale_factor[0] * im_scale_y, scale_factor[1] * im_scale_x],
                dtype=np.float32)
        else:
            sample['scale_factor'] = np.asarray(
                [im_scale_y, im_scale_x], dtype=np.float32)

        # apply bbox
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'],
                                                [im_scale_x, im_scale_y],
                                                [resize_w, resize_h])
        return sample


class BatchRandomResize(BaseOperator):
    """
    Resize image to target size randomly. random target_size and interpolation method
    Args:
        target_size (int, list, tuple): image target size, if random size is True, must be list or tuple
        keep_ratio (bool): whether keep_raio or not, default true
        interp (int): the interpolation method
        random_size (bool): whether random select target size of image
        random_interp (bool): whether random select interpolation method
    """

    def __init__(self,
                 target_size,
                 keep_ratio,
                 interp=cv2.INTER_NEAREST,
                 random_size=True,
                 random_interp=False):
        super(BatchRandomResize, self).__init__()
        self.keep_ratio = keep_ratio
        self.interps = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_AREA,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
        ]
        self.interp = interp

        if random_size and not isinstance(target_size, list):
            raise TypeError(
                "Type of target_size is invalid when random_size is True. Must be List, now is {}".
                format(type(target_size)))
        self.target_size = target_size
        self.random_size = random_size
        self.random_interp = random_interp

    def __call__(self, samples, context=None):
        if self.random_size:
            index = np.random.choice(len(self.target_size))
            target_size = self.target_size[index]
        else:
            target_size = self.target_size

        if self.random_interp:
            interp = np.random.choice(self.interps)
        else:
            interp = self.interp

        resizer = Resize(target_size, keep_ratio=self.keep_ratio, interp=interp)
        return resizer(samples, context=context)


class NormalizeImage(BaseOperator):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[1, 1, 1],
                 is_scale=True):
        """
        Args:
            mean (list): the pixel mean
            std (list): the pixel variance
        """
        super(NormalizeImage, self).__init__()
        self.mean = mean
        self.std = std
        self.is_scale = is_scale
        if not (isinstance(self.mean, list) and isinstance(self.std, list) and
                isinstance(self.is_scale, bool)):
            raise TypeError("{}: input type is invalid.".format(self))
        from functools import reduce
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

    def apply(self, sample, context=None):
        """Normalize the image.
        Operators:
            1.(optional) Scale the image to [0,1]
            2. Each pixel minus mean and is divided by std
        """
        im = sample['img']
        im = im.astype(np.float32, copy=False)
        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
        std = np.array(self.std)[np.newaxis, np.newaxis, :]

        if self.is_scale:
            im = im / 255.0

        im -= mean
        im /= std

        sample['img'] = im
        return sample


class Permute(BaseOperator):
    def __init__(self):
        """
        Change the channel to be (C, H, W)
        """
        super(Permute, self).__init__()

    def apply(self, sample, context=None):
        im = sample['img']
        im = im.transpose((2, 0, 1))
        sample['img'] = im
        return sample


pipeline_list = [
            BatchRandomResize(target_size=[320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768],
                              random_size=True, random_interp=True, keep_ratio=False),
            NormalizeImage(mean=[0., 0., 0.], std=[1., 1., 1.], is_scale=True),
            Permute()
        ]


@COLLATE_FUNCTIONS.register_module()
def ppyoloe_collate_temp(batch: Sequence) -> dict:
    for pipe in pipeline_list:
        batch = pipe(batch)

    imgs = []
    labels_list = []
    for ind, i in enumerate(batch):
        num_boxes = len(i['gt_bbox'])
        img = i['img']
        img = np.ascontiguousarray(img)

        pad_gt_class = np.zeros((num_boxes, 1), dtype=np.float32)
        pad_gt_bbox = np.zeros((num_boxes, 4), dtype=np.float32)
        batch_idx = torch.from_numpy(np.ones((num_boxes, 1), dtype=np.float32) * ind)
        # bboxes_labels = torch.cat((batch_idx, pad_gt_class, pad_gt_bbox), dim=1)
        # num_gt = len(i['gt_bbox'])
        if num_boxes > 0:
            pad_gt_class[:num_boxes] = i['gt_class'][:, None]
            pad_gt_bbox[:num_boxes] = i['gt_bbox']
        pad_gt_class = torch.from_numpy(pad_gt_class)
        pad_gt_bbox = torch.from_numpy(pad_gt_bbox)
        # labels = np.concatenate((pad_gt_class, pad_gt_bbox), axis=1)
        imgs.append(torch.from_numpy(img.astype(np.float32)))
        bboxes_labels = torch.cat((batch_idx, pad_gt_class, pad_gt_bbox), dim=1)
        labels_list.append(bboxes_labels)
        # labels_list.append(labels)
        # labels_list.append(torch.from_numpy(labels))
    return {'inputs': torch.stack(imgs, 0), 'data_samples': torch.cat(labels_list, 0)}



# class PPYOLOE_collate_class():
#
#     def __init__(self):
#         self.pipeline_list = [
#             BatchRandomResize(target_size=[320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768],
#                               random_size=True, random_interp=True, keep_ratio=False),
#             NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_scale=True),
#             Permute()
#         ]
#
#     def __call__(self, batch):
#         for pipe in self.pipeline_list:
#             batch = pipe(batch)
#
#         num_max_boxes = max([len(s['gt_bbox']) for s in batch])
#         imgs = []
#         labels_list = []
#         for ind, i in enumerate(batch):
#             img = i['img']
#             img = np.ascontiguousarray(img)
#
#             pad_gt_class = np.zeros((num_max_boxes, 1), dtype=np.float32)
#             pad_gt_bbox = np.zeros((num_max_boxes, 4), dtype=np.float32)
#             num_gt = len(i['gt_bbox'])
#             if num_gt > 0:
#                 pad_gt_class[:num_gt] = i['gt_class'][:, None]
#                 pad_gt_bbox[:num_gt] = i['gt_bbox']
#             labels = np.concatenate((pad_gt_class, pad_gt_bbox), axis=1)
#             imgs.append(torch.from_numpy(img))
#             labels_list.append(torch.from_numpy(labels))
#         return {'inputs': torch.stack(imgs, 0), 'data_sample': torch.stack(labels_list, 0)}
