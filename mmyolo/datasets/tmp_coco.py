import copy
import uuid
from typing import List, Sequence

import cv2
import mmcv
import mmengine
import numpy as np
from mmdet.datasets import CocoDataset
from mmengine.dataset import Compose

from mmyolo.registry import DATASETS


class BaseOperator(object):
    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__
        self._id = name + '_' + str(uuid.uuid4())[-6:]

    def transform(self, sample, context=None):
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
                sample[i] = self.transform(sample[i], context)
        else:
            sample = self.transform(sample, context)

        return sample

    def __str__(self):
        return str(self._id)


class Decode(BaseOperator):
    def __init__(self):
        """ Transform the image data to numpy format following the rgb format
        """
        super(Decode, self).__init__()

    def transform(self, sample, context=None):
        """ load image if 'im_file' field is not empty but 'image' is"""
        im = sample['img']
        # print('img0', type(im))
        # data = np.frombuffer(im, dtype='uint8')
        # im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode
        # if 'keep_ori_im' in sample and sample['keep_ori_im']:
        #     sample['ori_image'] = im
        # print('img1', type(im))

        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        sample['img'] = im
        if 'h' not in sample:
            sample['h'] = im.shape[0]
        elif sample['h'] != im.shape[0]:
            # logger.warning(
            #     "The actual image height: {} is not equal to the "
            #     "height: {} in annotation, and update sample['h'] by actual "
            #     "image height.".format(im.shape[0], sample['h']))
            sample['h'] = im.shape[0]
        if 'w' not in sample:
            sample['w'] = im.shape[1]
        elif sample['w'] != im.shape[1]:
            # logger.warning(
            #     "The actual image width: {} is not equal to the "
            #     "width: {} in annotation, and update sample['w'] by actual "
            #     "image width.".format(im.shape[1], sample['w']))
            sample['w'] = im.shape[1]

        sample['im_shape'] = np.array(im.shape[:2], dtype=np.float32)
        sample['scale_factor'] = np.array([1., 1.], dtype=np.float32)
        return sample


class Permute(BaseOperator):
    def __init__(self):
        """
        Change the channel to be (C, H, W)
        """
        super(Permute, self).__init__()

    def transform(self, sample, context=None):
        im = sample['img']
        im = im.transpose((2, 0, 1))
        sample['img'] = im
        return sample


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

    def transform(self, sample, context=None):
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


class RandomDistort(BaseOperator):
    """Random color distortion.
    Args:
        hue (list): hue settings. in [lower, upper, probability] format.
        saturation (list): saturation settings. in [lower, upper, probability] format.
        contrast (list): contrast settings. in [lower, upper, probability] format.
        brightness (list): brightness settings. in [lower, upper, probability] format.
        random_apply (bool): whether to apply in random (yolo) or fixed (SSD)
            order.
        count (int): the number of doing distrot
        random_channel (bool): whether to swap channels randomly
    """

    def __init__(self,
                 hue=[-18, 18, 0.5],
                 saturation=[0.5, 1.5, 0.5],
                 contrast=[0.5, 1.5, 0.5],
                 brightness=[0.5, 1.5, 0.5],
                 random_apply=True,
                 count=4,
                 random_channel=False):
        super(RandomDistort, self).__init__()
        self.hue = hue
        self.saturation = saturation
        self.contrast = contrast
        self.brightness = brightness
        self.random_apply = random_apply
        self.count = count
        self.random_channel = random_channel

    def apply_hue(self, img):
        low, high, prob = self.hue
        if np.random.uniform(0., 1.) < prob:
            return img

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
        return img

    def apply_saturation(self, img):
        low, high, prob = self.saturation
        if np.random.uniform(0., 1.) < prob:
            return img
        delta = np.random.uniform(low, high)
        img = img.astype(np.float32)
        # it works, but result differ from HSV version
        gray = img * np.array([[[0.299, 0.587, 0.114]]], dtype=np.float32)
        gray = gray.sum(axis=2, keepdims=True)
        gray *= (1.0 - delta)
        img *= delta
        img += gray
        return img

    def apply_contrast(self, img):
        low, high, prob = self.contrast
        if np.random.uniform(0., 1.) < prob:
            return img
        delta = np.random.uniform(low, high)
        img = img.astype(np.float32)
        img *= delta
        return img

    def apply_brightness(self, img):
        low, high, prob = self.brightness
        if np.random.uniform(0., 1.) < prob:
            return img
        delta = np.random.uniform(low, high)
        img = img.astype(np.float32)
        img += delta
        return img

    def transform(self, sample, context=None):
        img = sample['img']
        if self.random_apply:
            functions = [
                self.apply_brightness, self.apply_contrast,
                self.apply_saturation, self.apply_hue
            ]
            distortions = np.random.permutation(functions)[:self.count]
            for func in distortions:
                img = func(img)
            sample['img'] = img
            return sample

        img = self.apply_brightness(img)
        mode = np.random.randint(0, 2)

        if mode:
            img = self.apply_contrast(img)

        img = self.apply_saturation(img)
        img = self.apply_hue(img)

        if not mode:
            img = self.apply_contrast(img)

        if self.random_channel:
            if np.random.randint(0, 2):
                img = img[..., np.random.permutation(3)]
        sample['img'] = img
        return sample


class RandomFlip(BaseOperator):
    def __init__(self, prob=0.5):
        """
        Args:
            prob (float): the probability of flipping image
        """
        super(RandomFlip, self).__init__()
        self.prob = prob
        if not (isinstance(self.prob, float)):
            raise TypeError("{}: input type is invalid.".format(self))


    def apply_image(self, image):
        return image[:, ::-1, :]

    def apply_bbox(self, bbox, width):
        oldx1 = bbox[:, 0].copy()
        oldx2 = bbox[:, 2].copy()
        bbox[:, 0] = width - oldx2
        bbox[:, 2] = width - oldx1
        return bbox

    def transform(self, sample, context=None):
        """Filp the image and bounding box.
        Operators:
            1. Flip the image numpy.
            2. Transform the bboxes' x coordinates.
              (Must judge whether the coordinates are normalized!)
            3. Transform the segmentations' x coordinates.
              (Must judge whether the coordinates are normalized!)
        Output:
            sample: the image, bounding box and segmentation part
                    in sample are flipped.
        """
        if np.random.uniform(0, 1) < self.prob:
            im = sample['img']
            height, width = im.shape[:2]
            im = self.apply_image(im)
            if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
                sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'], width)

            sample['flipped'] = True
            sample['img'] = im
        return sample



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
        if not isinstance(target_size, (Integral, Sequence)):
            raise TypeError(
                "Type of target_size is invalid. Must be Integer or List or Tuple, now is {}".
                format(type(target_size)))
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

    def transform(self, sample, context=None):
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


class RandomExpand(BaseOperator):
    """Random expand the canvas.
    Args:
        ratio (float): maximum expansion ratio.
        prob (float): probability to expand.
        fill_value (list): color value used to fill the canvas. in RGB order.
    """

    def __init__(self, ratio=4., prob=0.5, fill_value=(127.5, 127.5, 127.5)):
        super(RandomExpand, self).__init__()
        assert ratio > 1.01, "expand ratio must be larger than 1.01"
        self.ratio = ratio
        self.prob = prob
        assert isinstance(fill_value, (Number, Sequence)), \
            "fill value must be either float or sequence"
        if isinstance(fill_value, Number):
            fill_value = (fill_value, ) * 3
        if not isinstance(fill_value, tuple):
            fill_value = tuple(fill_value)
        self.fill_value = fill_value

    def transform(self, sample, context=None):
        if np.random.uniform(0., 1.) < self.prob:
            return sample

        im = sample['img']
        height, width = im.shape[:2]
        ratio = np.random.uniform(1., self.ratio)
        h = int(height * ratio)
        w = int(width * ratio)
        if not h > height or not w > width:
            return sample
        y = np.random.randint(0, h - height)
        x = np.random.randint(0, w - width)
        offsets, size = [x, y], [h, w]

        pad = Pad(size,
                  pad_mode=-1,
                  offsets=offsets,
                  fill_value=self.fill_value)

        return pad(sample, context=context)


class RandomCrop(BaseOperator):
    """Random crop image and bboxes.
    Args:
        aspect_ratio (list): aspect ratio of cropped region.
            in [min, max] format.
        thresholds (list): iou thresholds for decide a valid bbox crop.
        scaling (list): ratio between a cropped region and the original image.
             in [min, max] format.
        num_attempts (int): number of tries before giving up.
        allow_no_crop (bool): allow return without actually cropping them.
        cover_all_box (bool): ensure all bboxes are covered in the final crop.
        is_mask_crop(bool): whether crop the segmentation.
    """

    def __init__(self,
                 aspect_ratio=[.5, 2.],
                 thresholds=[.0, .1, .3, .5, .7, .9],
                 scaling=[.3, 1.],
                 num_attempts=50,
                 allow_no_crop=True,
                 cover_all_box=False,
                 is_mask_crop=False):
        super(RandomCrop, self).__init__()
        self.aspect_ratio = aspect_ratio
        self.thresholds = thresholds
        self.scaling = scaling
        self.num_attempts = num_attempts
        self.allow_no_crop = allow_no_crop
        self.cover_all_box = cover_all_box
        self.is_mask_crop = is_mask_crop

    def transform(self, sample, transform=None):
        if 'gt_bbox' in sample and len(sample['gt_bbox']) == 0:
            return sample

        h, w = sample['img'].shape[:2]
        gt_bbox = sample['gt_bbox']

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
                return sample

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
                iou = self._iou_matrix(
                    gt_bbox, np.array(
                        [crop_box], dtype=np.float32))
                if iou.max() < thresh:
                    continue

                if self.cover_all_box and iou.min() < thresh:
                    continue

                cropped_box, valid_ids = self._crop_box_with_center_constraint(
                    gt_bbox, np.array(
                        crop_box, dtype=np.float32))
                if valid_ids.size > 0:
                    found = True
                    break

            if found:
                sample['img'] = self._crop_image(sample['img'], crop_box)
                sample['gt_bbox'] = np.take(cropped_box, valid_ids, axis=0)
                sample['gt_class'] = np.take(
                    sample['gt_class'], valid_ids, axis=0)
                if 'gt_score' in sample:
                    sample['gt_score'] = np.take(
                        sample['gt_score'], valid_ids, axis=0)

                if 'is_crowd' in sample:
                    sample['is_crowd'] = np.take(
                        sample['is_crowd'], valid_ids, axis=0)

                if 'difficult' in sample:
                    sample['difficult'] = np.take(
                        sample['difficult'], valid_ids, axis=0)

                return sample

        return sample

    def _iou_matrix(self, a, b):
        tl_i = np.maximum(a[:, np.newaxis, :2], b[:, :2])
        br_i = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

        area_i = np.prod(br_i - tl_i, axis=2) * (tl_i < br_i).all(axis=2)
        area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
        area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
        area_o = (area_a[:, np.newaxis] + area_b - area_i)
        return area_i / (area_o + 1e-10)

    def _crop_box_with_center_constraint(self, box, crop):
        cropped_box = box.copy()

        cropped_box[:, :2] = np.maximum(box[:, :2], crop[:2])
        cropped_box[:, 2:] = np.minimum(box[:, 2:], crop[2:])
        cropped_box[:, :2] -= crop[:2]
        cropped_box[:, 2:] -= crop[:2]

        centers = (box[:, :2] + box[:, 2:]) / 2
        valid = np.logical_and(crop[:2] <= centers,
                               centers < crop[2:]).all(axis=1)
        valid = np.logical_and(
            valid, (cropped_box[:, :2] < cropped_box[:, 2:]).all(axis=1))

        return cropped_box, np.where(valid)[0]

    def _crop_image(self, img, crop):
        x1, y1, x2, y2 = crop
        return img[y1:y2, x1:x2, :]

    def _crop_segm(self, segm, crop):
        x1, y1, x2, y2 = crop
        return segm[:, y1:y2, x1:x2]


class Pad(BaseOperator):
    def __init__(self,
                 size=None,
                 size_divisor=32,
                 pad_mode=0,
                 offsets=None,
                 fill_value=(127.5, 127.5, 127.5)):
        """
        Pad image to a specified size or multiple of size_divisor.
        Args:
            size (int, Sequence): image target size, if None, pad to multiple of size_divisor, default None
            size_divisor (int): size divisor, default 32
            pad_mode (int): pad mode, currently only supports four modes [-1, 0, 1, 2]. if -1, use specified offsets
                if 0, only pad to right and bottom. if 1, pad according to center. if 2, only pad left and top
            offsets (list): [offset_x, offset_y], specify offset while padding, only supported pad_mode=-1
            fill_value (bool): rgb value of pad area, default (127.5, 127.5, 127.5)
        """
        super(Pad, self).__init__()

        if not isinstance(size, (int, Sequence)):
            raise TypeError(
                "Type of target_size is invalid when random_size is True. \
                            Must be List, now is {}".format(type(size)))

        if isinstance(size, int):
            size = [size, size]

        assert pad_mode in [
            -1, 0, 1, 2
        ], 'currently only supports four modes [-1, 0, 1, 2]'
        if pad_mode == -1:
            assert offsets, 'if pad_mode is -1, offsets should not be None'

        self.size = size
        self.size_divisor = size_divisor
        self.pad_mode = pad_mode
        self.fill_value = fill_value
        self.offsets = offsets

    def apply_bbox(self, bbox, offsets):
        return bbox + np.array(offsets * 2, dtype=np.float32)

    def apply_image(self, image, offsets, im_size, size):
        x, y = offsets
        im_h, im_w = im_size
        h, w = size
        canvas = np.ones((h, w, 3), dtype=np.float32)
        canvas *= np.array(self.fill_value, dtype=np.float32)
        canvas[y:y + im_h, x:x + im_w, :] = image.astype(np.float32)
        return canvas

    def transform(self, sample, context=None):
        im = sample['img']
        im_h, im_w = im.shape[:2]
        if self.size:
            h, w = self.size
            assert (
                im_h <= h and im_w <= w
            ), '(h, w) of target size should be greater than (im_h, im_w)'
        else:
            h = int(np.ceil(im_h / self.size_divisor) * self.size_divisor)
            w = int(np.ceil(im_w / self.size_divisor) * self.size_divisor)

        if h == im_h and w == im_w:
            sample['img'] = im.astype(np.float32)
            return sample

        if self.pad_mode == -1:
            offset_x, offset_y = self.offsets
        elif self.pad_mode == 0:
            offset_y, offset_x = 0, 0
        elif self.pad_mode == 1:
            offset_y, offset_x = (h - im_h) // 2, (w - im_w) // 2
        else:
            offset_y, offset_x = h - im_h, w - im_w

        offsets, im_size, size = [offset_x, offset_y], [im_h, im_w], [h, w]

        sample['img'] = self.apply_image(im, offsets, im_size, size)

        if self.pad_mode == 0:
            return sample
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'], offsets)

        return sample


@DATASETS.register_module()
class TmpCocoDataset(CocoDataset):

    def __init__(self,
                 *args,
                 img_size=640,
                 batch_size=1,
                 stide=32,
                 file_client_args=None,
                 **kwargs
                 ):
        super(CocoDataset, self).__init__(*args, **kwargs)
        self.pipeline_train = Compose([
            Decode(),
            RandomDistort(),
            RandomExpand(fill_value=[123.675, 116.28, 103.53]),
            RandomCrop(),
            RandomFlip()
        ])

        self.file_client = None
        if file_client_args is not None:
            self.file_client = mmengine.FileClient(**file_client_args)

    def prepare_img(self, idx):
        results = copy.deepcopy(self.data_list[idx])
        return self.pipeline(results)

    def _load_image(self, index):
        # loads 1 image from dataset, returns img, original hw, resized hw
        path = self.data_list[index]['img_path']

        if self.file_client:
            img_bytes = self.file_client.get(path)
            img = mmcv.imfrombytes(img_bytes)
        else:
            img = cv2.imread(path)  # BGR

        if img is None:
            print('Image Not Found ' + path)
            return None, (0, 0)

        h0, w0 = img.shape[:2]  # orig hw
        # r = self.img_size / max(h0, w0)  # ratio
        # if r != 1:  # if sizes are not equal
        #     img = cv2.resize(
        #         img, (int(w0 * r), int(h0 * r)),
        #         interpolation=cv2.INTER_LINEAR)
        return img, (h0, w0)     #, img.shape[:2]

    def _train(self, idx):
        img, (orig_h, orig_w) = self._load_image(idx)   # BGR
        gt_bboxes = []   # x1y1x2y2
        gt_bboxes_labels = []
        for instance in self.data_list[idx]['instances']:
            if instance['ignore_flag'] == 0:
                gt_bboxes.append(instance['bbox'])
                gt_bboxes_labels.append(instance['bbox_label'])
        gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        gt_bboxes_labels = np.array(gt_bboxes_labels)
        bboxes = gt_bboxes  # xyxy
        labels = gt_bboxes_labels
        # labels = np.concatenate((labels[:, None], bboxes), axis=1)

        results = {
            'img': img,
            'gt_bbox': bboxes,
            'gt_class': labels
        }
        results = self.pipeline_train(results)
        # labels = np.concatenate((results['gt_class'][:, None], results['gt_bbox']), axis=1)

        # return results['img'], labels
        return results



    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_img(idx)
        else:
            results = self._train(idx)

            # img, labels = self._train(idx)
            #
            # nl = len(labels)  # 归一化 class cx cy w h
            # labels_out = torch.zeros((nl, 6))  # 归一化 batch_idx class cx cy w h
            # if nl:
            #     labels_out[:, 1:] = torch.from_numpy(labels)
            #
            # # Convert
            # img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            # img = np.ascontiguousarray(img)
            # # print(labels_out.shape)

            # return torch.from_numpy(img), labels_out
            return results


    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
            if filter_empty_gt:
                min_size = self.filter_cfg.get('min_size', 0)

                # obtain images that contain annotation
                ids_with_ann = set(data_info['img_id'] for data_info in self.data_list)
                # obtain images that contain annotations of the required categories
                ids_in_cat = set()
                for i, class_id in enumerate(self.cat_ids):
                    ids_in_cat |= set(self.cat_img_map[class_id])
                # merge the image id sets of the two conditions and use the merged set
                # to filter out images if self.filter_empty_gt=True
                ids_in_cat &= ids_with_ann

                valid_data_infos = []
                for i, data_info in enumerate(self.data_list):
                    img_id = data_info['img_id']
                    width = data_info['width']
                    height = data_info['height']
                    if filter_empty_gt and img_id not in ids_in_cat:
                        continue
                    if min(width, height) >= min_size:
                        valid_data_infos.append(data_info)
            else:
                return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        # obtain images that contain annotation
        ids_with_ann = set(data_info['img_id'] for data_info in self.data_list)
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            img_id = data_info['img_id']
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos