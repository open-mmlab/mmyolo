# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import unittest

import mmcv
import numpy as np
import torch
from mmdet.structures.bbox import HorizontalBoxes
from mmdet.structures.mask import BitmapMasks, PolygonMasks

from mmyolo.datasets.transforms import (LetterResize, LoadAnnotations,
                                        YOLOv5HSVRandomAug,
                                        YOLOv5KeepRatioResize,
                                        YOLOv5RandomAffine)
from mmyolo.datasets.transforms.transforms import (PPYOLOERandomCrop,
                                                   PPYOLOERandomDistort,
                                                   YOLOv5CopyPaste)


class TestLetterResize(unittest.TestCase):

    def setUp(self):
        """Set up the data info which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        rng = np.random.RandomState(0)
        self.data_info1 = dict(
            img=np.random.random((300, 400, 3)),
            gt_bboxes=np.array([[0, 0, 150, 150]], dtype=np.float32),
            batch_shape=np.array([192, 672], dtype=np.int64),
            gt_masks=PolygonMasks.random(1, height=300, width=400, rng=rng))
        self.data_info2 = dict(
            img=np.random.random((300, 400, 3)),
            gt_bboxes=np.array([[0, 0, 150, 150]], dtype=np.float32))
        self.data_info3 = dict(
            img=np.random.random((300, 400, 3)),
            batch_shape=np.array([192, 672], dtype=np.int64))
        self.data_info4 = dict(img=np.random.random((300, 400, 3)))

    def test_letter_resize(self):
        # Test allow_scale_up
        transform = LetterResize(scale=(640, 640), allow_scale_up=False)
        results = transform(copy.deepcopy(self.data_info1))
        self.assertEqual(results['img_shape'], (192, 672, 3))
        self.assertTrue(
            (results['gt_bboxes'] == np.array([[208., 0., 304., 96.]])).all())
        self.assertTrue((results['batch_shape'] == np.array([192, 672])).all())
        self.assertTrue((results['pad_param'] == np.array([0., 0., 208.,
                                                           208.])).all())
        self.assertTrue(
            (np.array(results['scale_factor'], dtype=np.float32) <= 1.).all())

        # Test pad_val
        transform = LetterResize(scale=(640, 640), pad_val=dict(img=144))
        results = transform(copy.deepcopy(self.data_info1))
        self.assertEqual(results['img_shape'], (192, 672, 3))
        self.assertTrue(
            (results['gt_bboxes'] == np.array([[208., 0., 304., 96.]])).all())
        self.assertTrue((results['batch_shape'] == np.array([192, 672])).all())
        self.assertTrue((results['pad_param'] == np.array([0., 0., 208.,
                                                           208.])).all())
        self.assertTrue(
            (np.array(results['scale_factor'], dtype=np.float32) <= 1.).all())

        # Test use_mini_pad
        transform = LetterResize(scale=(640, 640), use_mini_pad=True)
        results = transform(copy.deepcopy(self.data_info1))
        self.assertEqual(results['img_shape'], (192, 256, 3))
        self.assertTrue((results['gt_bboxes'] == np.array([[0., 0., 96.,
                                                            96.]])).all())
        self.assertTrue((results['batch_shape'] == np.array([192, 672])).all())
        self.assertTrue((results['pad_param'] == np.array([0., 0., 0.,
                                                           0.])).all())
        self.assertTrue(
            (np.array(results['scale_factor'], dtype=np.float32) <= 1.).all())

        # Test stretch_only
        transform = LetterResize(scale=(640, 640), stretch_only=True)
        results = transform(copy.deepcopy(self.data_info1))
        self.assertEqual(results['img_shape'], (192, 672, 3))
        self.assertTrue((results['gt_bboxes'] == np.array(
            [[0., 0., 251.99998474121094, 96.]])).all())
        self.assertTrue((results['batch_shape'] == np.array([192, 672])).all())
        self.assertTrue((results['pad_param'] == np.array([0., 0., 0.,
                                                           0.])).all())

        # Test
        transform = LetterResize(scale=(640, 640), pad_val=dict(img=144))
        for _ in range(5):
            input_h, input_w = np.random.randint(100, 700), np.random.randint(
                100, 700)
            output_h, output_w = np.random.randint(100,
                                                   700), np.random.randint(
                                                       100, 700)
            data_info = dict(
                img=np.random.random((input_h, input_w, 3)),
                gt_bboxes=np.array([[0, 0, 10, 10]], dtype=np.float32),
                batch_shape=np.array([output_h, output_w], dtype=np.int64),
                gt_masks=PolygonMasks(
                    [[np.array([0., 0., 0., 10., 10., 10., 10., 0.])]],
                    height=input_h,
                    width=input_w))
            results = transform(data_info)
            self.assertEqual(results['img_shape'], (output_h, output_w, 3))
            self.assertTrue(
                (results['batch_shape'] == np.array([output_h,
                                                     output_w])).all())

        # Test without batchshape
        transform = LetterResize(scale=(640, 640), pad_val=dict(img=144))
        for _ in range(5):
            input_h, input_w = np.random.randint(100, 700), np.random.randint(
                100, 700)
            data_info = dict(
                img=np.random.random((input_h, input_w, 3)),
                gt_bboxes=np.array([[0, 0, 10, 10]], dtype=np.float32),
                gt_masks=PolygonMasks(
                    [[np.array([0., 0., 0., 10., 10., 10., 10., 0.])]],
                    height=input_h,
                    width=input_w))
            results = transform(data_info)
            self.assertEqual(results['img_shape'], (640, 640, 3))

        # TODO: Testing the existence of multiple scale_factor and pad_param
        transform = [
            YOLOv5KeepRatioResize(scale=(32, 32)),
            LetterResize(scale=(64, 68), pad_val=dict(img=144))
        ]
        for _ in range(5):
            input_h, input_w = np.random.randint(100, 700), np.random.randint(
                100, 700)
            output_h, output_w = np.random.randint(100,
                                                   700), np.random.randint(
                                                       100, 700)
            data_info = dict(
                img=np.random.random((input_h, input_w, 3)),
                gt_bboxes=np.array([[0, 0, 5, 5]], dtype=np.float32),
                batch_shape=np.array([output_h, output_w], dtype=np.int64))
            for t in transform:
                data_info = t(data_info)
            # because of the "math.round" operation,
            # it is unable to strictly restore the original input shape
            # we just validate the correctness of scale_factor and pad_param
            self.assertIn('scale_factor', data_info)
            self.assertIn('pad_param', data_info)
            pad_param = data_info['pad_param'].reshape(-1, 2).sum(
                1)  # (top, b, l, r) -> (h, w)
            scale_factor = np.asarray(data_info['scale_factor'])  # (w, h)

            max_long_edge = max((32, 32))
            max_short_edge = min((32, 32))
            scale_factor_keepratio = min(
                max_long_edge / max(input_h, input_w),
                max_short_edge / min(input_h, input_w))
            validate_shape = np.asarray(
                (int(input_h * scale_factor_keepratio),
                 int(input_w * scale_factor_keepratio)))
            scale_factor_keepratio = np.asarray(
                (validate_shape[1] / input_w, validate_shape[0] / input_h))

            scale_factor_letter = ((np.asarray(
                (output_h, output_w)) - pad_param) / validate_shape)[::-1]
            self.assertTrue(data_info['img_shape'][:2] == (output_h, output_w))
            self.assertTrue((scale_factor == (scale_factor_keepratio *
                                              scale_factor_letter)).all())


class TestYOLOv5KeepRatioResize(unittest.TestCase):

    def setUp(self):
        """Set up the data info which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        rng = np.random.RandomState(0)
        self.data_info1 = dict(
            img=np.random.random((300, 400, 3)),
            gt_bboxes=np.array([[0, 0, 150, 150]], dtype=np.float32),
            gt_masks=PolygonMasks.random(
                num_masks=1, height=300, width=400, rng=rng))
        self.data_info2 = dict(img=np.random.random((300, 400, 3)))

    def test_yolov5_keep_ratio_resize(self):
        # test assertion for invalid keep_ratio
        with self.assertRaises(AssertionError):
            transform = YOLOv5KeepRatioResize(scale=(640, 640))
            transform.keep_ratio = False
            results = transform(copy.deepcopy(self.data_info1))

        # Test with gt_bboxes
        transform = YOLOv5KeepRatioResize(scale=(640, 640))
        results = transform(copy.deepcopy(self.data_info1))
        self.assertTrue(transform.keep_ratio, True)
        self.assertEqual(results['img_shape'], (480, 640))
        self.assertTrue(
            (results['gt_bboxes'] == np.array([[0., 0., 240., 240.]])).all())
        self.assertTrue((np.array(results['scale_factor'],
                                  dtype=np.float32) == 1.6).all())

        # Test only img
        transform = YOLOv5KeepRatioResize(scale=(640, 640))
        results = transform(copy.deepcopy(self.data_info2))
        self.assertEqual(results['img_shape'], (480, 640))
        self.assertTrue((np.array(results['scale_factor'],
                                  dtype=np.float32) == 1.6).all())


class TestYOLOv5HSVRandomAug(unittest.TestCase):

    def setUp(self):
        """Set up the data info which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.data_info = dict(
            img=mmcv.imread(
                osp.join(osp.dirname(__file__), '../../data/color.jpg'),
                'color'))

    def test_yolov5_hsv_random_aug(self):
        # Test with gt_bboxes
        transform = YOLOv5HSVRandomAug(
            hue_delta=0.015, saturation_delta=0.7, value_delta=0.4)
        results = transform(copy.deepcopy(self.data_info))
        self.assertTrue(
            results['img'].shape[:2] == self.data_info['img'].shape[:2])


class TestLoadAnnotations(unittest.TestCase):

    def setUp(self):
        """Set up the data info which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        data_prefix = osp.join(osp.dirname(__file__), '../../data')
        seg_map = osp.join(data_prefix, 'gray.jpg')
        self.results = {
            'ori_shape': (300, 400),
            'seg_map_path':
            seg_map,
            'instances': [{
                'bbox': [0, 0, 10, 20],
                'bbox_label': 1,
                'mask': [[0, 0, 0, 20, 10, 20, 10, 0]],
                'ignore_flag': 0
            }, {
                'bbox': [10, 10, 110, 120],
                'bbox_label': 2,
                'mask': [[10, 10, 110, 10, 110, 120, 110, 10]],
                'ignore_flag': 0
            }, {
                'bbox': [50, 50, 60, 80],
                'bbox_label': 2,
                'mask': [[50, 50, 60, 50, 60, 80, 50, 80]],
                'ignore_flag': 1
            }]
        }

    def test_load_bboxes(self):
        transform = LoadAnnotations(
            with_bbox=True,
            with_label=False,
            with_seg=False,
            with_mask=False,
            box_type=None)
        results = transform(copy.deepcopy(self.results))
        self.assertIn('gt_bboxes', results)
        self.assertTrue((results['gt_bboxes'] == np.array([[0, 0, 10, 20],
                                                           [10, 10, 110,
                                                            120]])).all())
        self.assertEqual(results['gt_bboxes'].dtype, np.float32)
        self.assertTrue(
            (results['gt_ignore_flags'] == np.array([False, False])).all())
        self.assertEqual(results['gt_ignore_flags'].dtype, bool)

        # test empty instance
        results = transform({})
        self.assertIn('gt_bboxes', results)
        self.assertTrue(results['gt_bboxes'].shape == (0, 4))
        self.assertIn('gt_ignore_flags', results)
        self.assertTrue(results['gt_ignore_flags'].shape == (0, ))

    def test_load_labels(self):
        transform = LoadAnnotations(
            with_bbox=False,
            with_label=True,
            with_seg=False,
            with_mask=False,
        )
        results = transform(copy.deepcopy(self.results))
        self.assertIn('gt_bboxes_labels', results)
        self.assertTrue((results['gt_bboxes_labels'] == np.array([1,
                                                                  2])).all())
        self.assertEqual(results['gt_bboxes_labels'].dtype, np.int64)

        # test empty instance
        results = transform({})
        self.assertIn('gt_bboxes_labels', results)
        self.assertTrue(results['gt_bboxes_labels'].shape == (0, ))


class TestYOLOv5RandomAffine(unittest.TestCase):

    def setUp(self):
        """Setup the data info which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.results = {
            'img':
            np.random.random((224, 224, 3)),
            'img_shape': (224, 224),
            'gt_bboxes_labels':
            np.array([1, 2, 3], dtype=np.int64),
            'gt_bboxes':
            np.array([[10, 10, 20, 20], [20, 20, 40, 40], [40, 40, 80, 80]],
                     dtype=np.float32),
            'gt_ignore_flags':
            np.array([0, 0, 1], dtype=bool),
        }

    def test_transform(self):
        # test assertion for invalid translate_ratio
        with self.assertRaises(AssertionError):
            transform = YOLOv5RandomAffine(max_translate_ratio=1.5)

        # test assertion for invalid scaling_ratio_range
        with self.assertRaises(AssertionError):
            transform = YOLOv5RandomAffine(scaling_ratio_range=(1.5, 0.5))

        with self.assertRaises(AssertionError):
            transform = YOLOv5RandomAffine(scaling_ratio_range=(0, 0.5))

        transform = YOLOv5RandomAffine()
        results = transform(copy.deepcopy(self.results))
        self.assertTrue(results['img'].shape[:2] == (224, 224))
        self.assertTrue(results['gt_bboxes_labels'].shape[0] ==
                        results['gt_bboxes'].shape[0])
        self.assertTrue(results['gt_bboxes_labels'].dtype == np.int64)
        self.assertTrue(results['gt_bboxes'].dtype == np.float32)
        self.assertTrue(results['gt_ignore_flags'].dtype == bool)

    def test_transform_with_boxlist(self):
        results = copy.deepcopy(self.results)
        results['gt_bboxes'] = HorizontalBoxes(results['gt_bboxes'])

        transform = YOLOv5RandomAffine()
        results = transform(copy.deepcopy(results))
        self.assertTrue(results['img'].shape[:2] == (224, 224))
        self.assertTrue(results['gt_bboxes_labels'].shape[0] ==
                        results['gt_bboxes'].shape[0])
        self.assertTrue(results['gt_bboxes_labels'].dtype == np.int64)
        self.assertTrue(results['gt_bboxes'].dtype == torch.float32)
        self.assertTrue(results['gt_ignore_flags'].dtype == bool)


class TestPPYOLOERandomCrop(unittest.TestCase):

    def setUp(self):
        """Setup the data info which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.results = {
            'img':
            np.random.random((224, 224, 3)),
            'img_shape': (224, 224),
            'gt_bboxes_labels':
            np.array([1, 2, 3], dtype=np.int64),
            'gt_bboxes':
            np.array([[10, 10, 20, 20], [20, 20, 40, 40], [40, 40, 80, 80]],
                     dtype=np.float32),
            'gt_ignore_flags':
            np.array([0, 0, 1], dtype=bool),
        }

    def test_transform(self):
        transform = PPYOLOERandomCrop()
        results = transform(copy.deepcopy(self.results))
        self.assertTrue(results['gt_bboxes_labels'].shape[0] ==
                        results['gt_bboxes'].shape[0])
        self.assertTrue(results['gt_bboxes_labels'].dtype == np.int64)
        self.assertTrue(results['gt_bboxes'].dtype == np.float32)
        self.assertTrue(results['gt_ignore_flags'].dtype == bool)

    def test_transform_with_boxlist(self):
        results = copy.deepcopy(self.results)
        results['gt_bboxes'] = HorizontalBoxes(results['gt_bboxes'])

        transform = PPYOLOERandomCrop()
        results = transform(copy.deepcopy(results))
        self.assertTrue(results['gt_bboxes_labels'].shape[0] ==
                        results['gt_bboxes'].shape[0])
        self.assertTrue(results['gt_bboxes_labels'].dtype == np.int64)
        self.assertTrue(results['gt_bboxes'].dtype == torch.float32)
        self.assertTrue(results['gt_ignore_flags'].dtype == bool)


class TestPPYOLOERandomDistort(unittest.TestCase):

    def setUp(self):
        """Setup the data info which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.results = {
            'img':
            np.random.random((224, 224, 3)),
            'img_shape': (224, 224),
            'gt_bboxes_labels':
            np.array([1, 2, 3], dtype=np.int64),
            'gt_bboxes':
            np.array([[10, 10, 20, 20], [20, 20, 40, 40], [40, 40, 80, 80]],
                     dtype=np.float32),
            'gt_ignore_flags':
            np.array([0, 0, 1], dtype=bool),
        }

    def test_transform(self):
        # test assertion for invalid prob
        with self.assertRaises(AssertionError):
            transform = PPYOLOERandomDistort(
                hue_cfg=dict(min=-18, max=18, prob=1.5))

        # test assertion for invalid num_distort_func
        with self.assertRaises(AssertionError):
            transform = PPYOLOERandomDistort(num_distort_func=5)

        transform = PPYOLOERandomDistort()
        results = transform(copy.deepcopy(self.results))
        self.assertTrue(results['img'].shape[:2] == (224, 224))
        self.assertTrue(results['gt_bboxes_labels'].shape[0] ==
                        results['gt_bboxes'].shape[0])
        self.assertTrue(results['gt_bboxes_labels'].dtype == np.int64)
        self.assertTrue(results['gt_bboxes'].dtype == np.float32)
        self.assertTrue(results['gt_ignore_flags'].dtype == bool)

    def test_transform_with_boxlist(self):
        results = copy.deepcopy(self.results)
        results['gt_bboxes'] = HorizontalBoxes(results['gt_bboxes'])

        transform = PPYOLOERandomDistort()
        results = transform(copy.deepcopy(results))
        self.assertTrue(results['img'].shape[:2] == (224, 224))
        self.assertTrue(results['gt_bboxes_labels'].shape[0] ==
                        results['gt_bboxes'].shape[0])
        self.assertTrue(results['gt_bboxes_labels'].dtype == np.int64)
        self.assertTrue(results['gt_bboxes'].dtype == torch.float32)
        self.assertTrue(results['gt_ignore_flags'].dtype == bool)


class TestYOLOv5CopyPaste(unittest.TestCase):

    def setUp(self):
        """Set up the data info which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.data_info = dict(
            img=np.random.random((300, 400, 3)),
            gt_bboxes=np.array([[0, 0, 10, 10]], dtype=np.float32),
            gt_masks=PolygonMasks(
                [[np.array([0., 0., 0., 10., 10., 10., 10., 0.])]],
                height=300,
                width=400))

    def test_transform(self):
        # test transform
        transform = YOLOv5CopyPaste(prob=1.0)
        results = transform(copy.deepcopy(self.data_info))
        self.assertTrue(len(results['gt_bboxes']) == 2)
        self.assertTrue(len(results['gt_masks']) == 2)

        rng = np.random.RandomState(0)
        # test with bitmap
        with self.assertRaises(AssertionError):
            results = transform(
                dict(
                    img=np.random.random((300, 400, 3)),
                    gt_bboxes=np.array([[0, 0, 10, 10]], dtype=np.float32),
                    gt_masks=BitmapMasks(
                        rng.rand(1, 300, 400), height=300, width=400)))
