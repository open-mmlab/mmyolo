# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import unittest

import numpy as np
import torch
from mmdet.structures.bbox import HorizontalBoxes
from mmdet.structures.mask import BitmapMasks

from mmyolo.datasets import YOLOv5CocoDataset
from mmyolo.datasets.transforms import Mosaic, YOLOv5MixUp, YOLOXMixUp


class TestMosaic(unittest.TestCase):

    def setUp(self):
        """Setup the data info which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        rng = np.random.RandomState(0)
        self.pre_transform = [
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='LoadAnnotations', with_bbox=True)
        ]

        self.dataset = YOLOv5CocoDataset(
            data_prefix=dict(
                img=osp.join(osp.dirname(__file__), '../../data')),
            ann_file=osp.join(
                osp.dirname(__file__), '../../data/coco_sample_color.json'),
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            pipeline=[])
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
            'gt_masks':
            BitmapMasks(rng.rand(3, 224, 224), height=224, width=224),
            'dataset':
            self.dataset
        }

    def test_transform(self):
        # test assertion for invalid img_scale
        with self.assertRaises(AssertionError):
            transform = Mosaic(img_scale=640)

        # test assertion for invalid probability
        with self.assertRaises(AssertionError):
            transform = Mosaic(prob=1.5)

        transform = Mosaic(
            img_scale=(10, 12), pre_transform=self.pre_transform)
        results = transform(copy.deepcopy(self.results))
        self.assertTrue(results['img'].shape[:2] == (20, 24))
        self.assertTrue(results['gt_bboxes_labels'].shape[0] ==
                        results['gt_bboxes'].shape[0])
        self.assertTrue(results['gt_bboxes_labels'].dtype == np.int64)
        self.assertTrue(results['gt_bboxes'].dtype == np.float32)
        self.assertTrue(results['gt_ignore_flags'].dtype == bool)

    def test_transform_with_no_gt(self):
        self.results['gt_bboxes'] = np.empty((0, 4), dtype=np.float32)
        self.results['gt_bboxes_labels'] = np.empty((0, ), dtype=np.int64)
        self.results['gt_ignore_flags'] = np.empty((0, ), dtype=bool)
        transform = Mosaic(
            img_scale=(10, 12), pre_transform=self.pre_transform)
        results = transform(copy.deepcopy(self.results))
        self.assertIsInstance(results, dict)
        self.assertTrue(results['img'].shape[:2] == (20, 24))
        self.assertTrue(
            results['gt_bboxes_labels'].shape[0] == results['gt_bboxes'].
            shape[0] == results['gt_ignore_flags'].shape[0])
        self.assertTrue(results['gt_bboxes_labels'].dtype == np.int64)
        self.assertTrue(results['gt_bboxes'].dtype == np.float32)
        self.assertTrue(results['gt_ignore_flags'].dtype == bool)

    def test_transform_with_box_list(self):
        transform = Mosaic(
            img_scale=(10, 12), pre_transform=self.pre_transform)
        results = copy.deepcopy(self.results)
        results['gt_bboxes'] = HorizontalBoxes(results['gt_bboxes'])
        results = transform(results)
        self.assertTrue(results['img'].shape[:2] == (20, 24))
        self.assertTrue(results['gt_bboxes_labels'].shape[0] ==
                        results['gt_bboxes'].shape[0])
        self.assertTrue(results['gt_bboxes_labels'].dtype == np.int64)
        self.assertTrue(results['gt_bboxes'].dtype == torch.float32)
        self.assertTrue(results['gt_ignore_flags'].dtype == bool)


class TestYOLOv5MixUp(unittest.TestCase):

    def setUp(self):
        """Setup the data info which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        rng = np.random.RandomState(0)
        self.pre_transform = [
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='LoadAnnotations', with_bbox=True)
        ]
        self.dataset = YOLOv5CocoDataset(
            data_prefix=dict(
                img=osp.join(osp.dirname(__file__), '../../data')),
            ann_file=osp.join(
                osp.dirname(__file__), '../../data/coco_sample_color.json'),
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            pipeline=[])

        self.results = {
            'img':
            np.random.random((288, 512, 3)),
            'img_shape': (288, 512),
            'gt_bboxes_labels':
            np.array([1, 2, 3], dtype=np.int64),
            'gt_bboxes':
            np.array([[10, 10, 20, 20], [20, 20, 40, 40], [40, 40, 80, 80]],
                     dtype=np.float32),
            'gt_ignore_flags':
            np.array([0, 0, 1], dtype=bool),
            'gt_masks':
            BitmapMasks(rng.rand(3, 288, 512), height=288, width=512),
            'dataset':
            self.dataset
        }

    def test_transform(self):
        transform = YOLOv5MixUp(pre_transform=self.pre_transform)
        results = transform(copy.deepcopy(self.results))
        self.assertTrue(results['img'].shape[:2] == (288, 512))
        self.assertTrue(results['gt_bboxes_labels'].shape[0] ==
                        results['gt_bboxes'].shape[0])
        self.assertTrue(results['gt_bboxes_labels'].dtype == np.int64)
        self.assertTrue(results['gt_bboxes'].dtype == np.float32)
        self.assertTrue(results['gt_ignore_flags'].dtype == bool)

    def test_transform_with_box_list(self):
        results = copy.deepcopy(self.results)
        results['gt_bboxes'] = HorizontalBoxes(results['gt_bboxes'])

        transform = YOLOv5MixUp(pre_transform=self.pre_transform)
        results = transform(results)
        self.assertTrue(results['img'].shape[:2] == (288, 512))
        self.assertTrue(results['gt_bboxes_labels'].shape[0] ==
                        results['gt_bboxes'].shape[0])
        self.assertTrue(results['gt_bboxes_labels'].dtype == np.int64)
        self.assertTrue(results['gt_bboxes'].dtype == torch.float32)
        self.assertTrue(results['gt_ignore_flags'].dtype == bool)


class TestYOLOXMixUp(unittest.TestCase):

    def setUp(self):
        """Setup the data info which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        rng = np.random.RandomState(0)
        self.pre_transform = [
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='LoadAnnotations', with_bbox=True)
        ]
        self.dataset = YOLOv5CocoDataset(
            data_prefix=dict(
                img=osp.join(osp.dirname(__file__), '../../data')),
            ann_file=osp.join(
                osp.dirname(__file__), '../../data/coco_sample_color.json'),
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            pipeline=[])
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
            'gt_masks':
            BitmapMasks(rng.rand(3, 224, 224), height=224, width=224),
            'dataset':
            self.dataset
        }

    def test_transform(self):
        # test assertion for invalid img_scale
        with self.assertRaises(AssertionError):
            transform = YOLOXMixUp(img_scale=640)

        transform = YOLOXMixUp(
            img_scale=(10, 12),
            ratio_range=(0.8, 1.6),
            pad_val=114.0,
            pre_transform=self.pre_transform)

        # self.results['mix_results'] = [copy.deepcopy(self.results)]
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

        transform = YOLOXMixUp(
            img_scale=(10, 12),
            ratio_range=(0.8, 1.6),
            pad_val=114.0,
            pre_transform=self.pre_transform)
        results['mix_results'] = [results]
        results = transform(results)
        self.assertTrue(results['img'].shape[:2] == (224, 224))
        self.assertTrue(results['gt_bboxes_labels'].shape[0] ==
                        results['gt_bboxes'].shape[0])
        self.assertTrue(results['gt_bboxes_labels'].dtype == np.int64)
        self.assertTrue(results['gt_bboxes'].dtype == torch.float32)
        self.assertTrue(results['gt_ignore_flags'].dtype == bool)
