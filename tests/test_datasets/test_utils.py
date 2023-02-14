# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import numpy as np
import torch
from mmdet.structures import DetDataSample
from mmdet.structures.bbox import HorizontalBoxes
from mmengine.structures import InstanceData

from mmyolo.datasets import BatchShapePolicy, yolov5_collate


def _rand_bboxes(rng, num_boxes, w, h):
    cx, cy, bw, bh = rng.rand(num_boxes, 4).T

    tl_x = ((cx * w) - (w * bw / 2)).clip(0, w)
    tl_y = ((cy * h) - (h * bh / 2)).clip(0, h)
    br_x = ((cx * w) + (w * bw / 2)).clip(0, w)
    br_y = ((cy * h) + (h * bh / 2)).clip(0, h)

    bboxes = np.vstack([tl_x, tl_y, br_x, br_y]).T
    return bboxes


class TestYOLOv5Collate(unittest.TestCase):

    def test_yolov5_collate(self):
        rng = np.random.RandomState(0)

        inputs = torch.randn((3, 10, 10))
        data_samples = DetDataSample()
        gt_instances = InstanceData()
        bboxes = _rand_bboxes(rng, 4, 6, 8)
        gt_instances.bboxes = HorizontalBoxes(bboxes, dtype=torch.float32)
        labels = rng.randint(1, 2, size=len(bboxes))
        gt_instances.labels = torch.LongTensor(labels)
        data_samples.gt_instances = gt_instances

        out = yolov5_collate([dict(inputs=inputs, data_samples=data_samples)])
        self.assertIsInstance(out, dict)
        self.assertTrue(out['inputs'].shape == (1, 3, 10, 10))
        self.assertTrue(out['data_samples'], dict)
        self.assertTrue(out['data_samples']['bboxes_labels'].shape == (4, 6))

        out = yolov5_collate([dict(inputs=inputs, data_samples=data_samples)] *
                             2)
        self.assertIsInstance(out, dict)
        self.assertTrue(out['inputs'].shape == (2, 3, 10, 10))
        self.assertTrue(out['data_samples'], dict)
        self.assertTrue(out['data_samples']['bboxes_labels'].shape == (8, 6))

    def test_yolov5_collate_with_multi_scale(self):
        rng = np.random.RandomState(0)

        inputs = torch.randn((3, 10, 10))
        data_samples = DetDataSample()
        gt_instances = InstanceData()
        bboxes = _rand_bboxes(rng, 4, 6, 8)
        gt_instances.bboxes = HorizontalBoxes(bboxes, dtype=torch.float32)
        labels = rng.randint(1, 2, size=len(bboxes))
        gt_instances.labels = torch.LongTensor(labels)
        data_samples.gt_instances = gt_instances

        out = yolov5_collate([dict(inputs=inputs, data_samples=data_samples)],
                             use_ms_training=True)
        self.assertIsInstance(out, dict)
        self.assertTrue(out['inputs'][0].shape == (3, 10, 10))
        self.assertTrue(out['data_samples'], dict)
        self.assertTrue(out['data_samples']['bboxes_labels'].shape == (4, 6))
        self.assertIsInstance(out['inputs'], list)
        self.assertIsInstance(out['data_samples']['bboxes_labels'],
                              torch.Tensor)

        out = yolov5_collate(
            [dict(inputs=inputs, data_samples=data_samples)] * 2,
            use_ms_training=True)
        self.assertIsInstance(out, dict)
        self.assertTrue(out['inputs'][0].shape == (3, 10, 10))
        self.assertTrue(out['data_samples'], dict)
        self.assertTrue(out['data_samples']['bboxes_labels'].shape == (8, 6))
        self.assertIsInstance(out['inputs'], list)
        self.assertIsInstance(out['data_samples']['bboxes_labels'],
                              torch.Tensor)


class TestBatchShapePolicy(unittest.TestCase):

    def test_batch_shape_policy(self):
        src_data_infos = [{
            'height': 20,
            'width': 100,
        }, {
            'height': 11,
            'width': 100,
        }, {
            'height': 21,
            'width': 100,
        }, {
            'height': 30,
            'width': 100,
        }, {
            'height': 10,
            'width': 100,
        }]

        expected_data_infos = [{
            'height': 10,
            'width': 100,
            'batch_shape': np.array([96, 672])
        }, {
            'height': 11,
            'width': 100,
            'batch_shape': np.array([96, 672])
        }, {
            'height': 20,
            'width': 100,
            'batch_shape': np.array([160, 672])
        }, {
            'height': 21,
            'width': 100,
            'batch_shape': np.array([160, 672])
        }, {
            'height': 30,
            'width': 100,
            'batch_shape': np.array([224, 672])
        }]

        batch_shapes_policy = BatchShapePolicy(batch_size=2)
        out_data_infos = batch_shapes_policy(src_data_infos)

        for i in range(5):
            self.assertEqual(
                (expected_data_infos[i]['height'],
                 expected_data_infos[i]['width']),
                (out_data_infos[i]['height'], out_data_infos[i]['width']))
            self.assertTrue(
                np.allclose(expected_data_infos[i]['batch_shape'],
                            out_data_infos[i]['batch_shape']))
