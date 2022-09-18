# Copyright (c) OpenMMLab. All rights reserved.
import unittest

from mmyolo.datasets import YOLOv5CocoDataset


class TestYOLOv5CocoDataset(unittest.TestCase):

    def test_batch_shapes_cfg(self):
        batch_shapes_cfg = dict(
            type='BatchShapePolicy',
            batch_size=2,
            img_size=640,
            size_divisor=32,
            extra_pad_ratio=0.5)

        # test serialize_data=True
        dataset = YOLOv5CocoDataset(
            data_prefix=dict(img='imgs'),
            ann_file='tests/data/coco_sample.json',
            filter_cfg=dict(filter_empty_gt=False, min_size=0),
            pipeline=[],
            serialize_data=True,
            batch_shapes_cfg=batch_shapes_cfg,
        )

        expected_img_ids = [3, 0, 2, 1]
        expected_batch_shapes = [[512, 672], [512, 672], [672, 672],
                                 [672, 672]]
        for i, data in enumerate(dataset):
            assert data['img_id'] == expected_img_ids[i]
            assert data['batch_shape'].tolist() == expected_batch_shapes[i]

        # test serialize_data=True
        dataset = YOLOv5CocoDataset(
            data_prefix=dict(img='imgs'),
            ann_file='tests/data/coco_sample.json',
            filter_cfg=dict(filter_empty_gt=False, min_size=0),
            pipeline=[],
            serialize_data=False,
            batch_shapes_cfg=batch_shapes_cfg,
        )

        expected_img_ids = [3, 0, 2, 1]
        expected_batch_shapes = [[512, 672], [512, 672], [672, 672],
                                 [672, 672]]
        for i, data in enumerate(dataset):
            assert data['img_id'] == expected_img_ids[i]
            assert data['batch_shape'].tolist() == expected_batch_shapes[i]

    def test_prepare_data(self):
        dataset = YOLOv5CocoDataset(
            data_prefix=dict(img='imgs'),
            ann_file='tests/data/coco_sample.json',
            filter_cfg=dict(filter_empty_gt=False, min_size=0),
            pipeline=[],
            serialize_data=True,
            batch_shapes_cfg=None,
        )
        for data in dataset:
            assert 'dataset' in data

        # test with test_mode = True
        dataset = YOLOv5CocoDataset(
            data_prefix=dict(img='imgs'),
            ann_file='tests/data/coco_sample.json',
            test_mode=True,
            pipeline=[])

        for data in dataset:
            assert 'dataset' not in data
