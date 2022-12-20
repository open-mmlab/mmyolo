# Copyright (c) OpenMMLab. All rights reserved.
import unittest

from mmengine.dataset import ConcatDataset

from mmyolo.datasets import YOLOv5VOCDataset
from mmyolo.utils import register_all_modules

register_all_modules()


class TestYOLOv5VocDataset(unittest.TestCase):

    def test_batch_shapes_cfg(self):
        batch_shapes_cfg = dict(
            type='BatchShapePolicy',
            batch_size=2,
            img_size=640,
            size_divisor=32,
            extra_pad_ratio=0.5)

        # test serialize_data=True
        dataset = YOLOv5VOCDataset(
            data_root='tests/data/VOCdevkit/',
            ann_file='VOC2007/ImageSets/Main/trainval.txt',
            data_prefix=dict(sub_data_root='VOC2007/'),
            test_mode=True,
            pipeline=[],
            batch_shapes_cfg=batch_shapes_cfg,
        )

        expected_img_ids = ['000001']
        expected_batch_shapes = [[672, 480]]
        for i, data in enumerate(dataset):
            assert data['img_id'] == expected_img_ids[i]
            assert data['batch_shape'].tolist() == expected_batch_shapes[i]

    def test_prepare_data(self):
        dataset = YOLOv5VOCDataset(
            data_root='tests/data/VOCdevkit/',
            ann_file='VOC2007/ImageSets/Main/trainval.txt',
            data_prefix=dict(sub_data_root='VOC2007/'),
            filter_cfg=dict(filter_empty_gt=False, min_size=0),
            pipeline=[],
            serialize_data=True,
            batch_shapes_cfg=None,
        )
        for data in dataset:
            assert 'dataset' in data

        # test with test_mode = True
        dataset = YOLOv5VOCDataset(
            data_root='tests/data/VOCdevkit/',
            ann_file='VOC2007/ImageSets/Main/trainval.txt',
            data_prefix=dict(sub_data_root='VOC2007/'),
            filter_cfg=dict(
                filter_empty_gt=True, min_size=32, bbox_min_size=None),
            pipeline=[],
            test_mode=True,
            batch_shapes_cfg=None)

        for data in dataset:
            assert 'dataset' not in data

    def test_concat_dataset(self):
        dataset = ConcatDataset(
            datasets=[
                dict(
                    type='YOLOv5VOCDataset',
                    data_root='tests/data/VOCdevkit/',
                    ann_file='VOC2007/ImageSets/Main/trainval.txt',
                    data_prefix=dict(sub_data_root='VOC2007/'),
                    filter_cfg=dict(filter_empty_gt=False, min_size=32),
                    pipeline=[]),
                dict(
                    type='YOLOv5VOCDataset',
                    data_root='tests/data/VOCdevkit/',
                    ann_file='VOC2012/ImageSets/Main/trainval.txt',
                    data_prefix=dict(sub_data_root='VOC2012/'),
                    filter_cfg=dict(filter_empty_gt=False, min_size=32),
                    pipeline=[])
            ],
            ignore_keys='dataset_type')

        dataset.full_init()
        self.assertEqual(len(dataset), 2)
