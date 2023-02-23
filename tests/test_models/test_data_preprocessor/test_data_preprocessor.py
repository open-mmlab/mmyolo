# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmdet.structures import DetDataSample
from mmengine import MessageHub

from mmyolo.models import PPYOLOEBatchRandomResize, PPYOLOEDetDataPreprocessor
from mmyolo.models.data_preprocessors import (YOLOv5DetDataPreprocessor,
                                              YOLOXBatchSyncRandomResize)
from mmyolo.utils import register_all_modules

register_all_modules()


class TestYOLOv5DetDataPreprocessor(TestCase):

    def test_forward(self):
        processor = YOLOv5DetDataPreprocessor(mean=[0, 0, 0], std=[1, 1, 1])

        data = {
            'inputs': [torch.randint(0, 256, (3, 11, 10))],
            'data_samples': [DetDataSample()]
        }
        out_data = processor(data, training=False)
        batch_inputs, batch_data_samples = out_data['inputs'], out_data[
            'data_samples']

        self.assertEqual(batch_inputs.shape, (1, 3, 11, 10))
        self.assertEqual(len(batch_data_samples), 1)

        # test channel_conversion
        processor = YOLOv5DetDataPreprocessor(
            mean=[0., 0., 0.], std=[1., 1., 1.], bgr_to_rgb=True)
        out_data = processor(data, training=False)
        batch_inputs, batch_data_samples = out_data['inputs'], out_data[
            'data_samples']
        self.assertEqual(batch_inputs.shape, (1, 3, 11, 10))
        self.assertEqual(len(batch_data_samples), 1)

        # test padding, training=False
        data = {
            'inputs': [
                torch.randint(0, 256, (3, 10, 11)),
                torch.randint(0, 256, (3, 9, 14))
            ]
        }
        processor = YOLOv5DetDataPreprocessor(
            mean=[0., 0., 0.], std=[1., 1., 1.], bgr_to_rgb=True)
        out_data = processor(data, training=False)
        batch_inputs, batch_data_samples = out_data['inputs'], out_data[
            'data_samples']
        self.assertEqual(batch_inputs.shape, (2, 3, 10, 14))
        self.assertIsNone(batch_data_samples)

        # test training
        data = {
            'inputs': torch.randint(0, 256, (2, 3, 10, 11)),
            'data_samples': {
                'bboxes_labels': torch.randint(0, 11, (18, 6))
            },
        }
        out_data = processor(data, training=True)
        batch_inputs, batch_data_samples = out_data['inputs'], out_data[
            'data_samples']
        self.assertIn('img_metas', batch_data_samples)
        self.assertIn('bboxes_labels', batch_data_samples)
        self.assertEqual(batch_inputs.shape, (2, 3, 10, 11))
        self.assertIsInstance(batch_data_samples['bboxes_labels'],
                              torch.Tensor)
        self.assertIsInstance(batch_data_samples['img_metas'], list)

        data = {
            'inputs': [torch.randint(0, 256, (3, 11, 10))],
            'data_samples': [DetDataSample()]
        }
        # data_samples must be dict
        with self.assertRaises(AssertionError):
            processor(data, training=True)


class TestPPYOLOEDetDataPreprocessor(TestCase):

    def test_batch_random_resize(self):
        processor = PPYOLOEDetDataPreprocessor(
            pad_size_divisor=32,
            batch_augments=[
                dict(
                    type='PPYOLOEBatchRandomResize',
                    random_size_range=(320, 480),
                    interval=1,
                    size_divisor=32,
                    random_interp=True,
                    keep_ratio=False)
            ],
            mean=[0., 0., 0.],
            std=[255., 255., 255.],
            bgr_to_rgb=True)
        self.assertTrue(
            isinstance(processor.batch_augments[0], PPYOLOEBatchRandomResize))
        message_hub = MessageHub.get_instance('test_batch_random_resize')
        message_hub.update_info('iter', 0)

        # test training
        data = {
            'inputs': [
                torch.randint(0, 256, (3, 10, 11)),
                torch.randint(0, 256, (3, 10, 11))
            ],
            'data_samples': {
                'bboxes_labels': torch.randint(0, 11, (18, 6)).float()
            },
        }
        out_data = processor(data, training=True)
        batch_data_samples = out_data['data_samples']
        self.assertIn('img_metas', batch_data_samples)
        self.assertIn('bboxes_labels', batch_data_samples)
        self.assertIsInstance(batch_data_samples['bboxes_labels'],
                              torch.Tensor)
        self.assertIsInstance(batch_data_samples['img_metas'], list)

        data = {
            'inputs': [torch.randint(0, 256, (3, 11, 10))],
            'data_samples': DetDataSample()
        }
        # data_samples must be list
        with self.assertRaises(AssertionError):
            processor(data, training=True)


class TestYOLOXDetDataPreprocessor(TestCase):

    def test_batch_sync_random_size(self):
        processor = YOLOXBatchSyncRandomResize(
            random_size_range=(480, 800), size_divisor=32, interval=1)
        self.assertTrue(isinstance(processor, YOLOXBatchSyncRandomResize))
        message_hub = MessageHub.get_instance(
            'test_yolox_batch_sync_random_resize')
        message_hub.update_info('iter', 0)

        # test training
        inputs = torch.randint(0, 256, (4, 3, 10, 11))
        data_samples = {'bboxes_labels': torch.randint(0, 11, (18, 6)).float()}

        inputs, data_samples = processor(inputs, data_samples)

        self.assertIn('bboxes_labels', data_samples)
        self.assertIsInstance(data_samples['bboxes_labels'], torch.Tensor)
        self.assertIsInstance(inputs, torch.Tensor)

        inputs = torch.randint(0, 256, (4, 3, 10, 11))
        data_samples = DetDataSample()

        # data_samples must be dict
        with self.assertRaises(AssertionError):
            processor(inputs, data_samples)
