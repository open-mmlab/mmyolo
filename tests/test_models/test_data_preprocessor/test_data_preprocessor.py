# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmdet.structures import DetDataSample
from mmyolo.models.data_preprocessors import YOLOv5DetDataPreprocessor


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
            'data_samples': torch.randint(0, 11, (18, 6)),
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
        # data_samples must be tensor
        with self.assertRaises(AssertionError):
            processor(data, training=True)
