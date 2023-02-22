# Copyright (c) OpenMMLab. All rights reserved.
import time
import unittest
from unittest import TestCase

import torch
from mmdet.structures import DetDataSample
from mmdet.testing import demo_mm_inputs
from mmengine.logging import MessageHub
from parameterized import parameterized

from mmyolo.testing import get_detector_cfg
from mmyolo.utils import register_all_modules


class TestSingleStageDetector(TestCase):

    def setUp(self):
        register_all_modules()

    @parameterized.expand([
        'yolov5/yolov5_n-v61_syncbn_fast_8xb16-300e_coco.py',
        'yolov6/yolov6_s_syncbn_fast_8xb32-400e_coco.py',
        'yolox/yolox_tiny_fast_8xb8-300e_coco.py',
        'rtmdet/rtmdet_tiny_syncbn_fast_8xb32-300e_coco.py',
        'yolov7/yolov7_tiny_syncbn_fast_8x16b-300e_coco.py',
        'yolov8/yolov8_n_syncbn_fast_8xb16-500e_coco.py'
    ])
    def test_init(self, cfg_file):
        model = get_detector_cfg(cfg_file)
        model.backbone.init_cfg = None

        from mmyolo.registry import MODELS
        detector = MODELS.build(model)
        self.assertTrue(detector.backbone)
        self.assertTrue(detector.neck)
        self.assertTrue(detector.bbox_head)

    @parameterized.expand([
        ('yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py', ('cuda', 'cpu')),
        ('yolov7/yolov7_tiny_syncbn_fast_8x16b-300e_coco.py', ('cuda', 'cpu')),
        ('rtmdet/rtmdet_tiny_syncbn_fast_8xb32-300e_coco.py', ('cuda', 'cpu')),
        ('yolov8/yolov8_n_syncbn_fast_8xb16-500e_coco.py', ('cuda', 'cpu'))
    ])
    def test_forward_loss_mode(self, cfg_file, devices):
        message_hub = MessageHub.get_instance(
            f'test_single_stage_forward_loss_mode-{time.time()}')
        message_hub.update_info('iter', 0)
        message_hub.update_info('epoch', 0)
        model = get_detector_cfg(cfg_file)
        model.backbone.init_cfg = None

        if 'fast' in cfg_file:
            model.data_preprocessor = dict(
                type='mmdet.DetDataPreprocessor',
                mean=[0., 0., 0.],
                std=[255., 255., 255.],
                bgr_to_rgb=True)

        from mmyolo.registry import MODELS
        assert all([device in ['cpu', 'cuda'] for device in devices])

        for device in devices:
            detector = MODELS.build(model)
            detector.init_weights()

            if device == 'cuda':
                if not torch.cuda.is_available():
                    return unittest.skip('test requires GPU and torch+cuda')
                detector = detector.cuda()

            packed_inputs = demo_mm_inputs(2, [[3, 320, 128], [3, 125, 320]])
            data = detector.data_preprocessor(packed_inputs, True)
            losses = detector.forward(**data, mode='loss')
            self.assertIsInstance(losses, dict)

    @parameterized.expand([
        ('yolov5/yolov5_n-v61_syncbn_fast_8xb16-300e_coco.py', ('cuda',
                                                                'cpu')),
        ('yolov6/yolov6_s_syncbn_fast_8xb32-400e_coco.py', ('cuda', 'cpu')),
        ('yolox/yolox_tiny_fast_8xb8-300e_coco.py', ('cuda', 'cpu')),
        ('yolov7/yolov7_tiny_syncbn_fast_8x16b-300e_coco.py', ('cuda', 'cpu')),
        ('rtmdet/rtmdet_tiny_syncbn_fast_8xb32-300e_coco.py', ('cuda', 'cpu')),
        ('yolov8/yolov8_n_syncbn_fast_8xb16-500e_coco.py', ('cuda', 'cpu'))
    ])
    def test_forward_predict_mode(self, cfg_file, devices):
        model = get_detector_cfg(cfg_file)
        model.backbone.init_cfg = None

        from mmyolo.registry import MODELS
        assert all([device in ['cpu', 'cuda'] for device in devices])

        for device in devices:
            detector = MODELS.build(model)

            if device == 'cuda':
                if not torch.cuda.is_available():
                    return unittest.skip('test requires GPU and torch+cuda')
                detector = detector.cuda()

            packed_inputs = demo_mm_inputs(2, [[3, 320, 128], [3, 125, 320]])
            data = detector.data_preprocessor(packed_inputs, False)
            # Test forward test
            detector.eval()
            with torch.no_grad():
                batch_results = detector.forward(**data, mode='predict')
                self.assertEqual(len(batch_results), 2)
                self.assertIsInstance(batch_results[0], DetDataSample)

    @parameterized.expand([
        ('yolov5/yolov5_n-v61_syncbn_fast_8xb16-300e_coco.py', ('cuda',
                                                                'cpu')),
        ('yolov6/yolov6_s_syncbn_fast_8xb32-400e_coco.py', ('cuda', 'cpu')),
        ('yolox/yolox_tiny_fast_8xb8-300e_coco.py', ('cuda', 'cpu')),
        ('yolov7/yolov7_tiny_syncbn_fast_8x16b-300e_coco.py', ('cuda', 'cpu')),
        ('rtmdet/rtmdet_tiny_syncbn_fast_8xb32-300e_coco.py', ('cuda', 'cpu')),
        ('yolov8/yolov8_n_syncbn_fast_8xb16-500e_coco.py', ('cuda', 'cpu'))
    ])
    def test_forward_tensor_mode(self, cfg_file, devices):
        model = get_detector_cfg(cfg_file)
        model.backbone.init_cfg = None

        from mmyolo.registry import MODELS
        assert all([device in ['cpu', 'cuda'] for device in devices])

        for device in devices:
            detector = MODELS.build(model)

            if device == 'cuda':
                if not torch.cuda.is_available():
                    return unittest.skip('test requires GPU and torch+cuda')
                detector = detector.cuda()

            packed_inputs = demo_mm_inputs(2, [[3, 320, 128], [3, 125, 320]])
            data = detector.data_preprocessor(packed_inputs, False)
            batch_results = detector.forward(**data, mode='tensor')
            self.assertIsInstance(batch_results, tuple)
