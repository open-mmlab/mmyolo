# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import sys
from unittest import TestCase

from mmengine import DefaultScope

from mmyolo.utils import register_all_modules


class TestSetupEnv(TestCase):

    def test_register_all_modules(self):
        from mmyolo.registry import DATASETS

        # not init default scope
        sys.modules.pop('mmyolo.datasets', None)
        sys.modules.pop('mmyolo.datasets.yolov5_coco', None)
        DATASETS._module_dict.pop('YOLOv5CocoDataset', None)
        self.assertFalse('YOLOv5CocoDataset' in DATASETS.module_dict)
        register_all_modules(init_default_scope=False)
        self.assertTrue('YOLOv5CocoDataset' in DATASETS.module_dict)

        # init default scope
        sys.modules.pop('mmyolo.datasets', None)
        sys.modules.pop('mmyolo.datasets.yolov5_coco', None)
        DATASETS._module_dict.pop('YOLOv5CocoDataset', None)
        self.assertFalse('YOLOv5CocoDataset' in DATASETS.module_dict)
        register_all_modules(init_default_scope=True)
        self.assertTrue('YOLOv5CocoDataset' in DATASETS.module_dict)
        self.assertEqual(DefaultScope.get_current_instance().scope_name,
                         'mmyolo')

        # init default scope when another scope is init
        name = f'test-{datetime.datetime.now()}'
        DefaultScope.get_instance(name, scope_name='test')
        with self.assertWarnsRegex(
                Warning, 'The current default scope "test" is not "mmyolo"'):
            register_all_modules(init_default_scope=True)
