# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import Mock

from mmyolo.engine.hooks import YOLOv6DeploySwitchHook
from mmyolo.models import RepVGGBlock
from mmyolo.utils import register_all_modules

register_all_modules()


class TestYOLOv6DeploySwitchHook(TestCase):

    def test(self):

        runner = Mock()
        runner.model = RepVGGBlock(256, 256)

        hook = YOLOv6DeploySwitchHook()
        self.assertFalse(runner.model.deploy)

        # test after change mode
        hook.before_test_epoch(runner)
        self.assertTrue(runner.model.deploy)
