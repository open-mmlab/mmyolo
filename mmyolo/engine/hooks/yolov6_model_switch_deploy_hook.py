# Copyright (c) OpenMMLab. All rights reserved.

from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmyolo.registry import HOOKS
from mmyolo.utils import switch_to_deploy


@HOOKS.register_module()
class YOLOv6DeploySwitchHook(Hook):
    """Switch YOLOv6 to deploy mode before testing.

    This hook converts the multi-channel structure of the training network
    (high performance) to the one-way structure of the testing network (fast
    speed and  memory saving).
    """

    def before_test_epoch(self, runner: Runner):
        switch_to_deploy(runner.model)
