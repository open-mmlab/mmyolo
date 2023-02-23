# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks.naive_visualization_hook import NaiveVisualizationHook
from mmengine.hooks import Hook
from mmyolo.registry import HOOKS


@HOOKS.register_module()
class WandbGradVisHook(Hook):
    """Switch to deploy mode before testing.

    This hook converts the multi-channel structure of the training network
    (high performance) to the one-way structure of the testing network (fast
    speed and  memory saving).
    """

    def before_train(self, runner) -> None:
        try:
            wandb = runner.visualizer.get_backend('WandbVisBackend')._wandb
            wandb.watch(runner.model, log_freq=1)
        except AttributeError:
            return