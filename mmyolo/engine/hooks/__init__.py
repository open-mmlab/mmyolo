# Copyright (c) OpenMMLab. All rights reserved.
from .yolov5_param_scheduler_hook import YOLOv5ParamSchedulerHook
from .yolov6_model_switch_deploy_hook import YOLOv6DeploySwitchHook
from .yolox_mode_switch_hook import YOLOXModeSwitchHook

__all__ = [
    'YOLOv5ParamSchedulerHook', 'YOLOXModeSwitchHook', 'YOLOv6DeploySwitchHook'
]
