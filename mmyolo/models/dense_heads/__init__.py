# Copyright (c) OpenMMLab. All rights reserved.
from .ppyoloe_head import PPYOLOEHead, PPYOLOEHeadModule
from .rtmdet_head import RTMDetHead, RTMDetSepBNHeadModule
from .rtmdet_ins_head import RTMDetInsSepBNHead, RTMDetInsSepBNHeadModule
from .rtmdet_rotated_head import (RTMDetRotatedHead,
                                  RTMDetRotatedSepBNHeadModule)
from .yolov5_head import YOLOv5Head, YOLOv5HeadModule
from .yolov5_ins_head import YOLOv5InsHead, YOLOv5InsHeadModule
from .yolov6_head import YOLOv6Head, YOLOv6HeadModule
from .yolov7_head import YOLOv7Head, YOLOv7HeadModule, YOLOv7p6HeadModule
from .yolov8_head import YOLOv8Head, YOLOv8HeadModule
from .yolox_head import YOLOXHead, YOLOXHeadModule
from .yolox_pose_head import YOLOXPoseHead, YOLOXPoseHeadModule

__all__ = [
    'YOLOv5Head', 'YOLOv6Head', 'YOLOXHead', 'YOLOv5HeadModule',
    'YOLOv6HeadModule', 'YOLOXHeadModule', 'RTMDetHead',
    'RTMDetSepBNHeadModule', 'YOLOv7Head', 'PPYOLOEHead', 'PPYOLOEHeadModule',
    'YOLOv7HeadModule', 'YOLOv7p6HeadModule', 'YOLOv8Head', 'YOLOv8HeadModule',
    'RTMDetRotatedHead', 'RTMDetRotatedSepBNHeadModule', 'RTMDetInsSepBNHead',
    'RTMDetInsSepBNHeadModule', 'YOLOv5InsHead', 'YOLOv5InsHeadModule',
    'YOLOXPoseHead', 'YOLOXPoseHeadModule'
]
