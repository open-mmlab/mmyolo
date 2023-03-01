# Copyright (c) OpenMMLab. All rights reserved.
from .rtmdet_head_assigner import RTMHeadAssigner
from .yolov5_head_assigner import YOLOv5HeadAssigner
from .yolov7_head_assigner import YOLOv7HeadAssigner
from .yolov8_head_assigner import YOLOv8HeadAssigner

__all__ = [
    'YOLOv5HeadAssigner', 'YOLOv7HeadAssigner', 'YOLOv8HeadAssigner',
    'RTMHeadAssigner'
]
