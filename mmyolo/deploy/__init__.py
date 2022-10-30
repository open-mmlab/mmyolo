# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.codebase.base import MMCodebase

from .models import *  # noqa: F401,F403
from .object_detection import MMYOLO, YOLOObjectDetection

__all__ = ['MMCodebase', 'MMYOLO', 'YOLOObjectDetection']
