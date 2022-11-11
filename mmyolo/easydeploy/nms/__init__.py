# Copyright (c) OpenMMLab. All rights reserved.
from .trt_nms import batched_nms, efficient_nms

__all__ = ['efficient_nms', 'batched_nms']
