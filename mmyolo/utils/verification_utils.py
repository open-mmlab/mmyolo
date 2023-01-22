# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.ops import box_iou_rotated


def verify_mmcv(boxes1: np.ndarray, boxes2: np.ndarray):
    """Verify whether mmcv has been installed successfully."""
    boxes1 = torch.from_numpy(boxes1)
    boxes2 = torch.from_numpy(boxes2)

    # test mmcv with CPU ops
    box_iou_rotated(boxes1, boxes2)
    print('CPU ops were compiled successfully.')

    # test mmcv with both CPU and CUDA ops
    if torch.cuda.is_available():
        boxes1 = boxes1.cuda()
        boxes2 = boxes2.cuda()
        box_iou_rotated(boxes1, boxes2)
        print('CUDA ops were compiled successfully.')
    else:
        print('No CUDA runtime is found, skipping the verifying of CUDA ops.')


def verify_inference():
    # TODO
    """Verify whether inference can be run successfully."""
    pass


def verify_training():
    # TODO
    """Verify whether training can be run successfully."""
    pass


def verify_testing():
    # TODO
    """Verify whether testing can be run successfully."""
    pass
