# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmyolo.utils import (verify_inference, verify_mmcv, verify_testing,
                          verify_training)

if __name__ == '__main__':
    # verify mmcv
    print('Start verifying the installation of mmcv ...')
    np_boxes1 = np.asarray(
        [[1.0, 1.0, 3.0, 4.0, 0.5], [2.0, 2.0, 3.0, 4.0, 0.6],
         [7.0, 7.0, 8.0, 8.0, 0.4]],
        dtype=np.float32)
    np_boxes2 = np.asarray(
        [[0.0, 2.0, 2.0, 5.0, 0.3], [2.0, 1.0, 3.0, 3.0, 0.5],
         [5.0, 5.0, 6.0, 7.0, 0.4]],
        dtype=np.float32)
    verify_mmcv(np_boxes1, np_boxes2)
    print('mmcv has been installed successfully.')
    # verify inference
    print('Start verifying the inference of mmyolo ...')
    verify_inference()
    print('Inference can be run successfully.')
    # verify training
    print('Start verifying the training of mmyolo ...')
    verify_training()
    print('Training can be run successfully.')
    # verify testing
    print('Start verifying the testing of mmyolo ...')
    verify_testing()
    print('Testing can be run successfully.')
