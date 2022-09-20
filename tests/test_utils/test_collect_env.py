# Copyright (c) OpenMMLab. All rights reserved.
import sys
from unittest import TestCase

import mmcv
import mmdet
import mmengine

from mmyolo.utils import collect_env


class TestCollectEnv(TestCase):

    def test_collect_env(self):
        env_info = collect_env()
        print(env_info)
        expected_keys = [
            'sys.platform', 'Python', 'CUDA available', 'PyTorch',
            'PyTorch compiling details', 'OpenCV', 'MMEngine', 'GCC'
        ]
        for key in expected_keys:
            assert key in env_info

        if env_info['CUDA available']:
            for key in ['CUDA_HOME', 'NVCC']:
                assert key in env_info

        assert env_info['sys.platform'] == sys.platform
        assert env_info['Python'] == sys.version.replace('\n', '')

        assert env_info['MMEngine'] == mmengine.__version__
        assert env_info['MMCV'] == mmcv.__version__
        assert env_info['MMDetection'] == mmdet.__version__
