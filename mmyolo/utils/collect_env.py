# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import mmdet
from mmengine.utils import get_git_hash
from mmengine.utils.dl_utils import collect_env as collect_base_env

import mmyolo


def collect_env() -> dict:
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    env_info['MMCV'] = mmcv.__version__
    env_info['MMDetection'] = mmdet.__version__
    env_info['MMYOLO'] = mmyolo.__version__ + '+' + get_git_hash()[:7]
    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print(f'{name}: {val}')
