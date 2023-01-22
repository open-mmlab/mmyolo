# Copyright (c) OpenMMLab. All rights reserved.
from .collect_env import collect_env
from .misc import switch_to_deploy
from .setup_env import register_all_modules
from .verification_utils import (verify_inference, verify_mmcv, verify_testing,
                                 verify_training)

__all__ = [
    'register_all_modules', 'collect_env', 'switch_to_deploy', 'verify_mmcv',
    'verify_inference', 'verify_training', 'verify_testing'
]
