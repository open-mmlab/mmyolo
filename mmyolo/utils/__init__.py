# Copyright (c) OpenMMLab. All rights reserved.
from .collect_env import collect_env
from .misc import judge_metainfo_is_lower, switch_to_deploy
from .setup_env import register_all_modules

__all__ = [
    'register_all_modules', 'collect_env', 'switch_to_deploy',
    'judge_metainfo_is_lower'
]
