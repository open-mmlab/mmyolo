# Copyright (c) OpenMMLab. All rights reserved.
import importlib
from mmdeploy.utils import get_root_logger
from mmdeploy.version import __version__  # noqa F401

if importlib.util.find_spec('torch'):
    importlib.import_module('mmdeploy.pytorch')
else:
    logger = get_root_logger()
    logger.debug('torch is not installed.')

if importlib.util.find_spec('mmcv'):
    importlib.import_module('mmdeploy.mmcv')
else:
    logger = get_root_logger()
    logger.debug('mmcv is not installed.')
