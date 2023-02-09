# Copyright (c) OpenMMLab. All rights reserved.
import copy

import pytest
from mmcv import Config
from mmengine.model import BaseModel


@pytest.mark.parametrize('cfg_file', [
    './tests/data/configs_mmrazor/'
    'rtmdet_tiny_ofa_lat31_syncbn_16xb16-300e_coco.py'
])
def test_rtmdet_ofa_lat31_forward(cfg_file):
    config = Config.fromfile(cfg_file)
    model_cfg = copy.deepcopy(config.model)

    from mmrazor.registry import MODELS
    model = MODELS.build(model_cfg)
    assert isinstance(model, BaseModel)
