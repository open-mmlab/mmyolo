# Copyright (c) OpenMMLab. All rights reserved.
import copy

import pytest
from mmcls.models.backbones.base_backbone import BaseBackbone
from mmengine.config import Config


@pytest.mark.parametrize('cfg_file', [
    './tests/test_downstream/configs_mmrazor/'
    'yolov5_s_spos_shufflenetv2_syncbn_8xb16-300e_coco.py',
    './tests/test_downstream/configs_mmrazor/'
    'rtmdet_tiny_ofa_lat31_syncbn_16xb16-300e_coco.py',
    './tests/test_downstream/configs_mmrazor/'
    'yolov6_l_attentivenas_a6_d12_syncbn_fast_16xb16-300e_coco.py'
])
def test_yolov5_s_spos_forward(cfg_file):
    config = Config.fromfile(cfg_file)
    model_cfg = copy.deepcopy(config.model.backbone)
    from mmrazor.registry import MODELS
    model = MODELS.build(model_cfg)
    assert isinstance(model, BaseBackbone)
