# Copyright (c) OpenMMLab. All rights reserved.
import copy

import pytest
from mmcls.models.backbones.base_backbone import BaseBackbone

from mmyolo.testing import get_detector_cfg


@pytest.mark.parametrize('cfg_file', [
    'razor/subnets/'
    'yolov5_s_spos_shufflenetv2_syncbn_8xb16-300e_coco.py', 'razor/subnets/'
    'rtmdet_tiny_ofa_lat31_syncbn_16xb16-300e_coco.py', 'razor/subnets/'
    'yolov6_l_attentivenas_a6_d12_syncbn_fast_8xb32-300e_coco.py'
])
def test_razor_backbone_init(cfg_file):
    model = get_detector_cfg(cfg_file)
    model_cfg = copy.deepcopy(model.backbone)
    from mmrazor.registry import MODELS
    model = MODELS.build(model_cfg)
    assert isinstance(model, BaseBackbone)
