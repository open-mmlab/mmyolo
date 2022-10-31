# Copyright (c) OpenMMLab. All rights reserved.
import pytest


@pytest.fixture(autouse=True)
def init_test():
    # init default scope
    from mmdet.utils import register_all_modules as register_det

    from mmyolo.utils import register_all_modules as register_yolo

    register_yolo(True)
    register_det(False)
