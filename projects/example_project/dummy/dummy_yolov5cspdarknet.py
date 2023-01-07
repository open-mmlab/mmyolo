# Copyright (c) OpenMMLab. All rights reserved.

from mmyolo.registry import MODELS
from mmyolo.models import YOLOv5CSPDarknet


@MODELS.register_module()
class DummyYOLOv5CSPDarknet(YOLOv5CSPDarknet):
    """Implements a dummy YOLOv5CSPDarknet wrapper for demonstration purpose.
    Args:
        **kwargs: All the arguments are passed to the parent class.
    """

    def __init__(self, **kwargs) -> None:
        print('Hello world!')
        super().__init__(**kwargs)
