# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.backbones import ResNet
from mmdet.registry import MODELS


@MODELS.register_module()
class DummyYOLOv5CSPDarknet(YOLOv5CSPDarknet):
    
    """Implements a dummy YOLOv5CSPDarknet wrapper for demonstration purpose.
    Args:
        **kwargs: All the arguments are passed to the parent class.
    """

    def __init__(self, **kwargs) -> None:
        print('Hello world!')
        super().__init__(**kwargs)
