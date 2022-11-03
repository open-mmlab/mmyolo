# Copyright (c) OpenMMLab. All rights reserved.
from mmyolo.models import RepVGGBlock


def switch_to_deploy(model):
    """Model switch to deploy status."""
    for layer in model.modules():
        if isinstance(layer, RepVGGBlock):
            layer.switch_to_deploy()

    print('Switch model to deploy modality.')
