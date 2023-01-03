# Copyright (c) OpenMMLab. All rights reserved.
from .backendwrapper import ORTWrapper, TRTWrapper
from .model import DeployModel

__all__ = ['DeployModel', 'TRTWrapper', 'ORTWrapper']
