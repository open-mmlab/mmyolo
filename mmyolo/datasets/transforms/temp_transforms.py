# Copyright (c) OpenMMLab. All rights reserved.
import cv2
from mmcv import BaseTransform

from mmyolo.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PPYOLOECvt(BaseTransform):

    def transform(self, results):
        img = results['img']
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results['img'] = img
        return results
