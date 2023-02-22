# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.structures import BaseDataElement, InstanceData, PixelData

class DataSample6D(BaseDataElement):
    
    @property
    def gt_instances(self) -> InstanceData:
        return self._gt_instances
    
    @gt_instances.setter
    def gt_instances(self, value: InstanceData):
        self.set_field(value, '_gt_instances', dtype=InstanceData)
    
    @gt_instances.deleter
    def gt_instances(self):
        del self._gt_instances
    
    @property
    def pred_instances(self) -> InstanceData:
        return self._pred_instances
    
    @pred_instances.setter
    def pred_instances(self, value: InstanceData):
        self.set_field(value, '_pred_instances', dtype=InstanceData)
    
    @pred_instances.deleter
    def pred_instances(self):
        del self._pred_instances
    