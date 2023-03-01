from typing import Union, Sequence, Tuple, List
from mmdet.utils import OptMultiConfig

from torch import Tensor
from mmyolo.models.utils import make_divisible
from mmengine.structures import InstanceData
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmengine.model import BaseModule

class YOLO6DHeadModule(BaseModule):
    """YOLO6DHead head module used in 'YOLO6D'.
    
    """
    def __init__(self,
                 num_classes: int,
                 in_channels: Union[int, Sequence],
                 widen_factor: float = 1.0,
                 num_base_priors: int = 3,
                 featmap_strides: Sequence[int] = (8, 16, 32),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)
    
    def forward():
    
    def forward_single():
        
        
class YOLO6DHead(BaseDenseHead):
    def __init__():
        
            
    def forward(self, x: Tuple[Tensor]):
        """Forward features from the upstream network
        
        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
            a 4D-tensor
        
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions, and objectnesses.
        """
        return self.head_module(x)
    
    def predict_by_feat(self,
                        cls_scores) -> List[InstanceData]
    
    def loss():
        
    def loss_by_feat():
        
        