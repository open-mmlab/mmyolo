from mmengine.config import ConfigDict

from mmyolo.models import YOLOv8Head
from mmyolo.utils import register_all_modules

register_all_modules()

num_classes = 80
last_stage_out_channels = 1024
widen_factor = 0.5
model = YOLOv8Head(
    head_module=ConfigDict(
        dict(
            type='YOLOv8HeadModule',
            num_classes=num_classes,
            in_channels=[256, 512, last_stage_out_channels],
            widen_factor=widen_factor,
            reg_max=16,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True),
            featmap_strides=[8, 16, 32])),
    train_cfg=ConfigDict(
        dict(
            assigner=dict(
                type='BatchTaskAlignedAssigner',
                num_classes=num_classes,
                topk=10,
                alpha=0.5,
                beta=6.0,
                eps=1e-9,
                use_ciou=True))))
model.cuda()

model.loss_by_feat(None, None, None, None, None, None)
