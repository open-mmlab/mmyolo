_base_ = 'yolov5_l_clip_backbone_freeze_bn.py'

model = dict(
    backbone=dict(
        freeze_backbone=True,  # 冻结backbone
        freeze_bn=False))
