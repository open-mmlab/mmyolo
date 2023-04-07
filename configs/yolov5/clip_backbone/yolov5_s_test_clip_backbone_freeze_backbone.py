_base_ = 'yolov5_s_test_clip_backbone_freeze_bn.py'

model = dict(
    backbone=dict(
        freeze_backbone=True,  # 冻结backbone
        freeze_bn=False))
