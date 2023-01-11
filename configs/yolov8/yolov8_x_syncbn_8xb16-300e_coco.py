_base_ = './yolov8_m_syncbn_8xb16-500e_coco.py'

deepen_factor = 1.00
widen_factor = 1.25

model = dict(
    backbone=dict(
        last_stage_out_channels=512,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, 512],
        out_channels=[256, 512, 512]),
    bbox_head=dict(
        head_module=dict(
            widen_factor=widen_factor, in_channels=[256, 512, 512])))
