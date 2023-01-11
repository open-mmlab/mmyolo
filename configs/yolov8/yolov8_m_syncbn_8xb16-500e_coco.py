_base_ = './yolov8_s_syncbn_8xb16-500e_coco.py'

deepen_factor = 0.67
widen_factor = 0.75

model = dict(
    backbone=dict(
        last_stage_out_channels=768,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, 768],
        out_channels=[256, 512, 768]),
    bbox_head=dict(
        head_module=dict(
            widen_factor=widen_factor, in_channels=[256, 512, 768])))
