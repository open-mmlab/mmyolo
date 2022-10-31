_base_ = './yolov6_s_syncbn_fast_8xb32-400e_coco.py'

deepen_factor = 1
widen_factor = 1

model = dict(
    backbone=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        block_cfg=dict(type='ConvWrapper'),
        stage_cfg=dict(type='BepC3StageBlock', e=float(1) / 2)),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        block_cfg=dict(type='ConvWrapper'),
        stage_cfg=dict(type='BepC3StageBlock', e=float(1) / 2)),
    bbox_head=dict(
        type='YOLOv6Head',
        head_module=dict(widen_factor=widen_factor),
    ))
