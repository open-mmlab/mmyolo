_base_ = './yolov6_s_syncbn_fast_8xb32-400e_coco.py'

deepen_factor = 0.6
widen_factor = 0.75

model = dict(
    backbone=dict(
        type='YOLOv6CSPBep',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        expansion=float(2) / 3,
        block_cfg=dict(type='ConvWrapper')),
    neck=dict(
        type='YOLOv6CSPRepPAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        expansion=float(2) / 3,
        block_cfg=dict(type='ConvWrapper')),
    bbox_head=dict(
        type='YOLOv6Head',
        head_module=dict(widen_factor=widen_factor),
    ))
