_base_ = './yolov5_s-v61_syncbn_fast_8xb8-50e_voc.py'

deepen_factor = 0.67
widen_factor = 0.75

model = dict(
    backbone=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))
