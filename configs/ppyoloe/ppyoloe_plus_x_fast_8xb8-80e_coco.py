_base_ = './ppyoloe_plus_s_fast_8xb8-80e_coco.py'

deepen_factor = 1.33
widen_factor = 1.25

model = dict(
    backbone=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))
