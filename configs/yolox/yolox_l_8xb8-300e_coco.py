_base_ = './yolox_s_8xb8-300e_coco.py'

deepen_factor = 1.0
widen_factor = 1.0

# model settings
model = dict(
    backbone=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    neck=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))
