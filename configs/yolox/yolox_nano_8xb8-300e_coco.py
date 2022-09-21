_base_ = './yolox_tiny_8xb8-300e_coco.py'

deepen_factor = 0.33
widen_factor = 0.25
use_depthwise = True

# model settings
model = dict(
    backbone=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        use_depthwise=use_depthwise),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        use_depthwise=use_depthwise),
    bbox_head=dict(
        head_module=dict(
            widen_factor=widen_factor, use_depthwise=use_depthwise)))
