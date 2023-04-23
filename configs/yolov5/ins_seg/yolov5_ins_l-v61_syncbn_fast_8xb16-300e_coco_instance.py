_base_ = './yolov5_ins_m-v61_sync_bn_fast_8xb16-300e_coco_instance.py'  # noqa

deepen_factor = 1.0
widen_factor = 1.0

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
