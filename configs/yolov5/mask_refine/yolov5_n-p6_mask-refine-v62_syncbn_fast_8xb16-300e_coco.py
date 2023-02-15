_base_ = 'yolov5_s-p6_mask-refine-v62_syncbn_fast_8xb16-300e_coco.py'

# ========================modified parameters======================
deepen_factor = 0.33
widen_factor = 0.25

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
