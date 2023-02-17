_base_ = './yolov5_s_mask-refine-v61_syncbn_fast_8xb16-300e_coco.py'

# This config will refine bbox by mask while loading annotations and
# transforming after `YOLOv5RandomAffine`

# ========================modified parameters======================
deepen_factor = 0.33
widen_factor = 0.25

# ===============================Unmodified in most cases====================
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
