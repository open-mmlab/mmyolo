_base_ = './yolov5u_s_syncbn_fast_8xb16-300e_coco.py'

# ========================modified parameters======================
# TODO: Update the training hyperparameters
deepen_factor = 1.0
widen_factor = 1.0

# =======================Unmodified in most cases==================
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
