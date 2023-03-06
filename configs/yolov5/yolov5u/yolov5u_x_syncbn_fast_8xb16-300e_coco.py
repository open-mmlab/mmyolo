_base_ = './yolov5u_l_syncbn_fast_8xb16-300e_coco.py'

# ========================modified parameters======================
# TODO: Update the training hyperparameters
deepen_factor = 1.33
widen_factor = 1.25

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
