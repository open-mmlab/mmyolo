_base_ = './yolov6_v3_s_syncbn_fast_8xb32-300e_widerface.py'

# ======================= Possible modified parameters =======================
# -----model related-----
# The scaling factor that controls the depth of the network structure
deepen_factor = 0.67
# The scaling factor that controls the width of the network structure
widen_factor = 0.75

# -----train val related-----
affine_scale = 0.9  # YOLOv5RandomAffine scaling ratio

# ============================== Unmodified in most cases ===================
model = dict(
    backbone=dict(
        use_cspsppf=False,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor),
    neck=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    bbox_head=dict(head_module=dict(reg_max=16, widen_factor=widen_factor)))
