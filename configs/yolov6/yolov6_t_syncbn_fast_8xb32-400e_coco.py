_base_ = './yolov6_s_syncbn_fast_8xb32-400e_coco.py'

# ======================= Possible modified parameters =======================
# -----model related-----
# The scaling factor that controls the depth of the network structure
deepen_factor = 0.33
# The scaling factor that controls the width of the network structure
widen_factor = 0.375

# ============================== Unmodified in most cases ===================
model = dict(
    backbone=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    neck=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    bbox_head=dict(
        type='YOLOv6Head',
        head_module=dict(widen_factor=widen_factor),
        loss_bbox=dict(iou_mode='siou')))
