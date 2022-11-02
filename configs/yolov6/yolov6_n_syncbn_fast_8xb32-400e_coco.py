_base_ = './yolov6_s_syncbn_fast_8xb32-400e_coco.py'

deepen_factor = 0.33
widen_factor = 0.25

model = dict(
    backbone=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    neck=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    bbox_head=dict(
        head_module=dict(widen_factor=widen_factor),
        loss_bbox=dict(iou_mode='siou')))

default_hooks = dict(param_scheduler=dict(lr_factor=0.02))
