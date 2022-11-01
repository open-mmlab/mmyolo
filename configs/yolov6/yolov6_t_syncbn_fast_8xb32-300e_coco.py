_base_ = './yolov6_s_syncbn_fast_8xb32-300e_coco.py'

deepen_factor = 0.33
widen_factor = 0.375

model = dict(
    backbone=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    neck=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    bbox_head=dict(
        type='YOLOv6Head',
        head_module=dict(widen_factor=widen_factor),
        loss_bbox=dict(iou_mode='siou')))
