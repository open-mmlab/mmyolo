_base_ = './yolov8_s_syncbn_8xb16-500e_coco.py'

deepen_factor = 0.33
widen_factor = 0.25

model = dict(
    type='YOLODetector',
    backbone=dict(
        type='YOLOv8CSPDarknet',
        arch='P5',
        out_channels=1024,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor),
    neck=dict(
        type='YOLOv8PAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor),
    bbox_head=dict(
        type='YOLOv8Head',
        head_module=dict(type='YOLOv8HeadModule', widen_factor=widen_factor)))
