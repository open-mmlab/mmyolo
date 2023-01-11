_base_ = './yolov8_s_syncbn_8xb16-500e_coco.py'

deepen_factor = 1.00
widen_factor = 1.00

model = dict(
    type='YOLODetector',
    backbone=dict(
        type='YOLOv8CSPDarknet',
        arch='P5',
        out_channels=768,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor),
    neck=dict(
        type='YOLOv8PAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, 768],
        out_channels=[256, 512, 768]),
    bbox_head=dict(
        type='YOLOv8Head',
        head_module=dict(type='YOLOv8HeadModule', widen_factor=widen_factor)))
