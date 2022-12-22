_base_ = [
    '../../../configs/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'
]

custom_imports = dict(imports=['projects.showassignment.dense_heads'])

model = dict(bbox_head=dict(type='YOLOv5HeadAssigner'))
