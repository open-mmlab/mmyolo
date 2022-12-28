_base_ = [
    '../../../configs/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'
]

custom_imports = dict(imports=[
    'projects.assigner_visualization.detectors',
    'projects.assigner_visualization.dense_heads'
])

model = dict(
    type='YOLODetectorAssigner', bbox_head=dict(type='YOLOv5HeadAssigner'))
