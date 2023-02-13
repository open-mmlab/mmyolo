_base_ = ['../../../configs/yolov7/yolov7_tiny_syncbn_fast_8x16b-300e_coco.py']

custom_imports = dict(imports=[
    'projects.assigner_visualization.detectors',
    'projects.assigner_visualization.dense_heads'
])

model = dict(
    type='YOLODetectorAssigner', bbox_head=dict(type='YOLOv7HeadAssigner'))
