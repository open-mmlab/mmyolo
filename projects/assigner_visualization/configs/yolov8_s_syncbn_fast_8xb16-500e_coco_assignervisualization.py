_base_ = ['../../../configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py']

custom_imports = dict(imports=[
    'projects.assigner_visualization.detectors',
    'projects.assigner_visualization.dense_heads'
])

model = dict(
    type='YOLODetectorAssigner', bbox_head=dict(type='YOLOv8HeadAssigner'))
