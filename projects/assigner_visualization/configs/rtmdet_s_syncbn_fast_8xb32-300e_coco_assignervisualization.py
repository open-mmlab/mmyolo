_base_ = ['../../../configs/rtmdet/rtmdet_s_syncbn_fast_8xb32-300e_coco.py']

custom_imports = dict(imports=[
    'projects.assigner_visualization.detectors',
    'projects.assigner_visualization.dense_heads'
])

model = dict(
    type='YOLODetectorAssigner', bbox_head=dict(type='RTMHeadAssigner'))
