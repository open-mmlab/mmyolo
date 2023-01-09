_base_ = '../../../configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py'

custom_imports = dict(imports=['projects.example_project.dummy'])

_base_.model.backbone.type = 'DummyYOLOv5CSPDarknet'
