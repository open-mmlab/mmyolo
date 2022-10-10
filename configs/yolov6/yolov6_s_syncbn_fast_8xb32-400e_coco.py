_base_ = 'yolov6_s_syncbn_8xb32-400e_coco.py'
load_from='mmyolov6s.pt'
# fast means faster training speed,
# but less flexibility for multitasking
model = dict(
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True))
train_dataloader = dict(collate_fn=dict(type='yolov5_collate'))