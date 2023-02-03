_base_ = 'yolox_s_8xb8-300e_coco.py'

# fast means faster training speed,
# but less flexibility for multitasking
model = dict(
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True))

train_dataloader = dict(collate_fn=dict(type='yolov5_collate'))