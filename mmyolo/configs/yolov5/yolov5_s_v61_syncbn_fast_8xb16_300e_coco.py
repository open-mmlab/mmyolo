if '_base_':
    from .yolov5_s_v61_syncbn_8xb16_300e_coco import *
from mmyolo.models.data_preprocessors.data_preprocessor import YOLOv5DetDataPreprocessor
from mmyolo.datasets.utils import yolov5_collate

# fast means faster training speed,
# but less flexibility for multitasking
model.merge(
    dict(
        data_preprocessor=dict(
            type=YOLOv5DetDataPreprocessor,
            mean=[0., 0., 0.],
            std=[255., 255., 255.],
            bgr_to_rgb=True)))

train_dataloader.merge(dict(collate_fn=dict(type=yolov5_collate)))
