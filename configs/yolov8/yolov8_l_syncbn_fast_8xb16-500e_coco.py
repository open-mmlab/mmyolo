_base_ = './yolov8_m_syncbn_fast_8xb16-500e_coco.py'

deepen_factor = 1.00
widen_factor = 1.00
last_stage_out_channels = 512
mixup_ratio = 0.15

model = dict(
    backbone=dict(
        last_stage_out_channels=last_stage_out_channels,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, last_stage_out_channels],
        out_channels=[256, 512, last_stage_out_channels]),
    bbox_head=dict(
        head_module=dict(
            widen_factor=widen_factor,
            in_channels=[256, 512, last_stage_out_channels])))

pre_transform = _base_.pre_transform
albu_train_transform = _base_.albu_train_transform
mosaic_affine_transform = _base_.mosaic_affine_transform

last_transform = [
    dict(
        type='YOLOv5MixUp',
        prob=mixup_ratio,
        pre_transform=[*pre_transform, *mosaic_affine_transform]),
    dict(
        type='mmdet.Albu',
        transforms=albu_train_transform,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        }),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

train_pipeline = [
    *pre_transform,
    *mosaic_affine_transform,
    *last_transform
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
