_base_ = './yolov8_s_syncbn_fast_8xb16-500e_coco.py'

deepen_factor = 0.67
widen_factor = 0.75
last_stage_out_channels = 768

affine_scale = 0.9
mixup_ratio = 0.1

num_classes = _base_.num_classes
num_det_layers = _base_.num_det_layers
img_scale = _base_.img_scale

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
last_transform = _base_.last_transform

mosaic_affine_transform = [
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(type='YOLOv5CopyPaste', prob=0.1),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_aspect_ratio=100,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114))
]

train_pipeline = [
    *pre_transform, *mosaic_affine_transform,
    dict(
        type='YOLOv5MixUp',
        prob=mixup_ratio,
        pre_transform=[*pre_transform, *mosaic_affine_transform]),
    *last_transform
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

train_pipeline_stage2 = [
    *pre_transform,
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114.0)),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        max_aspect_ratio=100,
        border_val=(114, 114, 114)), *last_transform
]

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=_base_.max_epochs - 10,
        switch_pipeline=train_pipeline_stage2)
]
