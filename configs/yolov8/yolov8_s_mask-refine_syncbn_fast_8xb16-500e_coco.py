_base_ = './yolov8_s_syncbn_fast_8xb16-500e_coco.py'

pre_transform = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(
        type='LoadAnnotations', with_bbox=True, with_mask=True, mask2bbox=True)
]

last_transform = _base_.last_transform

train_pipeline = [
    *pre_transform,
    dict(
        type='Mosaic',
        img_scale=_base_.img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(0.5, 1.5),
        max_aspect_ratio=100.,
        # img_scale is (width, height)
        border=(-_base_.img_scale[0] // 2, -_base_.img_scale[1] // 2),
        border_val=(114, 114, 114),
        min_area_ratio=0.01,
        use_mask_refine=True),
    *last_transform
]

train_pipeline_stage2 = [
    *pre_transform,
    dict(type='YOLOv5KeepRatioResize', scale=_base_.img_scale),
    dict(
        type='LetterResize',
        scale=_base_.img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114.0)),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(0.5, 1.5),
        max_aspect_ratio=100.,
        border_val=(114, 114, 114)), *last_transform
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

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
