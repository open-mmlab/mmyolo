_base_ = './rtmdet-ins_l_syncbn_fast_8xb32-300e_coco.py'
checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e.pth'  # noqa

deepen_factor = 0.33
widen_factor = 0.5
img_scale = _base_.img_scale

model = dict(
    backbone=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=checkpoint)),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='Mosaic',
        img_scale=img_scale,
        use_cached=True,
        max_cached_images=40,
        pad_val=114.0),
    dict(
        type='mmdet.RandomResize',
        scale=(1280, 1280),
        ratio_range=(0.5, 2.0),
        resize_type='mmdet.Resize',
        keep_ratio=True),
    dict(
        type='mmdet.RandomCrop',
        crop_size=(640, 640),
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='YOLOXMixUp',
        img_scale=img_scale,
        use_cached=True,
        ratio_range=(1.0, 1.0),
        max_cached_images=20,
        pad_val=(114, 114, 114)),
    dict(type='mmdet.FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(type='mmdet.PackDetInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='mmdet.RandomResize',
        scale=(640, 640),
        ratio_range=(0.5, 2.0),
        resize_type='mmdet.Resize',
        keep_ratio=True),
    dict(
        type='mmdet.RandomCrop',
        crop_size=(640, 640),
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='mmdet.FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='mmdet.PackDetInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=280,
        switch_pipeline=train_pipeline_stage2)
]
