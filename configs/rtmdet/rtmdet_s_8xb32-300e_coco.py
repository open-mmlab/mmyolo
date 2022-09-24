_base_ = './rtmdet_l_8xb32-300e_coco.py'
checkpoint = 'TODO:imagenet_pretrain'  # noqa

img_scale = _base_.img_scale

model = dict(
    backbone=dict(
        deepen_factor=0.33,
        widen_factor=0.5,
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=checkpoint)),
    neck=dict(in_channels=[128, 256, 512], out_channels=128, num_csp_blocks=1),
    bbox_head=dict(head_module=dict(in_channels=128, feat_channels=128)))

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=_base_.file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Mosaic',
         img_scale=img_scale,
         use_cached=True,
         max_cached_images=40,
         pad_val=114.0),
    dict(
        type='mmdet.RandomResize',
        scale=(img_scale[0] * 2, img_scale[1] * 2),
        keep_ratio=True),
    dict(type='mmdet.RandomCrop', crop_size=img_scale),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(
        type='MixUp',
        img_scale=img_scale,
        use_cached=True,
        ratio_range=(1.0, 1.0),
        max_cached_images=20,
        pad_val=(114, 114, 114)),
    dict(type='mmdet.PackDetInputs')
]

train_pipeline_stage2 = [
    dict(
        type='LoadImageFromFile',
        file_client_args=_base_.file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='mmdet.RandomResize',
        scale=img_scale,
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='mmdet.RandomCrop', crop_size=img_scale),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='mmdet.PackDetInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=20,
        new_train_pipeline=train_pipeline_stage2)
]