_base_ = './rtmdet_s_syncbn_fast_8xb32-300e_coco.py'
checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'  # noqa

# ========================modified parameters======================
deepen_factor = 0.167
widen_factor = 0.375
img_scale = _base_.img_scale

# ratio range for random resize
random_resize_ratio_range = (0.5, 2.0)
# Number of cached images in mosaic
mosaic_max_cached_images = 20
# Number of cached images in mixup
mixup_max_cached_images = 10

# =======================Unmodified in most cases==================
model = dict(
    backbone=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        init_cfg=dict(checkpoint=checkpoint)),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Mosaic',
        img_scale=img_scale,
        use_cached=True,
        max_cached_images=mosaic_max_cached_images,  # note
        random_pop=False,  # note
        pad_val=114.0),
    dict(
        type='mmdet.RandomResize',
        # img_scale is (width, height)
        scale=(img_scale[0] * 2, img_scale[1] * 2),
        ratio_range=random_resize_ratio_range,
        resize_type='mmdet.Resize',
        keep_ratio=True),
    dict(type='mmdet.RandomCrop', crop_size=img_scale),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(
        type='YOLOv5MixUp',
        use_cached=True,
        random_pop=False,
        max_cached_images=mixup_max_cached_images,
        prob=0.5),
    dict(type='mmdet.PackDetInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
