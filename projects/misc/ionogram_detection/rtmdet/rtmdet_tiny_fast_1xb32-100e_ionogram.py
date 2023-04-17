_base_ = './rtmdet_s_fast_1xb32-100e_ionogram.py'

# ======================= Modified parameters ======================
deepen_factor = 0.167
widen_factor = 0.375
img_scale = _base_.img_scale

load_from = 'https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_tiny_syncbn_fast_8xb32-300e_coco/rtmdet_tiny_syncbn_fast_8xb32-300e_coco_20230102_140117-dbb1dc83.pth'  # noqa

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0,
        end=300),
    dict(
        # use cosine lr from 50 to 100 epoch
        type='CosineAnnealingLR',
        eta_min=_base_.base_lr * 0.05,
        begin=_base_.max_epochs // 2,
        end=_base_.max_epochs,
        T_max=_base_.max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# =======================Unmodified in most cases==================
model = dict(
    backbone=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    neck=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Mosaic',
        img_scale=img_scale,
        use_cached=True,
        max_cached_images=20,  # note
        random_pop=False,  # note
        pad_val=114.0),
    dict(
        type='mmdet.RandomResize',
        # img_scale is (width, height)
        scale=(img_scale[0] * 2, img_scale[1] * 2),
        ratio_range=(0.5, 2.0),
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
        max_cached_images=10,
        prob=0.5),
    dict(type='mmdet.PackDetInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
