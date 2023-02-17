_base_ = './rtmdet_s_syncbn_fast_8xb32-300e_coco.py'

# ========================modified parameters======================
use_mask2refine = True

# ===============================Unmodified in most cases====================
img_scale = _base_.img_scale

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='LoadAnnotations',
         with_bbox=True,
         with_mask=True,
         mask2bbox=use_mask2refine),
    dict(
        type='Mosaic',
        img_scale=img_scale,
        use_cached=True,
        max_cached_images=_base_.mosaic_max_cached_images,
        pad_val=114.0),
    dict(
        type='mmdet.RandomResize',
        # img_scale is (width, height)
        scale=(img_scale[0] * 2, img_scale[1] * 2),
        ratio_range=_base_.random_resize_ratio_range,  # note
        resize_type='mmdet.Resize',
        keep_ratio=True),
    dict(type='mmdet.RandomCrop',
         crop_size=img_scale,
         allow_negative_crop=True,
         recompute_bbox=use_mask2refine),
    # Delete gt_masks to avoid more computation
    dict(type='RemoveDataElement', keys=['gt_masks']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(
        type='YOLOv5MixUp',
        use_cached=True,
        max_cached_images=_base_.mixup_max_cached_images),
    dict(type='mmdet.PackDetInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='LoadAnnotations',
         with_bbox=True,
         with_mask=True,
         mask2bbox=use_mask2refine),
    dict(
        type='mmdet.RandomResize',
        scale=img_scale,
        ratio_range=_base_.random_resize_ratio_range,  # note
        resize_type='mmdet.Resize',
        keep_ratio=True),
    dict(type='mmdet.RandomCrop',
         crop_size=img_scale,
         allow_negative_crop=True,
         recompute_bbox=use_mask2refine),
    # Delete gt_masks to avoid more computation
    dict(type='RemoveDataElement', keys=['gt_masks']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='mmdet.PackDetInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
_base_.custom_hooks[1].switch_pipeline = train_pipeline_stage2
