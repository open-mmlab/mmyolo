# Copyright (c) OpenMMLab. All rights reserved.
if '_base_':
    from .rtmdet_l_syncbn_fast_8xb32_300e_coco import *

from mmengine.model import PretrainedInit

checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e.pth'  # noqa

# ========================modified parameters======================
deepen_factor = 0.33
widen_factor = 0.5

# ratio range for random resize
random_resize_ratio_range = (0.5, 2.0)
# Number of cached images in mosaic
mosaic_max_cached_images = 40
# Number of cached images in mixup
mixup_max_cached_images = 20

# =======================Unmodified in most cases==================
model.update(
    backbone=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        # Since the checkpoint includes CUDA:0 data,
        # it must be forced to set map_location.
        # Once checkpoint is fixed, it can be removed.
        init_cfg=dict(
            type=PretrainedInit,
            prefix='backbone.',
            checkpoint=checkpoint,
            map_location='cpu')),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))

train_pipeline = [
    dict(type=LoadImageFromFile, file_client_args=file_client_args),
    dict(type=LoadAnnotations, with_bbox=True),
    dict(
        type=Mosaic,
        img_scale=img_scale,
        use_cached=True,
        max_cached_images=mosaic_max_cached_images,
        pad_val=114.0),
    dict(
        type=RandomResize,
        # img_scale is (width, height)
        scale=(img_scale[0] * 2, img_scale[1] * 2),
        ratio_range=random_resize_ratio_range,  # note
        resize_type=Resize,
        keep_ratio=True),
    dict(type=RandomCrop, crop_size=img_scale),
    dict(type=YOLOXHSVRandomAug),
    dict(type=RandomFlip, prob=0.5),
    dict(type=Pad, size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(
        type=YOLOv5MixUp,
        use_cached=True,
        max_cached_images=mixup_max_cached_images),
    dict(type=PackDetInputs)
]

train_pipeline_stage2 = [
    dict(type=LoadImageFromFile, file_client_args=file_client_args),
    dict(type=LoadAnnotations, with_bbox=True),
    dict(
        type=RandomResize,
        scale=img_scale,
        ratio_range=random_resize_ratio_range,  # note
        resize_type=Resize,
        keep_ratio=True),
    dict(type=RandomCrop, crop_size=img_scale),
    dict(type=YOLOXHSVRandomAug),
    dict(type=RandomFlip, prob=0.5),
    dict(type=Pad, size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type=PackDetInputs)
]

train_dataloader.update(dataset=dict(pipeline=train_pipeline))

custom_hooks = [
    dict(
        type=EMAHook,
        ema_type=ExpMomentumEMA,
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type=PipelineSwitchHook,
        switch_epoch=max_epochs - num_epochs_stage2,
        switch_pipeline=train_pipeline_stage2)
]
