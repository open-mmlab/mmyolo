_base_ = './rtmdet_s_syncbn_fast_8xb32-300e_coco.py'

model = dict(
    bbox_head=dict(
        type='RTMDetInsSepBNHead',
        head_module=dict(
            _delete_=True,
            type='RTMDetInsSepBNHeadModule',
            num_classes=80,
            in_channels=256,
            feat_channels=256,
            stacked_convs=2,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='SiLU', inplace=True),
            share_conv=True,
            pred_kernel_size=1,
            featmap_strides=[8, 16, 32],
            widen_factor=0.5),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_mask=dict(
            type='mmdet.DiceLoss', loss_weight=2.0, eps=5e-6,
            reduction='mean')),
    test_cfg=dict(
        multi_label=False,
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100,
        mask_thr_binary=0.5))

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Mosaic',
        img_scale=_base_.img_scale,
        use_cached=True,
        max_cached_images=_base_.mosaic_max_cached_images,
        pad_val=114.0),
    dict(
        type='mmdet.RandomResize',
        # img_scale is (width, height)
        scale=(_base_.img_scale[0] * 2, _base_.img_scale[1] * 2),
        ratio_range=_base_.random_resize_ratio_range,  # note
        resize_type='mmdet.Resize',
        keep_ratio=True),
    dict(type='mmdet.RandomCrop', crop_size=_base_.img_scale),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.Pad',
        size=_base_.img_scale,
        pad_val=dict(img=(114, 114, 114))),
    dict(
        type='YOLOv5MixUp',
        use_cached=True,
        max_cached_images=_base_.mixup_max_cached_images),
    dict(type='mmdet.PackDetInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='mmdet.RandomResize',
        scale=_base_.img_scale,
        ratio_range=_base_.random_resize_ratio_range,  # note
        resize_type='mmdet.Resize',
        keep_ratio=True),
    dict(type='mmdet.RandomCrop', crop_size=_base_.img_scale),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.Pad',
        size=_base_.img_scale,
        pad_val=dict(img=(114, 114, 114))),
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
        switch_epoch=_base_.max_epochs - _base_.num_epochs_stage2,
        switch_pipeline=train_pipeline_stage2)
]

val_evaluator = dict(metric=['bbox', 'segm'])
test_evaluator = val_evaluator
