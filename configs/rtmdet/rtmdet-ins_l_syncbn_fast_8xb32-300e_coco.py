_base_ = './rtmdet_l_syncbn_fast_8xb32-300e_coco.py'
model = dict(
    bbox_head=dict(
        _delete_=True,
        type='RTMDetInsHead',
        head_module=dict(
            type='RTMDetInsSepBNHeadModule',
            num_classes=80,
            in_channels=256,
            stacked_convs=2,
            share_conv=True,
            pred_kernel_size=1,
            feat_channels=256,
            act_cfg=dict(type='SiLU', inplace=True),
            norm_cfg=dict(type='BN'),
            featmap_strides=_base_.strides),
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='mmdet.QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='mmdet.GIoULoss', loss_weight=2.0),
        loss_mask=dict(
            type='mmdet.DiceLoss', loss_weight=2.0, eps=5e-6, reduction='mean')),
    test_cfg=dict(
        multi_label=True,
        nms_pre=30000,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100,  # 300 is to slow.
        mask_thr_binary=0.5)  # 44.1/38.8
)

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=_base_.file_client_args),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='Mosaic',
        img_scale=_base_.img_scale,
        use_cached=True,
        max_cached_images=40,
        pad_val=114.0),
    dict(
        type='mmdet.RandomResize',
        scale=(1280, 1280),
        ratio_range=(0.1, 2.0),
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
        img_scale=_base_.img_scale,
        use_cached=True,
        ratio_range=(1.0, 1.0),
        max_cached_images=20,
        pad_val=(114, 114, 114)),
    dict(type='mmdet.FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(type='mmdet.PackDetInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

train_pipeline_stage2 = [
    dict(
        type='LoadImageFromFile',
        file_client_args=_base_.file_client_args),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='mmdet.RandomResize',
        scale=(640, 640),
        ratio_range=(0.1, 2.0),
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

val_evaluator = dict(metric=['bbox', 'segm'])
test_evaluator = val_evaluator
