_base_ = '../../_base_/default_runtime.py'

data_root = '/datasets/dota_mmrotate_ss/'
dataset_type = 'YOLOv5DOTADataset'
checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'  # noqa

img_scale = (1024, 1024)  # width, height
deepen_factor = 0.167
widen_factor = 0.375
max_epochs = 36
interval = 12
num_classes = 15
angle_version = 'le90'

train_batch_size_per_gpu = 8
train_num_workers = 8
val_batch_size_per_gpu = 8
val_num_workers = 8
# persistent_workers must be False if num_workers is 0.
persistent_workers = True
strides = [8, 16, 32]
base_lr = 0.004 / 16

submission_dir = './work_dirs/rtm_dota/submission_convert3'

# single-scale training is recommended to
# be turned on, which can speed up training.
env_cfg = dict(cudnn_benchmark=True)

# only on Val
batch_shapes_cfg = dict(
    type='BatchShapePolicy',
    batch_size=val_batch_size_per_gpu,
    img_size=img_scale[0],
    size_divisor=32,
    extra_pad_ratio=0.5)

model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        channel_attention=True,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU', inplace=True),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint=checkpoint,
            map_location='cpu')),
    neck=dict(
        type='CSPNeXtPAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, 1024],
        out_channels=256,
        num_csp_blocks=3,
        expand_ratio=0.5,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='RotatedRTMDetHead',
        head_module=dict(
            type='RotatedRTMDetSepBNHeadModule',
            num_classes=num_classes,
            widen_factor=widen_factor,
            in_channels=256,
            stacked_convs=2,
            feat_channels=256,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='SiLU', inplace=True),
            share_conv=True,
            pred_kernel_size=1,
            featmap_strides=strides),
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0, strides=strides),
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version=angle_version),
        # bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='mmdet.QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmrotate.RotatedIoULoss', mode='linear', loss_weight=2.0),
        angle_version=angle_version,
        use_hbbox_loss=False,
        angle_coder=dict(type='mmrotate.PseudoAngleCoder'),
        loss_angle=None),
    train_cfg=dict(
        assigner=dict(
            type='BatchRotatedDSLAssigner',
            num_classes=num_classes,
            topk=13,
            iou_calculator=dict(type='mmrotate.RBboxOverlaps2D')),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        multi_label=True,
        decode_with_angle=True,
        nms_pre=6000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms_rotated', iou_threshold=0.1),
        max_per_img=2000),
)

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(
        type='mmrotate.ConvertBoxType',
        box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.RandomResize',
        scale=img_scale,
        ratio_range=(0.5, 2.0),  # note
        resize_type='mmdet.Resize',
        keep_ratio=True),
    dict(type='mmdet.RandomCrop', crop_size=img_scale),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(
        type='mmrotate.RandomRotate',
        prob=0.5,
        angle_range=180,
        rotate_type='mmrotate.Rotate',
        rect_obj_labels=[9, 11]),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='RegularizeRotatedBox', angle_version=angle_version),
    dict(type='mmdet.PackDetInputs')
]

val_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        box_type='qbox',
        _scope_='mmdet'),
    dict(
        type='mmrotate.ConvertBoxType',
        box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='mmdet.Pad', size=(1024, 1024),
        pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    collate_fn=dict(type='yolov5_collate'),
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='trainval/annfiles/',
        data_prefix=dict(img_path='trainval/images/'),
        img_shape=(1024, 1024),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val/annfiles/',
        data_prefix=dict(img_path='val/images/'),
        img_shape=(1024, 1024),
        test_mode=True,
        # batch_shapes_cfg=batch_shapes_cfg,
        pipeline=val_pipeline))

val_evaluator = dict(type='mmrotate.DOTAMetric', metric='mAP')

# Inference on val dataset
# test_dataloader = val_dataloader
# test_evaluator = val_evaluator

# Inference on test dataset and format the output results
# for submission. Note: the test set has no annotation.
test_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='test/images/'),
        img_shape=(1024, 1024),
        test_mode=True,
        batch_shapes_cfg=batch_shapes_cfg,
        pipeline=test_pipeline))
test_evaluator = dict(
    type='mmrotate.DOTAMetric',
    format_only=True,
    merge_patches=True,
    outfile_prefix=submission_dir)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        # use cosine lr from 150 to 300 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# hooks
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=interval,
        max_keep_ckpts=3  # only keep latest 3 checkpoints
    ))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49)
]

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=interval)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

visualizer = dict(type='mmrotate.RotLocalVisualizer')
