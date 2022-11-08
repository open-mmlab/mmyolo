_base_ = '../_base_/default_runtime.py'

# dataset settings
data_root = 'data/coco/'
dataset_type = 'YOLOv5CocoDataset'

# parameters that often need to be modified
img_scale = (640, 640)  # height, width
deepen_factor = 0.33
widen_factor = 0.5
max_epochs = 80
save_epoch_intervals = 10
train_batch_size_per_gpu = 8
train_num_workers = 8
val_batch_size_per_gpu = 1
val_num_workers = 2

# persistent_workers must be False if num_workers is 0.
persistent_workers = True

strides = [8, 16, 32]

model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        type='PPYOLOEDetDataPreprocessor',
        pad_size_divisor=None,
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True,
        batch_augments=[
            dict(
                type='PPYOLOEBatchSyncRandomResize',
                random_size_range=(320, 800),
                interval=1,
                size_divisor=32,
                random_interp=True,
                interp_mode=['nearest', 'bilinear', 'bicubic', 'area'],
                keep_ratio=False)
        ]),
    backbone=dict(
        type='PPYOLOECSPResNet',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        block_cfg=dict(
            type='PPYOLOEBasicBlock', shortcut=True, use_alpha=True),
        norm_cfg=dict(type='BN', momentum=0.1, eps=1e-5),
        act_cfg=dict(type='SiLU', inplace=True),
        attention_cfg=dict(
            type='EffectiveSELayer', act_cfg=dict(type='HSigmoid')),
        use_large_stem=True),
    neck=dict(
        type='PPYOLOECSPPAFPN',
        in_channels=[256, 512, 1024],
        out_channels=[192, 384, 768],
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        num_csplayer=1,
        num_blocks_per_layer=3,
        block_cfg=dict(
            type='PPYOLOEBasicBlock', shortcut=False, use_alpha=False),
        norm_cfg=dict(type='BN', momentum=0.1, eps=1e-5),
        act_cfg=dict(type='SiLU', inplace=True),
        drop_block_cfg=None,
        use_spp=True),
    bbox_head=dict(
        type='PPYOLOEHead',
        head_module=dict(
            type='PPYOLOEHeadModule',
            num_classes=80,
            in_channels=[192, 384, 768],
            widen_factor=widen_factor,
            featmap_strides=strides,
            num_base_priors=1)),
    test_cfg=dict(
        multi_label=True,
        nms_pre=1000,
        score_thr=0.01,
        nms=dict(type='nms', iou_threshold=0.7),
        max_per_img=300))

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args={{_base_.file_client_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PPYOLOERandomDistort'),
    dict(type='PPYOLOERandomExpand'),
    dict(type='PPYOLOERandomCrop'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=0),
        pipeline=train_pipeline))

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args={{_base_.file_client_args}}),
    dict(
        type='mmdet.FixShapeResize',
        width=img_scale[1],
        height=img_scale[0],
        keep_ratio=False,
        interpolation='bicubic'),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

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
        test_mode=True,
        data_prefix=dict(img='val2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=0),
        ann_file='annotations/instances_val2017.json',
        pipeline=test_pipeline))

test_dataloader = val_dataloader

param_scheduler = None
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=0.001, momentum=0.9, weight_decay=5e-4, nesterov=False),
    paramwise_cfg=dict(norm_decay_mult=0.))

default_hooks = dict(
    param_scheduler=dict(
        type='PPYOLOEParamSchedulerHook',
        warmup_min_iter=1000,
        start_factor=0.,
        warmup_epochs=5,
        min_lr_ratio=0.0,
        total_epochs=96),
    checkpoint=dict(
        type='CheckpointHook', interval=save_epoch_intervals,
        max_keep_ckpts=3))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49)
]

val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=save_epoch_intervals)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
