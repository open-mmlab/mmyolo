_base_ = '../_base_/default_runtime.py'

# dataset settings
data_root = 'data/coco/'
dataset_type = 'YOLOv5CocoDataset'

# parameters that often need to be modified
img_scale = (640, 640)  # height, width
deepen_factor = 0.33
widen_factor = 0.5
max_epochs = 400
save_epoch_intervals = 10
train_batch_size_per_gpu = 2
train_num_workers = 0
val_batch_size_per_gpu = 1
val_num_workers = 2

# persistent_workers must be False if num_workers is 0.
persistent_workers = False

model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True),
    backbone=dict(
        type='YOLOv6EfficientRep',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='ReLU', inplace=True)),
    neck=dict(
        type='YOLOv6RepPAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, 1024],
        out_channels=[128, 256, 512],
        num_csp_blocks=12,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='ReLU', inplace=True),
    ),
    bbox_head=dict(
        type='YOLOv6Head',
        head_module=dict(
            type='YOLOv6HeadModule',
            num_classes=80,
            in_channels=[128, 256, 512],
            widen_factor=widen_factor,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True),
            featmap_strides=[8, 16, 32]),
        loss_cls=dict(
            type='mmdet.VarifocalLoss',
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=1.0),        
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='siou',
            bbox_format='xywh',
            eps=1e-16,
            reduction='mean',
            loss_weight=5.0),
    ),
    train_cfg=dict(
        initial_epoch=4,
        initial_assigner=dict(type='BatchATSSAssigner', topk=9, iou2d_calculator=dict(type='mmdet.BboxOverlaps2D'), num_classes=80),
        assigner=dict(type='BatchTaskAlignedAssigner', topk=13, alpha=1, beta=6),
        ))

pre_transform = [
    dict(
        type='LoadImageFromFile',
        file_client_args={{_base_.file_client_args}}),
    dict(type='LoadAnnotations', with_bbox=True)
]

train_pipeline_stage1 = [
    *pre_transform,
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='mmdet.RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='YOLOXMixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.FilterAnnotations',
        min_gt_bbox_wh=(1, 1),
        keep_empty=False),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]


train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline_stage1))

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args={{_base_.file_client_args}}),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='mmdet.Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
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
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

# Reduce evaluation time
val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox')

test_evaluator = val_evaluator

# optimizer
# default 8 gpu
base_lr = 0.01
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=base_lr, momentum=0.9, weight_decay=5e-4,
        nesterov=True),
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))



default_hooks = dict(
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook',
        scheduler_type='linear',
        lr_factor=0.01,
        max_epochs=max_epochs),
    checkpoint=dict(
        type='CheckpointHook', interval=save_epoch_intervals,
        max_keep_ckpts=3))


custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49)
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=save_epoch_intervals)

auto_scale_lr = dict(base_batch_size=64)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
