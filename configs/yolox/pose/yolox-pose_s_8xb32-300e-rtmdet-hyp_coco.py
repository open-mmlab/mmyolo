_base_ = ['../yolox_s_fast_8xb32-300e-rtmdet-hyp_coco.py']

# model settings
model = dict(
    type='YOLODetector',
    init_cfg=dict(
        _delete_=True,
        type='Pretrained',
        checkpoint='https://download.openmmlab.com/mmyolo/v0/yolox/'
        'yolox_s_fast_8xb32-300e-rtmdet-hyp_coco/yolox_s_fast_'
        '8xb32-300e-rtmdet-hyp_coco_20230210_134645-3a8dfbd7.pth'),
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        pad_size_divisor=32,
        batch_augments=[
            dict(
                type='PoseBatchSyncRandomResize',
                random_size_range=(480, 800),
                size_divisor=32,
                interval=1)
        ]),
    bbox_head=dict(
        type='YOLOXPoseHead',
        head_module=dict(
            type='YOLOXPoseHeadModule',
            num_classes=1,
            num_keypoints=17,
        ),
        loss_pose=dict(
            type='OksLoss',
            metainfo='configs/_base_/datasets/coco.py',
            loss_weight=30.0)),
    train_cfg=dict(
        assigner=dict(
            type='PoseSimOTAAssigner',
            center_radius=2.5,
            iou_calculator=dict(type='mmdet.BboxOverlaps2D'),
            oks_calculator=dict(
                type='OksLoss', metainfo='configs/_base_/datasets/coco.py'))),
    test_cfg=dict(
        yolox_style=True,
        multi_label=False,
        score_thr=0.2,
        max_per_img=300,
        nms=dict(type='nms', iou_threshold=0.65)))

# pipelines
pre_transform = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='LoadAnnotations', with_bbox=True, with_keypoints=True)
    # dict(type='PoseToDetConverter')
]

img_scale = _base_.img_scale

train_pipeline_stage1 = [
    *pre_transform,
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.75, 1.0),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='YOLOXMixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='FilterAnnotations', by_keypoints=True, keep_empty=False),
    dict(
        type='PackDetPoseInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape'))
]

train_pipeline_stage2 = [
    *pre_transform,
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='mmdet.Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='FilterAnnotations', by_keypoints=True, keep_empty=False),
    dict(type='PackDetPoseInputs')
]

test_pipeline = [
    *pre_transform,
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='mmdet.Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(
        type='PackDetPoseInputs',
        meta_keys=('id', 'img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip_indices'))
]

# dataset settings
dataset_type = 'CocoPoseDataset'
data_mode = 'bottomup'
data_root = 'data/coco/'

train_dataloader = dict(
    _delete_=True,
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_mode=data_mode,
        data_root=data_root,
        ann_file='annotations/person_keypoints_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline_stage1))

val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_mode=data_mode,
        data_root=data_root,
        ann_file='annotations/person_keypoints_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline))

test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(
    _delete_=True,
    type='mmpose.CocoMetric',
    ann_file=data_root + 'annotations/person_keypoints_val2017.json',
)
test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(save_best='coco/AP', rule='greater'),
    visualization=dict(type='mmdet.DetVisualizationHook'))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='mmpose.PoseLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

# optimizer
base_lr = 0.004
max_epochs = 300
num_last_epochs = 20
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(_delete_=True, type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        new_train_pipeline=train_pipeline_stage2,
        priority=48),
    dict(type='mmdet.SyncNormHook', priority=48),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49)
]

auto_scale_lr = dict(base_batch_size=256)
