_base_ = '../_base_/default_runtime.py'

# dataset settings
data_root = 'data/coco25/'
dataset_type = 'mmdet.CocoDataset'

# parameters that often need to be modified
img_scale = (640, 640)  # height, width
max_epochs = 300
save_epoch_intervals = 10
train_batch_size_per_gpu = 2
train_num_workers = 2
val_batch_size_per_gpu = 25
val_num_workers = 2
base_lr = 0.01

# persistent_workers must be False if num_workers is 0.
persistent_workers = True

model = dict(
    type='DAMOYOLODetector',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[0., 0., 0.],
        std=[1., 1., 1.],
        bgr_to_rgb=True),
    backbone=dict(type='TinyNAS_res',
                  out_indices=(2, 4, 5),
                  with_spp=True,
                  use_focus=True,
                  act='relu',
                  reparam=True,
                  backbone_structure='s',
                  ),
    neck=dict(type='GiraffeNeckv2',
              depth=1.0,
              hidden_ratio=0.75,
              in_channels=[128, 256, 512],
              out_channels=[128, 256, 512],
              act='relu',
              spp=False,
              block_name='BasicBlock_3x3_Reverse',
              ),
    bbox_head=dict(type='ZeroHead',
                   num_classes=80,
                   in_channels=[128, 256, 512],
                   stacked_convs=0,
                   reg_max=16,
                   act='silu',
                   nms_conf_thre=0.05,
                   nms_iou_thre=0.7
                   ))

albu_train_transforms = [
    dict(type='Blur', p=0.01),
    dict(type='MedianBlur', p=0.01),
    dict(type='ToGray', p=0.01),
    dict(type='CLAHE', p=0.01)
]

pre_transform = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='LoadAnnotations', with_bbox=True)
]

train_pipeline = [
    *pre_transform,
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(0.5, 1.5),
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114)),
    dict(
        type='mmdet.Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        }),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
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
        pipeline=train_pipeline))

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=save_epoch_intervals)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=base_lr,
        momentum=0.937,
        weight_decay=0.0005,
        nesterov=True,
        batch_size_per_gpu=train_batch_size_per_gpu),
    constructor='YOLOv5OptimizerConstructor')

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root=data_root,
        test_mode=True,
        data_prefix=dict(img='val2017/'),
        ann_file='annotations/instances_val2017.json',
        pipeline=[
            dict(
                type='LoadImageFromFile',
                imdecode_backend = 'pillow',
                file_client_args=dict(backend='disk')),
            dict(type='DamoyoloResize',
                 width=img_scale[1],
                 height=img_scale[0],
                 keep_ratio=True,
                 interpolation='bilinear'),
            dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ],
        batch_shapes_cfg=dict(
            type='BatchShapePolicy',
            batch_size=val_batch_size_per_gpu,
            img_size=img_scale[0],
            size_divisor=32,
            extra_pad_ratio=0)))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox')
test_evaluator = val_evaluator

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
