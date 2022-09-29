_base_ = '../yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'

# dataset settings
data_root = 'data/VOCdevkit/'
dataset_type = 'YOLOv5VOCDataset'

# parameters that often need to be modified
img_scale = (512, 512)
max_epochs = 50
train_batch_size_per_gpu = 8
train_num_workers = 8
val_batch_size_per_gpu = 1
val_num_workers = 2

# persistent_workers must be False if num_workers is 0.
persistent_workers = True

lr_factor = 0.15135
affine_scale = 0.75544

# only on Val
batch_shapes_cfg = dict(img_size=img_scale[0])

anchors = [[(26, 44), (67, 57), (61, 130)], [(121, 118), (120, 239),
                                             (206, 182)],
           [(376, 161), (234, 324), (428, 322)]]

model = dict(
    bbox_head=dict(
        head_module=dict(num_classes=20),
        prior_generator=dict(base_sizes=anchors),
        loss_cls=dict(loss_weight=0.21638, class_weight=0.5),
        loss_bbox=dict(loss_weight=0.02),
        loss_obj=dict(loss_weight=0.51728, class_weight=0.67198),
        prior_match_thr=3.3744,
        obj_level_weights=[4., 1., 0.4]),
    test_cfg=dict(nms=dict(iou_threshold=0.6)))

albu_train_transforms = _base_.albu_train_transforms
pre_transform = _base_.pre_transform

mosaic_affine_pipeline = [
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_translate_ratio=0.04591,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114)),
    dict(
        type='YOLOv5MixUp',
        prob=0.04266,
        pre_transform=[
            *pre_transform,
            dict(
                type='Mosaic',
                img_scale=img_scale,
                pad_val=114.0,
                pre_transform=pre_transform,
                prob=0.85834),
            dict(
                type='YOLOv5RandomAffine',
                max_rotate_degree=0.0,
                max_translate_ratio=0.04591,
                max_shear_degree=0.0,
                scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
                border=(-img_scale[0] // 2, -img_scale[1] // 2),
                border_val=(114, 114, 114)),
        ])
]

no_mosaic_affine_pipeline = [
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_translate_ratio=0.04591,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        border=(0, 0),
        border_val=(114, 114, 114)),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114))
]

mosaic_prob_config = dict(
    type='RandomTransform',
    transform_list=[mosaic_affine_pipeline, no_mosaic_affine_pipeline],
    prob_list=[
        0.85834,
    ])

# enable mixup
train_pipeline = [
    *pre_transform, mosaic_prob_config,
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
    dict(
        type='YOLOv5HSVRandomAug',
        hue_delta=0.01041,
        saturation_delta=0.54703,
        value_delta=0.27739),
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
        type='ConcatDataset',
        datasets=[
            dict(
                type=dataset_type,
                data_root=data_root,
                ann_file='VOC2007/ImageSets/Main/trainval.txt',
                data_prefix=dict(sub_data_root='VOC2007/'),
                filter_cfg=dict(filter_empty_gt=False, min_size=32),
                pipeline=train_pipeline),
            dict(
                type=dataset_type,
                data_root=data_root,
                ann_file='VOC2012/ImageSets/Main/trainval.txt',
                data_prefix=dict(sub_data_root='VOC2012/'),
                filter_cfg=dict(filter_empty_gt=False, min_size=32),
                pipeline=train_pipeline)
        ]),
    collate_fn=dict(type='yolov5_collate'),
    _delete_=True)

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args={{_base_.file_client_args}}),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    # need gt_ignore
    dict(type='mmdet.LoadAnnotations', with_bbox=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
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
        ann_file='VOC2007/ImageSets/Main/test.txt',
        data_prefix=dict(sub_data_root='VOC2007/'),
        test_mode=True,
        pipeline=test_pipeline,
        batch_shapes_cfg=batch_shapes_cfg))

test_dataloader = val_dataloader

param_scheduler = None
optim_wrapper = dict(
    optimizer=dict(
        lr=0.00334,
        momentum=0.74832,
        weight_decay=0.00025,
        batch_size_per_gpu=train_batch_size_per_gpu))

default_hooks = dict(
    param_scheduler=dict(
        lr_factor=lr_factor,
        max_epochs=max_epochs,
        warmup_epochs=3.3835,
        warmup_momentum=0.59462,
        warmup_bias_lr=0.18657))

# TODO: support using coco metric in voc dataset
val_evaluator = dict(
    type='mmdet.VOCMetric', metric='mAP', eval_mode='11points', _delete_=True)

test_evaluator = val_evaluator

train_cfg = dict(max_epochs=max_epochs)
