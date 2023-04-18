_base_ = '../yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'

# dataset settings
data_root = 'data/VOCdevkit/'
dataset_type = 'YOLOv5VOCDataset'

# parameters that often need to be modified
num_classes = 20
img_scale = (512, 512)  # width, height
max_epochs = 50
train_batch_size_per_gpu = 64
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
num_det_layers = 3

load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth'  # noqa

tta_img_scales = [img_scale, (416, 416), (640, 640)]

# Hyperparameter reference from:
# https://github.com/ultralytics/yolov5/blob/master/data/hyps/hyp.VOC.yaml
model = dict(
    bbox_head=dict(
        head_module=dict(num_classes=num_classes),
        prior_generator=dict(base_sizes=anchors),
        loss_cls=dict(
            loss_weight=0.21638 * (num_classes / 80 * 3 / num_det_layers),
            class_weight=0.5),
        loss_bbox=dict(loss_weight=0.02 * (3 / num_det_layers)),
        loss_obj=dict(
            loss_weight=0.51728 *
            ((img_scale[0] / 640)**2 * 3 / num_det_layers),
            class_weight=0.67198),
        # Different from COCO
        prior_match_thr=3.3744),
    test_cfg=dict(nms=dict(iou_threshold=0.6)))

albu_train_transforms = _base_.albu_train_transforms
pre_transform = _base_.pre_transform

with_mosiac_pipeline = [
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
        # img_scale is (width, height)
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
                pre_transform=pre_transform),
            dict(
                type='YOLOv5RandomAffine',
                max_rotate_degree=0.0,
                max_translate_ratio=0.04591,
                max_shear_degree=0.0,
                scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
                # img_scale is (width, height)
                border=(-img_scale[0] // 2, -img_scale[1] // 2),
                border_val=(114, 114, 114))
        ])
]

without_mosaic_pipeline = [
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

# Because the border parameter is inconsistent when
# using mosaic or not, `RandomChoice` is used here.
randchoice_mosaic_pipeline = dict(
    type='RandomChoice',
    transforms=[with_mosiac_pipeline, without_mosaic_pipeline],
    prob=[0.85834, 0.14166])

train_pipeline = [
    *pre_transform, randchoice_mosaic_pipeline,
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
    _delete_=True,
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
        ],
        # Use ignore_keys to avoid judging metainfo is
        # not equal in `ConcatDataset`.
        ignore_keys='dataset_type'),
    collate_fn=dict(type='yolov5_collate'))

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
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

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        # To load COCO pretrained model, need to set `strict_load=False`
        strict_load=False,
        priority=49)
]

# TODO: Support using coco metric in voc dataset
val_evaluator = dict(
    _delete_=True, type='mmdet.VOCMetric', metric='mAP', eval_mode='area')

test_evaluator = val_evaluator

train_cfg = dict(max_epochs=max_epochs)

# Config for Test Time Augmentation. (TTA)
_multiscale_resize_transforms = [
    dict(
        type='Compose',
        transforms=[
            dict(type='YOLOv5KeepRatioResize', scale=s),
            dict(
                type='LetterResize',
                scale=s,
                allow_scale_up=False,
                pad_val=dict(img=114))
        ]) for s in tta_img_scales
]

tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(
        type='TestTimeAug',
        transforms=[
            _multiscale_resize_transforms,
            [
                dict(type='mmdet.RandomFlip', prob=1.),
                dict(type='mmdet.RandomFlip', prob=0.)
            ], [dict(type='mmdet.LoadAnnotations', with_bbox=True)],
            [
                dict(
                    type='mmdet.PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                               'scale_factor', 'pad_param', 'flip',
                               'flip_direction'))
            ]
        ])
]
