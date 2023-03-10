_base_ = 'yolov5_s-v61_syncbn_8xb16-300e_coco.py'

model = dict(
    type='YOLODetector',
    bbox_head=dict(
        type='YOLOv5InsHead',
        head_module=dict(
            type='YOLOv5InsHeadModule',
            mask_channels=32,
            proto_channels=256,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True),
            num_classes=_base_.num_classes,
            in_channels=[256, 512, 1024],
            widen_factor=_base_.widen_factor,
            featmap_strides=_base_.strides,
            num_base_priors=3)))

train_pipeline = [
    *_base_.pre_transform,
    dict(
        type='Mosaic',
        img_scale=_base_.img_scale,
        pad_val=114.0,
        pre_transform=_base_.pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(0.5, 1.5),
        border=(-_base_.img_scale[0] // 2, -_base_.img_scale[1] // 2),
        border_val=(114, 114, 114),
        min_area_ratio=0.01,
        max_aspect_ratio=100,
        use_mask_refine=True),
    dict(type='Polygon2Mask', downsample_ratio=4, mask_overlap=False),
    dict(
        type='Albu',
        transforms=_base_.albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes',
            'gt_masks': 'masks'
        }),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='YOLOv5KeepRatioResize', scale=_base_.img_scale),
    dict(
        type='LetterResize',
        scale=_base_.img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False,
        _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(metric=['bbox', 'segm'])
test_evaluator = val_evaluator
