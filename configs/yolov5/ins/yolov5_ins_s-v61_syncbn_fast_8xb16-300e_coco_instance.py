_base_ = '../mask_refine/yolov5_s_mask-refine-v61_syncbn_fast_8xb16-300e_coco.py'  # noqa

# Batch size of a single GPU during validation
val_batch_size_per_gpu = 2
# Worker to pre-fetch data for each single GPU during validation
val_num_workers = 2

batch_shapes_cfg = dict(
    _delete_=True,
    type='BatchShapePolicy',
    batch_size=val_batch_size_per_gpu,
    img_size=_base_.img_scale[0],
    # The image scale of padding should be divided by pad_size_divisor
    size_divisor=32,
    # Additional paddings for pixel scale
    extra_pad_ratio=0.5)

# Testing take a long time due to model_test_cfg.
# If you want to speed it up, you can increase score_thr
# or decraese nms_pre and max_per_img
model_test_cfg = dict(
    multi_label=True,
    nms_pre=30000,
    min_bbox_size=0,
    score_thr=0.001,
    nms=dict(type='nms', iou_threshold=0.6),
    max_per_img=300,
    mask_thr_binary=0.5,
    # fast_test: Whether to use fast test methods. When set
    # to False, the implementation here is the same as the
    # official, with higher mAP. If set to True, mask will first
    # be upsampled to origin image shape through Pytorch, and
    # then use mask_thr_binary to determine which pixels belong
    # to the object. If set to False, will first use
    # mask_thr_binary to determine which pixels belong to the
    # object , and then use opencv to upsample mask to origin
    # image shape. Default to False.
    fast_test=False)
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
            num_base_priors=3),
        loss_mask=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=0.05,
            reduction='none')),
    test_cfg=model_test_cfg)

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
        scaling_ratio_range=(1 - _base_.affine_scale, 1 + _base_.affine_scale),
        border=(-_base_.img_scale[0] // 2, -_base_.img_scale[1] // 2),
        border_val=(114, 114, 114),
        min_area_ratio=_base_.min_area_ratio,
        max_aspect_ratio=100,
        use_mask_refine=True),
    dict(
        type='mmdet.Albu',
        transforms=_base_.albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes',
        }),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='Polygon2Mask',
        downsample_ratio=4,
        mask_overlap=True,
        coco_style=False),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    dataset=dict(batch_shapes_cfg=batch_shapes_cfg))
test_dataloader = val_dataloader

val_evaluator = dict(metric=['bbox', 'segm'])
test_evaluator = val_evaluator
