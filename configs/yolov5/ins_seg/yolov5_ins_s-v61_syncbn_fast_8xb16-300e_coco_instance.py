_base_ = '../yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'  # noqa

# ========================modified parameters======================
# YOLOv5RandomAffine
use_mask2refine = True
max_aspect_ratio = 100
min_area_ratio = 0.01
# Polygon2Mask
downsample_ratio = 4
mask_overlap = True
# LeterResize
# half_pad_param: if set to True, left and right pad_param will
# be given by dividing padding_h by 2. If set to False, pad_param is
# in int format. We recommend setting this to False for object
# detection tasks, and True for instance segmentation tasks.
# Default to False.
half_pad_param = True

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
    fast_test=True)

# ===============================Unmodified in most cases====================
model = dict(
    type='YOLODetector',
    bbox_head=dict(
        type='YOLOv5InsHead',
        head_module=dict(
            type='YOLOv5InsHeadModule', mask_channels=32, proto_channels=256),
        mask_overlap=mask_overlap,
        loss_mask=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=True, reduction='none'),
        loss_mask_weight=0.05),
    test_cfg=model_test_cfg)

pre_transform = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        mask2bbox=use_mask2refine)
]

train_pipeline = [
    *pre_transform,
    dict(
        type='Mosaic',
        img_scale=_base_.img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - _base_.affine_scale, 1 + _base_.affine_scale),
        border=(-_base_.img_scale[0] // 2, -_base_.img_scale[1] // 2),
        border_val=(114, 114, 114),
        min_area_ratio=min_area_ratio,
        max_aspect_ratio=max_aspect_ratio,
        use_mask_refine=use_mask2refine),
    # TODO: support mask transform in albu
    # Geometric transformations are not supported in albu now.
    dict(
        type='mmdet.Albu',
        transforms=_base_.albu_train_transforms,
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
        type='Polygon2Mask',
        downsample_ratio=downsample_ratio,
        mask_overlap=mask_overlap),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='YOLOv5KeepRatioResize', scale=_base_.img_scale),
    dict(
        type='LetterResize',
        scale=_base_.img_scale,
        allow_scale_up=False,
        half_pad_param=half_pad_param,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
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
