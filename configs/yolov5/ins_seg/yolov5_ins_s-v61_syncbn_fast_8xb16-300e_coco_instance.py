_base_ = '../yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'  # noqa

# ========================modified parameters======================
# YOLOv5RandomAffine
use_mask2refine = True
max_aspect_ratio = 100
min_area_ratio = 0.01
# Polygon2Mask
downsample_ratio = 4
mask_overlap = True

# Batch size of per single GPU during validation
# The official yolov5 set the batchsize as 32 during validation,
# so we set the batchsize as 8(gpu)*4=32 to align with the official settings.
val_batch_size_per_gpu = 1
# Worker to pre-fetch data for each single GPU during validation
val_num_workers = 1

batch_shapes_cfg = dict(batch_size=val_batch_size_per_gpu)

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

# ===============================Unmodified in most cases====================
model = dict(
    type='YOLODetector',
    bbox_head=dict(
        type='YOLOv5InsHead',
        head_module=dict(
            type='YOLOv5InsHeadModule', mask_channels=32, proto_channels=256),
        mask_overlap=mask_overlap,
        loss_mask=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=0.05,
            reduction='none')),
    test_cfg=model_test_cfg)

pre_transform = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
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
        downsample_ratio=downsample_ratio,
        mask_overlap=mask_overlap,
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
