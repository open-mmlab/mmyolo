_base_ = 'yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'

# ========================modified parameters======================
img_scale = (1280, 1280)  # width, height
num_classes = 80  # Number of classes for classification
# Config of batch shapes. Only on val.
# It means not used if batch_shapes_cfg is None.
batch_shapes_cfg = dict(
    img_size=img_scale[0],
    # The image scale of padding should be divided by pad_size_divisor
    size_divisor=64)
# Basic size of multi-scale prior box
anchors = [
    [(19, 27), (44, 40), (38, 94)],  # P3/8
    [(96, 68), (86, 152), (180, 137)],  # P4/16
    [(140, 301), (303, 264), (238, 542)],  # P5/32
    [(436, 615), (739, 380), (925, 792)]  # P6/64
]
# Strides of multi-scale prior box
strides = [8, 16, 32, 64]
num_det_layers = 4  # The number of model output scales
loss_cls_weight = 0.5
loss_bbox_weight = 0.05
loss_obj_weight = 1.0
# The obj loss weights of the three output layers
obj_level_weights = [4.0, 1.0, 0.25, 0.06]
affine_scale = 0.5  # YOLOv5RandomAffine scaling ratio

tta_img_scales = [(1280, 1280), (1024, 1024), (1536, 1536)]
# =======================Unmodified in most cases==================
model = dict(
    backbone=dict(arch='P6', out_indices=(2, 3, 4, 5)),
    neck=dict(
        in_channels=[256, 512, 768, 1024], out_channels=[256, 512, 768, 1024]),
    bbox_head=dict(
        head_module=dict(
            in_channels=[256, 512, 768, 1024], featmap_strides=strides),
        prior_generator=dict(base_sizes=anchors, strides=strides),
        # scaled based on number of detection layers
        loss_cls=dict(loss_weight=loss_cls_weight *
                      (num_classes / 80 * 3 / num_det_layers)),
        loss_bbox=dict(loss_weight=loss_bbox_weight * (3 / num_det_layers)),
        loss_obj=dict(loss_weight=loss_obj_weight *
                      ((img_scale[0] / 640)**2 * 3 / num_det_layers)),
        obj_level_weights=obj_level_weights))

pre_transform = _base_.pre_transform
albu_train_transforms = _base_.albu_train_transforms

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
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        # img_scale is (width, height)
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

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

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
    dataset=dict(pipeline=test_pipeline, batch_shapes_cfg=batch_shapes_cfg))

test_dataloader = val_dataloader

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
