_base_ = './yolov7_l_syncbn_fast_8x16b-300e_coco.py'

# ========================modified parameters========================
# -----data related-----
img_scale = (1280, 1280)  # height, width
num_classes = 80  # Number of classes for classification
# Config of batch shapes. Only on val
# It means not used if batch_shapes_cfg is None.
batch_shapes_cfg = dict(
    img_size=img_scale[
        0],  # The image scale of padding should be divided by pad_size_divisor
    size_divisor=64)  # Additional paddings for pixel scale
tta_img_scales = [(1280, 1280), (1024, 1024), (1536, 1536)]

# -----model related-----
# Basic size of multi-scale prior box
anchors = [
    [(19, 27), (44, 40), (38, 94)],  # P3/8
    [(96, 68), (86, 152), (180, 137)],  # P4/16
    [(140, 301), (303, 264), (238, 542)],  # P5/32
    [(436, 615), (739, 380), (925, 792)]  # P6/64
]
strides = [8, 16, 32, 64]  # Strides of multi-scale prior box
num_det_layers = 4  # # The number of model output scales
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)

# Data augmentation
max_translate_ratio = 0.2  # YOLOv5RandomAffine
scaling_ratio_range = (0.1, 2.0)  # YOLOv5RandomAffine
mixup_prob = 0.15  # YOLOv5MixUp
randchoice_mosaic_prob = [0.8, 0.2]
mixup_alpha = 8.0  # YOLOv5MixUp
mixup_beta = 8.0  # YOLOv5MixUp

# -----train val related-----
loss_cls_weight = 0.3
loss_bbox_weight = 0.05
loss_obj_weight = 0.7
obj_level_weights = [4.0, 1.0, 0.25, 0.06]
simota_candidate_topk = 20

# The only difference between P6 and P5 in terms of
# hyperparameters is lr_factor
lr_factor = 0.2

# ===============================Unmodified in most cases====================
pre_transform = _base_.pre_transform

model = dict(
    backbone=dict(arch='W', out_indices=(2, 3, 4, 5)),
    neck=dict(
        in_channels=[256, 512, 768, 1024],
        out_channels=[128, 256, 384, 512],
        use_maxpool_in_downsample=False,
        use_repconv_outs=False),
    bbox_head=dict(
        head_module=dict(
            type='YOLOv7p6HeadModule',
            in_channels=[128, 256, 384, 512],
            featmap_strides=strides,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='SiLU', inplace=True)),
        prior_generator=dict(base_sizes=anchors, strides=strides),
        simota_candidate_topk=simota_candidate_topk,  # note
        # scaled based on number of detection layers
        loss_cls=dict(loss_weight=loss_cls_weight *
                      (num_classes / 80 * 3 / num_det_layers)),
        loss_bbox=dict(loss_weight=loss_bbox_weight * (3 / num_det_layers)),
        loss_obj=dict(loss_weight=loss_obj_weight *
                      ((img_scale[0] / 640)**2 * 3 / num_det_layers)),
        obj_level_weights=obj_level_weights))

mosiac4_pipeline = [
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_translate_ratio=max_translate_ratio,  # note
        scaling_ratio_range=scaling_ratio_range,  # note
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114)),
]

mosiac9_pipeline = [
    dict(
        type='Mosaic9',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_translate_ratio=max_translate_ratio,  # note
        scaling_ratio_range=scaling_ratio_range,  # note
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114)),
]

randchoice_mosaic_pipeline = dict(
    type='RandomChoice',
    transforms=[mosiac4_pipeline, mosiac9_pipeline],
    prob=randchoice_mosaic_prob)

train_pipeline = [
    *pre_transform,
    randchoice_mosaic_pipeline,
    dict(
        type='YOLOv5MixUp',
        alpha=mixup_alpha,  # note
        beta=mixup_beta,  # note
        prob=mixup_prob,
        pre_transform=[*pre_transform, randchoice_mosaic_pipeline]),
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

default_hooks = dict(param_scheduler=dict(lr_factor=lr_factor))

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
