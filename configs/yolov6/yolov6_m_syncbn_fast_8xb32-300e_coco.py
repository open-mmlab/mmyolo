_base_ = './yolov6_s_syncbn_fast_8xb32-300e_coco.py'

# ======================= Possible modified parameters =======================
# -----model related-----
# The scaling factor that controls the depth of the network structure
deepen_factor = 0.6
# The scaling factor that controls the width of the network structure
widen_factor = 0.75

# -----train val related-----
affine_scale = 0.9  # YOLOv5RandomAffine scaling ratio

# ============================== Unmodified in most cases ===================
model = dict(
    backbone=dict(
        type='YOLOv6CSPBep',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        hidden_ratio=2. / 3,
        block_cfg=dict(type='RepVGGBlock'),
        act_cfg=dict(type='ReLU', inplace=True)),
    neck=dict(
        type='YOLOv6CSPRepPAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        block_cfg=dict(type='RepVGGBlock'),
        hidden_ratio=2. / 3,
        block_act_cfg=dict(type='ReLU', inplace=True)),
    bbox_head=dict(
        type='YOLOv6Head', head_module=dict(widen_factor=widen_factor)))

mosaic_affine_pipeline = [
    dict(
        type='Mosaic',
        img_scale=_base_.img_scale,
        pad_val=114.0,
        pre_transform=_base_.pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        # img_scale is (width, height)
        border=(-_base_.img_scale[0] // 2, -_base_.img_scale[1] // 2),
        border_val=(114, 114, 114))
]

train_pipeline = [
    *_base_.pre_transform, *mosaic_affine_pipeline,
    dict(
        type='YOLOv5MixUp',
        prob=0.1,
        pre_transform=[*_base_.pre_transform, *mosaic_affine_pipeline]),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
