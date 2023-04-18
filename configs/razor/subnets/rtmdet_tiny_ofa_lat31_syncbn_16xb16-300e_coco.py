_base_ = [
    'mmrazor::_base_/nas_backbones/ofa_mobilenetv3_supernet.py',
    '../../rtmdet/rtmdet_s_syncbn_fast_8xb32-300e_coco.py'
]

checkpoint_file = 'https://download.openmmlab.com/mmrazor/v1/ofa/ofa_mobilenet_subnet_8xb256_in1k_note8_lat%4031ms_top1%4072.8_finetune%4025.py_20221214_0939-981a8b2a.pth'  # noqa
fix_subnet = 'https://download.openmmlab.com/mmrazor/v1/yolo_nas_backbone/OFA_SUBNET_NOTE8_LAT31.yaml'  # noqa
deepen_factor = 0.167
widen_factor = 1.0
channels = [40, 112, 160]
train_batch_size_per_gpu = 16
img_scale = (960, 960)

_base_.nas_backbone.out_indices = (2, 4, 5)
_base_.nas_backbone.conv_cfg = dict(type='mmrazor.OFAConv2d')
_base_.nas_backbone.init_cfg = dict(
    type='Pretrained',
    checkpoint=checkpoint_file,
    prefix='architecture.backbone.')
nas_backbone = dict(
    type='mmrazor.sub_model',
    fix_subnet=fix_subnet,
    cfg=_base_.nas_backbone,
    extra_prefix='backbone.')

_base_.model.backbone = nas_backbone
_base_.model.neck.widen_factor = widen_factor
_base_.model.neck.deepen_factor = deepen_factor
_base_.model.neck.in_channels = channels
_base_.model.neck.out_channels = channels[0]
_base_.model.bbox_head.head_module.in_channels = channels[0]
_base_.model.bbox_head.head_module.feat_channels = channels[0]
_base_.model.bbox_head.head_module.widen_factor = widen_factor

_base_.model.test_cfg = dict(
    multi_label=True,
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_threshold=0.6),
    max_per_img=100)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Mosaic',
        img_scale=img_scale,
        use_cached=True,
        max_cached_images=20,
        random_pop=False,
        pad_val=114.0),
    dict(
        type='mmdet.RandomResize',
        scale=(1280, 1280),
        ratio_range=(0.5, 2.0),  # note
        resize_type='mmdet.Resize',
        keep_ratio=True),
    dict(type='mmdet.RandomCrop', crop_size=img_scale),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(
        type='YOLOXMixUp',
        img_scale=(960, 960),
        ratio_range=(1.0, 1.0),
        max_cached_images=10,
        use_cached=True,
        random_pop=False,
        pad_val=(114, 114, 114),
        prob=0.5),
    dict(type='mmdet.PackDetInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='mmdet.RandomResize',
        scale=img_scale,
        ratio_range=(0.5, 2.0),  # note
        resize_type='mmdet.Resize',
        keep_ratio=True),
    dict(type='mmdet.RandomCrop', crop_size=img_scale),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='mmdet.PackDetInputs')
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu, dataset=dict(pipeline=train_pipeline))

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='mmdet.Resize', scale=(960, 960), keep_ratio=True),
    dict(type='mmdet.Pad', size=(960, 960), pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_dataloader = dict(
    dataset=dict(pipeline=test_pipeline, batch_shapes_cfg=None))

test_dataloader = val_dataloader

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=_base_.max_epochs - _base_.num_epochs_stage2,
        switch_pipeline=train_pipeline_stage2)
]

find_unused_parameters = True
