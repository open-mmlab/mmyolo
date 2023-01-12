_base_ = [
    'mmrazor::_base_/nas_backbones/ofa_mobilenetv3_supernet.py',
    '../rtmdet/rtmdet_s_syncbn_fast_8xb32-300e_coco.py'
]

custom_imports = dict(imports=['mmrazor.models'], allow_failed_imports=False)
checkpoint_file = 'https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmrazor/v1/ofa/ofa_mobilenet_subnet_8xb256_in1k_note8_lat%4031ms_top1%4072.8_finetune%4025.py_20221214_0939-981a8b2a.pth'  # noqa
fix_subnet = 'https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmrazor/v1/ofa/rtmdet/OFA_SUBNET_NOTE8_LAT31.yaml'  # noqa
deepen_factor = 0.167
widen_factor = 1.0
channels = [40, 112, 160]
train_batch_size_per_gpu = 16
img_scale = (960, 960)

nas_backbone = dict(
    _delete_=True,
    type='mmrazor.sub_model',
    fix_subnet=fix_subnet,
    cfg=dict(
        type='mmrazor.AttentiveMobileNetV3',
        arch_setting=_base_.arch_setting,
        out_indices=(2, 4, 5),
        stride_list=[1, 2, 2, 2, 1, 2],
        with_se_list=[False, False, True, False, True, True],
        act_cfg_list=[
            'HSwish', 'ReLU', 'ReLU', 'ReLU', 'HSwish', 'HSwish', 'HSwish',
            'HSwish', 'HSwish'
        ],
        conv_cfg=dict(type='mmrazor.OFAConv2d'),
        norm_cfg=dict(type='mmrazor.DynamicBatchNorm2d', momentum=0.1),
        fine_grained_mode=True,
        with_attentive_shortcut=False),
    extra_prefix='backbone.')

model = dict(
    backbone=nas_backbone,
    neck=dict(
        in_channels=channels,
        out_channels=channels[0],
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    bbox_head=dict(
        head_module=dict(
            in_channels=channels[0],
            feat_channels=channels[0],
            widen_factor=widen_factor)))

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Mosaic',
        img_scale=img_scale,
        use_cached=True,
        max_cached_images=40,
        pad_val=114.0),
    dict(
        type='mmdet.RandomResize',
        # img_scale is (width, height)
        scale=(img_scale[0] * 2, img_scale[1] * 2),
        ratio_range=(0.5, 2.0),  # note
        resize_type='mmdet.Resize',
        keep_ratio=True),
    dict(type='mmdet.RandomCrop', crop_size=img_scale),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='YOLOv5MixUp', use_cached=True, max_cached_images=20),
    dict(type='mmdet.PackDetInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
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
        switch_epoch=_base_.max_epochs - _base_.stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
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

batch_shapes_cfg = dict(img_size=img_scale[0])

val_dataloader = dict(
    dataset=dict(batch_shapes_cfg=batch_shapes_cfg, pipeline=test_pipeline))

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
        switch_epoch=_base_.max_epochs - _base_.stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

find_unused_parameters = True
