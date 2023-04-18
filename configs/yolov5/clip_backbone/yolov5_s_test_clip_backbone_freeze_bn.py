_base_ = '../yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'

model = dict(
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        # 按照clip里的预处理方式
        mean=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255],
        std=[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255]),
    backbone=dict(
        _delete_=True,
        type='CLIPModifiedResNet',
        freeze_backbone=False,  # 只冻结bn，不冻结backbone
        freeze_bn=True,
        output_dim=1024,
        layers=[3, 4, 6, 3],
        width=64,
        heads=64 * 32 // 64),
    neck=[
        dict(
            type='TempCLIPdownsampleneck',
            in_channels=[512, 1024, 2048],
            output_channels=[128, 256, 512],
            norm_cfg=_base_.norm_cfg,
            act_cfg=dict(type='SiLU', inplace=True)),
        dict(
            type='YOLOv5PAFPN',
            deepen_factor=_base_.deepen_factor,
            widen_factor=_base_.widen_factor,
            in_channels=[256, 512, 1024],
            out_channels=[256, 512, 1024],
            num_csp_blocks=3,
            norm_cfg=_base_.norm_cfg,
            act_cfg=dict(type='SiLU', inplace=True))
    ])

base_lr = 0.004
weight_decay = 0.05
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=weight_decay),
    paramwise_cfg=dict(
        custom_keys=dict(backbone=dict(lr_mult=0.1, decay_mult=0.1)),
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True))

save_checkpoint_intervals = 10
max_keep_ckpts = 3
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_checkpoint_intervals,
        max_keep_ckpts=max_keep_ckpts  # only keep latest 3 checkpoints
    ))

load_from = 'CLIPResNet50.pth'
