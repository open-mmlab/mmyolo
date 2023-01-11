_base_ = [
    'mmrazor::_base_/nas_backbones/attentive_mobilenetv3_supernet.py',
    'mmyolo::yolov6/yolov6_l_syncbn_fast_8xb32-300e_coco.py'
]

custom_imports = dict(imports=['mmrazor.models'], allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmrazor/v1/bignas/attentive_mobilenet_subnet_8xb256_in1k_flops-0.93G_acc-80.81_20221229_200440-73d92cc6.pth'
deepen_factor = 1.2
widen_factor = 1
channels = [40, 128, 224]
mid_channels = [40, 128, 224]

train_dataloader = dict(
    batch_size=16
)

nas_backbone = dict(
    _delete_=True,
    type='mmrazor.sub_model',
    fix_subnet='configs/razor_subnets/ATTENTIVE_SUBNET_A6.yaml',
    cfg=dict(
        type='mmrazor.AttentiveMobileNetV3',
        out_indices=(2, 4, 6),
        arch_setting=_base_.arch_setting,
        conv_cfg=dict(type='mmrazor.BigNasConv2d'),
        norm_cfg=dict(type='mmrazor.DynamicBatchNorm2d'),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=checkpoint_file,
            prefix='architecture.backbone.')),
    extra_prefix='backbone.')

model = dict(
    backbone=nas_backbone,
    neck=dict(
        in_channels=channels,
        out_channels=mid_channels,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        hidden_ratio=1. / 2,
        block_cfg=dict(
            type='ConvWrapper',
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001)),
        block_act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        head_module=dict(in_channels=mid_channels, widen_factor=widen_factor)))

find_unused_parameters=True
