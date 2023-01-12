_base_ = [
    'mmrazor::_base_/nas_backbones/spos_shufflenet_supernet.py',
    'mmyolo::yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'
]

custom_imports = dict(imports=['mmrazor.models'], allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmrazor/v0.1/nas/spos/spos_shufflenetv2_subnet_8xb128_in1k/spos_shufflenetv2_subnet_8xb128_in1k_flops_0.33M_acc_73.87_20211222-1f0a0b4d.pth'  # noqa
fix_subnet = 'https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmrazor/v1/spos/yolov5/SPOS_SUBNET.yaml'  # noqa
widen_factor = 1.0
channels = [160, 320, 640]

nas_backbone = dict(
    _delete_=True,
    type='mmrazor.sub_model',
    fix_subnet=fix_subnet,
    cfg=dict(
        type='mmrazor.SearchableShuffleNetV2',
        out_indices=(1, 2, 3),
        arch_setting=_base_.arch_setting,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=checkpoint_file,
            prefix='architecture.model.backbone.')),
    extra_prefix='backbone.')

model = dict(
    backbone=nas_backbone,
    neck=dict(
        type='YOLOv5PAFPN',
        widen_factor=widen_factor,
        in_channels=channels,
        out_channels=channels),
    bbox_head=dict(
        type='YOLOv5Head',
        head_module=dict(
            type='YOLOv5HeadModule',
            in_channels=channels,
            widen_factor=widen_factor)))

find_unused_parameters = True
