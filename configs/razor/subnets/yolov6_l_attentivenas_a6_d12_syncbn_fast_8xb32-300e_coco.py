_base_ = [
    'mmrazor::_base_/nas_backbones/attentive_mobilenetv3_supernet.py',
    '../../yolov6/yolov6_l_syncbn_fast_8xb32-300e_coco.py'
]

checkpoint_file = 'https://download.openmmlab.com/mmrazor/v1/bignas/attentive_mobilenet_subnet_8xb256_in1k_flops-0.93G_acc-80.81_20221229_200440-73d92cc6.pth'  # noqa
fix_subnet = 'https://download.openmmlab.com/mmrazor/v1/bignas/ATTENTIVE_SUBNET_A6.yaml'  # noqa
deepen_factor = 1.2
widen_factor = 1
channels = [40, 128, 224]
mid_channels = [40, 128, 224]

_base_.train_dataloader.batch_size = 16
_base_.nas_backbone.out_indices = (2, 4, 6)
_base_.nas_backbone.conv_cfg = dict(type='mmrazor.BigNasConv2d')
_base_.nas_backbone.norm_cfg = dict(type='mmrazor.DynamicBatchNorm2d')
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
_base_.model.neck.out_channels = mid_channels
_base_.model.bbox_head.head_module.in_channels = mid_channels
_base_.model.bbox_head.head_module.widen_factor = widen_factor

find_unused_parameters = True
