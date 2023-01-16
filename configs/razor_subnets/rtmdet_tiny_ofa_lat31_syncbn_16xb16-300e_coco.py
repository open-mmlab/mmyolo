_base_ = [
    'mmrazor::_base_/nas_backbones/ofa_mobilenetv3_supernet.py',
    '../rtmdet/rtmdet_s_syncbn_fast_8xb32-300e_coco.py'
]

checkpoint_file = 'https://download.openmmlab.com/mmrazor/v1/ofa/ofa_mobilenet_subnet_8xb256_in1k_note8_lat%4031ms_top1%4072.8_finetune%4025.py_20221214_0939-981a8b2a.pth'  # noqa
fix_subnet = 'https://download.openmmlab.com/mmrazor/v1/ofa/rtmdet/OFA_SUBNET_NOTE8_LAT31.yaml'  # noqa
deepen_factor = 0.167
widen_factor = 1.0
channels = [40, 112, 160]
train_batch_size_per_gpu = 16
img_scale = (960, 960)

_base_.base_lr = 0.002
_base_.optim_wrapper.optimizer.lr = 0.002
_base_.param_scheduler[1].eta_min = 0.002 * 0.05

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

_base_.train_dataloader.dataset.pipeline[2].img_scale = img_scale
_base_.train_dataloader.dataset.pipeline[3].scale = (img_scale[0] * 2,
                                                     img_scale[1] * 2)
_base_.train_dataloader.dataset.pipeline[4].crop_size = img_scale
_base_.train_dataloader.dataset.pipeline[7].size = img_scale
_base_.train_dataloader.batch_size = train_batch_size_per_gpu

_base_.custom_hooks[1].switch_pipeline[2].scale = img_scale
_base_.custom_hooks[1].switch_pipeline[3].crop_size = img_scale
_base_.custom_hooks[1].switch_pipeline[6].size = img_scale

_base_.val_dataloader.dataset.batch_shapes_cfg.img_size = img_scale[0]
_base_.val_dataloader.dataset.pipeline[1].scale = img_scale
_base_.val_dataloader.dataset.pipeline[2].scale = img_scale
_base_.test_dataloader = _base_.val_dataloader

find_unused_parameters = True
