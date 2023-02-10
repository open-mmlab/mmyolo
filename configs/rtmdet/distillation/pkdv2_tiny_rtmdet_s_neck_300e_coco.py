_base_ = '../rtmdet_tiny_syncbn_fast_8xb32-300e_coco.py'

teacher_ckpt = 'https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_s_syncbn_fast_8xb32-300e_coco/rtmdet_s_syncbn_fast_8xb32-300e_coco_20221230_182329-0a8c901a.pth'  # noqa: E501
model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='FpnTeacherDistill',
    architecture=dict(
        cfg_path='mmyolo::rtmdet/rtmdet_tiny_syncbn_fast_8xb32-300e_coco.py'),
    teacher=dict(
        cfg_path='mmyolo::rtmdet/rtmdet_s_syncbn_fast_8xb32-300e_coco.py'),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            fpn0=dict(type='ModuleOutputs', source='neck.out_layers.0.conv'),
            fpn1=dict(type='ModuleOutputs', source='neck.out_layers.1.conv'),
            fpn2=dict(type='ModuleOutputs', source='neck.out_layers.2.conv'),
        ),
        teacher_recorders=dict(
            fpn0=dict(type='ModuleOutputs', source='neck.out_layers.0.conv'),
            fpn1=dict(type='ModuleOutputs', source='neck.out_layers.1.conv'),
            fpn2=dict(type='ModuleOutputs', source='neck.out_layers.2.conv')),
        connectors=dict(
            fpn0=dict(
                type='ConvModuleConnector',
                in_channel=96,
                out_channel=128,
                bias=False,
                act_cfg=None),
            fpn1=dict(
                type='ConvModuleConnector',
                in_channel=96,
                out_channel=128,
                bias=False,
                act_cfg=None),
            fpn2=dict(
                type='ConvModuleConnector',
                in_channel=96,
                out_channel=128,
                bias=False,
                act_cfg=None)),
        distill_losses=dict(
            loss_pkd_fpn0=dict(type='PKDV2', loss_weight=1),
            loss_pkd_fpn1=dict(type='PKDV2', loss_weight=1),
            loss_pkd_fpn2=dict(type='PKDV2', loss_weight=1)),
        loss_forward_mappings=dict(
            loss_pkd_fpn0=dict(
                preds_S=dict(
                    from_student=True, recorder='fpn0', connector='fpn0'),
                preds_T=dict(from_student=False, recorder='fpn0')),
            loss_pkd_fpn1=dict(
                preds_S=dict(
                    from_student=True, recorder='fpn1', connector='fpn1'),
                preds_T=dict(from_student=False, recorder='fpn1')),
            loss_pkd_fpn2=dict(
                preds_S=dict(
                    from_student=True, recorder='fpn2', connector='fpn2'),
                preds_T=dict(from_student=False, recorder='fpn2')))))

find_unused_parameters = True

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
        switch_pipeline=_base_.train_pipeline_stage2),
    dict(type='mmrazor.DistillationLossDetachHook')
]
