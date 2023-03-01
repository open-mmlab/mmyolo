_base_ = '../rtmdet_m_syncbn_fast_8xb32-300e_coco.py'

teacher_ckpt = 'https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_l_syncbn_fast_8xb32-300e_coco/rtmdet_l_syncbn_fast_8xb32-300e_coco_20230102_135928-ee3abdc4.pth'  # noqa: E501

norm_cfg = dict(type='BN', affine=False, track_running_stats=False)

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='FpnTeacherDistill',
    architecture=dict(
        cfg_path='mmyolo::rtmdet/rtmdet_m_syncbn_fast_8xb32-300e_coco.py'),
    teacher=dict(
        cfg_path='mmyolo::rtmdet/rtmdet_l_syncbn_fast_8xb32-300e_coco.py'),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        # `recorders` are used to record various intermediate results during
        # the model forward.
        student_recorders=dict(
            fpn0=dict(type='ModuleOutputs', source='neck.out_layers.0.conv'),
            fpn1=dict(type='ModuleOutputs', source='neck.out_layers.1.conv'),
            fpn2=dict(type='ModuleOutputs', source='neck.out_layers.2.conv'),
        ),
        teacher_recorders=dict(
            fpn0=dict(type='ModuleOutputs', source='neck.out_layers.0.conv'),
            fpn1=dict(type='ModuleOutputs', source='neck.out_layers.1.conv'),
            fpn2=dict(type='ModuleOutputs', source='neck.out_layers.2.conv')),
        # `connectors` are adaptive layers which usually map teacher's and
        # students features to the same dimension.
        connectors=dict(
            fpn0_s=dict(
                type='ConvModuleConnector',
                in_channel=192,
                out_channel=256,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=None),
            fpn0_t=dict(
                type='NormConnector', in_channels=256, norm_cfg=norm_cfg),
            fpn1_s=dict(
                type='ConvModuleConnector',
                in_channel=192,
                out_channel=256,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=None),
            fpn1_t=dict(
                type='NormConnector', in_channels=256, norm_cfg=norm_cfg),
            fpn2_s=dict(
                type='ConvModuleConnector',
                in_channel=192,
                out_channel=256,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=None),
            fpn2_t=dict(
                type='NormConnector', in_channels=256, norm_cfg=norm_cfg)),
        distill_losses=dict(
            loss_fpn0=dict(type='ChannelWiseDivergence', loss_weight=1),
            loss_fpn1=dict(type='ChannelWiseDivergence', loss_weight=1),
            loss_fpn2=dict(type='ChannelWiseDivergence', loss_weight=1)),
        # `loss_forward_mappings` are mappings between distill loss forward
        # arguments and records.
        loss_forward_mappings=dict(
            loss_fpn0=dict(
                preds_S=dict(
                    from_student=True, recorder='fpn0', connector='fpn0_s'),
                preds_T=dict(
                    from_student=False, recorder='fpn0', connector='fpn0_t')),
            loss_fpn1=dict(
                preds_S=dict(
                    from_student=True, recorder='fpn1', connector='fpn1_s'),
                preds_T=dict(
                    from_student=False, recorder='fpn1', connector='fpn1_t')),
            loss_fpn2=dict(
                preds_S=dict(
                    from_student=True, recorder='fpn2', connector='fpn2_s'),
                preds_T=dict(
                    from_student=False, recorder='fpn2',
                    connector='fpn2_t')))))

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
        switch_epoch=_base_.max_epochs - _base_.num_epochs_stage2,
        switch_pipeline=_base_.train_pipeline_stage2),
    # stop distillation after the 280th epoch
    dict(type='mmrazor.StopDistillHook', stop_epoch=280)
]
