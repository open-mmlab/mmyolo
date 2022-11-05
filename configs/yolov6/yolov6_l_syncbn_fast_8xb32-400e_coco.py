_base_ = './yolov6_m_syncbn_fast_8xb32-400e_coco.py'

deepen_factor = 1
widen_factor = 1

model = dict(
    backbone=dict(
        type='YOLOv6CSPBep',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        expansion=1. / 2,
        block_cfg=dict(
            type='ConvWrapper',
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001)),
        csp_act_cfg=dict(type='SiLU', inplace=True)),
    neck=dict(
        type='YOLOv6CSPRepPAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        expansion=1. / 2,
        block_cfg=dict(
            type='ConvWrapper',
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001)),
        csp_act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='YOLOv6Head',
        head_module=dict(widen_factor=widen_factor),
    ))
