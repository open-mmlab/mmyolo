_base_ = ['./yolov6_v3_n_syncbn_fast_8xb32-300e_widerface.py']

deepen_factor = 0.70
# The scaling factor that controls the width of the network structure
widen_factor = 0.50

model = dict(
    backbone=dict(
        type='YOLOv6CSPBep',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        block_cfg=dict(type='RepVGGBlock'),
        hidden_ratio=0.5,
        act_cfg=dict(type='ReLU', inplace=True)),
    neck=dict(
        type='YOLOv6CSPRepBiPAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        block_cfg=dict(type='RepVGGBlock'),
        block_act_cfg=dict(type='ReLU', inplace=True),
        hidden_ratio=0.5),
    bbox_head=dict(
        type='YOLOv6FaceHead',
        head_module=dict(stemout_channels=256, widen_factor=widen_factor),
        loss_bbox=dict(type='IoULoss', iou_mode='giou')))
