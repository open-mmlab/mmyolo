# Training mode is currently not supported

_base_ = '../yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py'
max_epochs = 400
train_batch_size_per_gpu = 32

deepen_factor = _base_.deepen_factor
widen_factor = _base_.widen_factor
model = dict(
    backbone=dict(
        _delete_=True,
        type='YOLOv6EfficientRep',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='ReLU', inplace=True)),
    neck=dict(
        _delete_=True,
        type='YOLOv6RepPAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, 1024],
        out_channels=[128, 256, 512],
        num_csp_blocks=12,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='ReLU', inplace=True),
    ),
    bbox_head=dict(
        _delete_=True,
        type='YOLOv6Head',
        head_module=dict(
            type='YOLOv6HeadModule',
            num_classes=80,
            in_channels=[128, 256, 512],
            widen_factor=widen_factor,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True),
            featmap_strides=[8, 16, 32])),
    train_cfg=dict(
        initial_epoch=4,
        initial_assigner=dict(type='BatchATSSAssigner', topk=9, iou2d_calculator=dict(type='mmdet.BboxOverlaps2D'), num_classes=80),
        assigner=dict(type='BatchTaskAlignedAssigner', topk=13, alpha=1, beta=6),
        ))
