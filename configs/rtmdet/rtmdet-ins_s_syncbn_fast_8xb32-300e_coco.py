_base_ = './rtmdet_s_syncbn_fast_8xb32-300e_coco.py'

model = dict(
    bbox_head=dict(
        type='RTMDetInsSepBNHead',
        head_module=dict(
            # rewrite head_module cfg
            _delete_=True,
            type='RTMDetInsSepBNHeadModule',
            num_classes=_base_.num_classes,
            in_channels=256,
            feat_channels=256,
            stacked_convs=2,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='SiLU', inplace=True),
            share_conv=True,
            pred_kernel_size=1,
            use_sigmoid_cls=True,
            featmap_strides=[8, 16, 32],
            widen_factor=0.5),
        loss_mask=dict(
            type='mmdet.DiceLoss', loss_weight=2.0, eps=5e-6,
            reduction='mean')),
    test_cfg=dict(
        # rewrite test_cfg
        _delete_=True,
        multi_label=False,
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100,
        mask_thr_binary=0.5))

val_evaluator = dict(metric=['bbox', 'segm'])
test_evaluator = val_evaluator
