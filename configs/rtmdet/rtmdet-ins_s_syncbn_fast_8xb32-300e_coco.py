_base_ = './rtmdet_s_syncbn_fast_8xb32-300e_coco.py'

widen_factor = 0.5

model = dict(
    bbox_head=dict(
        type='RTMDetInsSepBNHead',
        head_module=dict(
            type='RTMDetInsSepBNHeadModule',
            use_sigmoid_cls=True,
            widen_factor=widen_factor),
        loss_mask=dict(
            type='mmdet.DiceLoss', loss_weight=2.0, eps=5e-6,
            reduction='mean')),
    test_cfg=dict(
        multi_label=True,
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100,
        mask_thr_binary=0.5))

_base_.test_pipeline[-2] = dict(
    type='LoadAnnotations', with_bbox=True, with_mask=True, _scope_='mmdet')

val_dataloader = dict(dataset=dict(pipeline=_base_.test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(metric=['bbox', 'segm'])
test_evaluator = val_evaluator
