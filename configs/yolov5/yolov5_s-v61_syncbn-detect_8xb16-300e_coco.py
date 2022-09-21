_base_ = 'yolov5_s-v61_syncbn_8xb16-300e_coco.py'

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args={{_base_.file_client_args}}),
    dict(
        type='LetterResize',
        scale=_base_.img_scale,
        allow_scale_up=True,
        use_mini_pad=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]

val_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline,
        batch_shapes_cfg=dict(
            type='BatchShapePolicy',
            batch_size=_base_.val_batch_size_per_gpu,
            img_size=_base_.img_scale[0],
            size_divisor=32,
            extra_pad_ratio=0.0)))
test_dataloader = val_dataloader

model = dict(
    test_cfg=dict(
        multi_label=False, score_thr=0.25, nms=dict(iou_threshold=0.45)))
