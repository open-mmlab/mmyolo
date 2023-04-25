_base_ = 'yolov5_s-v61_syncbn_8xb16-300e_coco.py'

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
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
    dataset=dict(pipeline=test_pipeline, batch_shapes_cfg=None))
test_dataloader = val_dataloader

model = dict(
    test_cfg=dict(
        multi_label=False, score_thr=0.25, nms=dict(iou_threshold=0.45)))
