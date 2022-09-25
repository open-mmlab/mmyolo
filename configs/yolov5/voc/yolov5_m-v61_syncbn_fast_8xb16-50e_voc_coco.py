_base_ = './yolov5_m-v61_syncbn_fast_8xb16-50e_voc.py'

img_scale = (512, 512)
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args={{_base_.file_client_args}}),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]

data_root_coco = 'data/coco_voc/'
dataset_type_coco = 'YOLOv5CocoDataset'
persistent_workers = True
val_batch_size_per_gpu = 1
val_num_workers = 2


# only on Val
batch_shapes_cfg = dict(
    type='BatchShapePolicy',
    batch_size=val_batch_size_per_gpu,
    img_size=img_scale[0],
    size_divisor=32,
    extra_pad_ratio=0.5)

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type_coco,
        data_root=data_root_coco,
        test_mode=True,
        data_prefix=dict(img='test/'),
        ann_file='annotations/test.json',
        pipeline=test_pipeline,
        batch_shapes_cfg=batch_shapes_cfg),
    _delete_=True)

test_dataloader = val_dataloader

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49,
        strict_load=False
    )
]

val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=data_root_coco + 'annotations/test.json',
    metric='bbox',
    _delete_=True)
test_evaluator = val_evaluator

load_from = 'yolov5m.pth'